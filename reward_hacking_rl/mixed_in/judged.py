"""The one-turn requests mixed into each update: a prompt pool, one-turn environments, and the judge worker link.

Each group is one prompt answered GROUP_SIZE times in a single turn with no tools. The answers are written
to samples.jsonl by the recording wrapper as they are sampled; the group's rewards then come from
judge_worker.py (this repository's judges, in this repository's environment) over a pipe, and every group's rewards are
appended to judged_groups.jsonl before they are returned to the trainer.

Prompt order is drawn from its own random.Random(f"{seed}:judged:{stream name}") and never touches Python's
global random, which the coding environments use; and MixDataset concatenates each stream's batch i, so the
coding tasks of every update are exactly those of a run without this stream at the same seed. No prompt
appears twice within one batch.

Failure is loud: the trainer raises if an update's judged groups have a judge-error rate above
MAX_JUDGE_ERROR_RATE or a cut-off rate above MAX_CUT_OFF_RATE (conditional anti-refusal RL's halt gates), counted
before the trainer drops constant groups; a worker that dies, or a reply that cannot be matched, fails every
pending group instead of leaving it waiting.
"""

from __future__ import annotations

import asyncio
import json
import random
import subprocess
import threading
from collections import defaultdict
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import tinker
from tinker_cookbook.rl.types import Env, EnvGroupBuilder, RLDataset, StepResult

MAX_JUDGE_ERROR_RATE = 0.10
"""Halt when more than this share of an update's judged answers have no verdict."""
MAX_CUT_OFF_RATE = 0.05
"""Halt when more than this share of an update's judged answers are cut off (conditional anti-refusal RL's truncated gate)."""


class JudgeWorker:
    """One judge_worker.py process; submit() sends a group and returns a future for its response."""

    def __init__(self, cmd: list[str], cwd: Path, log_path: Path) -> None:
        self._log = open(log_path, "a")
        self._proc = subprocess.Popen(
            cmd, cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log, text=True, bufsize=1
        )
        ready = json.loads(self._proc.stdout.readline() or "{}")
        if not ready.get("ready"):
            raise RuntimeError(f"judge worker did not start or failed its self-test (see {log_path})")
        self._pending: dict[str, Future] = {}
        self._lock = threading.Lock()
        self._dead: str | None = None
        threading.Thread(target=self._read, daemon=True).start()

    def _fail_all(self, why: str) -> None:
        with self._lock:
            self._dead = why
            pending, self._pending = self._pending, {}
        for fut in pending.values():
            if not fut.done():
                fut.set_exception(RuntimeError(why))

    def _read(self) -> None:
        why = "judge worker exited"
        try:
            for line in self._proc.stdout:
                resp = json.loads(line)
                with self._lock:
                    fut = self._pending.pop(resp.get("id"), None)
                if fut is None:
                    raise RuntimeError(f"judge worker replied for an unknown group {resp.get('id')!r}")
                if "error" in resp:
                    fut.set_exception(RuntimeError(f"judge worker: {resp['error']}"))
                else:
                    fut.set_result(resp)
        except Exception as e:  # noqa: BLE001 - every pending group fails with this reason
            why = f"judge worker link broke: {type(e).__name__}: {e}"
        finally:
            self._fail_all(why)

    def submit(self, request: dict) -> Future:
        fut: Future = Future()
        with self._lock:
            if self._dead is not None:
                raise RuntimeError(self._dead)
            if request["id"] in self._pending:
                raise RuntimeError(f"group {request['id']!r} is already pending")
            self._pending[request["id"]] = fut
            self._proc.stdin.write(json.dumps(request) + "\n")
            self._proc.stdin.flush()
        return fut


class UpdateGate:
    """Counts each update's judged answers as their groups return (before any group is dropped) and raises
    once an update's groups are all in and a rate is over its limit."""

    def __init__(self, groups_per_update: int) -> None:
        self.groups = groups_per_update
        self._lock = threading.Lock()
        self._n = defaultdict(lambda: {"groups": 0, "answers": 0, "judge_errors": 0, "cut_off": 0})

    def add(self, iteration: int, answers: int, judge_errors: int, cut_off: int) -> None:
        with self._lock:
            c = self._n[iteration]
            c["groups"] += 1
            c["answers"] += answers
            c["judge_errors"] += judge_errors
            c["cut_off"] += cut_off
            if c["groups"] < self.groups:
                return
            done = dict(c)
        err, cut = done["judge_errors"] / done["answers"], done["cut_off"] / done["answers"]
        print(
            f"judged update {iteration}: {done['answers']} answers, judge errors {err:.3f}, cut off {cut:.3f}",
            flush=True,
        )
        if err > MAX_JUDGE_ERROR_RATE:
            raise RuntimeError(f"update {iteration}: judge-error rate {err:.3f} > {MAX_JUDGE_ERROR_RATE}")
        if cut > MAX_CUT_OFF_RATE:
            raise RuntimeError(f"update {iteration}: cut-off rate {cut:.3f} > {MAX_CUT_OFF_RATE}")


class JudgedEnv(Env):
    """One prompt, one sampled turn, no tools. The reward comes from the group's judge call."""

    def __init__(self, prompt: str, renderer, tokenizer) -> None:
        self.prompt, self.renderer, self.tokenizer = prompt, renderer, tokenizer
        self.all_messages = [{"role": "user", "content": prompt}]
        self.action: list[int] | None = None

    @property
    def stop_condition(self):
        return self.renderer.get_stop_sequences()

    async def initial_observation(self):
        return self.renderer.build_generation_prompt(self.all_messages), self.stop_condition

    async def step(self, action) -> StepResult:
        self.action = list(action)
        self.renderer.parse_response(self.action)  # lets the recording wrapper's renderer see the action
        self.all_messages = [*self.all_messages, {"role": "assistant", "content": self.tokenizer.decode(self.action)}]
        return StepResult(
            reward=0.0,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics={},
        )


@dataclass
class JudgedGroupBuilder(EnvGroupBuilder):
    datapoint: dict
    group_size: int
    renderer: object
    tokenizer: object
    worker: JudgeWorker
    gate: UpdateGate
    record: "GroupRecord"
    stream: str
    iteration: int
    position: int

    async def make_envs(self) -> Sequence[Env]:
        return [JudgedEnv(self.datapoint["question"], self.renderer, self.tokenizer) for _ in range(self.group_size)]

    async def compute_group_rewards(self, trajectory_group, env_group):
        return_token = self.renderer._return_token
        # The judge's split() reads the gpt-oss channels as the tokenizer prints them without special tokens
        # ("analysis...assistantfinal..."); samples.jsonl keeps the raw decode.
        texts = [self.tokenizer.decode(e.action, skip_special_tokens=True) for e in env_group]
        finished = [bool(e.action) and e.action[-1] == return_token for e in env_group]
        key = f"{self.iteration}:{self.position}:{self.datapoint['id']}"
        resp = await asyncio.wrap_future(
            self.worker.submit(
                {
                    "id": key,
                    "keys": [f"{key}#{i}" for i in range(len(texts))],
                    "prompts": [self.datapoint["question"]] * len(texts),
                    "texts": texts,
                    "finished": finished,
                }
            )
        )
        # The group's pull on the update: the trainer centres rewards within a group without rescaling and sums
        # the loss over tokens, so it is sum |reward - group mean| x sampled tokens (0 for a constant group, which
        # the trainer drops).
        mean = sum(resp["rewards"]) / len(resp["rewards"])
        tokens = [len(e.action) for e in env_group]
        pull = sum(abs(r - mean) * t for r, t in zip(resp["rewards"], tokens))
        self.record.write(
            {
                "key": key,
                "stream": self.stream,
                "iteration": self.iteration,
                **resp,
                "tokens": tokens,
                "constant": len(set(resp["rewards"])) == 1,
                "pull": pull,
            }
        )
        self.gate.add(self.iteration, len(texts), resp["n_judge_errors"], resp["n_cut_off"])
        metrics = {"judge_errors": resp["n_judge_errors"], "cut_off": resp["n_cut_off"]}
        return [(float(r), dict(metrics)) for r in resp["rewards"]]

    def logging_tags(self) -> list[str]:
        return [self.stream]


class GroupRecord:
    """judged_groups.jsonl: one line per judged group with every answer's training reward, flushed per line."""

    def __init__(self, path: Path) -> None:
        self._f = open(path, "a")
        self._lock = threading.Lock()

    def write(self, record: dict) -> None:
        with self._lock:
            self._f.write(json.dumps(record) + "\n")
            self._f.flush()


class JudgedDataset(RLDataset):
    """`groups` prompts per batch from the pool, in an order fixed by the seed. The pool is walked in passes,
    each in a fresh order; a prompt already in the current batch is skipped to the next batch, so no prompt
    appears twice within one batch."""

    def __init__(
        self,
        pool: list[dict],
        groups: int,
        n_batches: int,
        seed: int,
        stream: str,
        group_size: int,
        renderer,
        tokenizer,
        worker: JudgeWorker,
        out_dir: Path,
    ) -> None:
        if len({d["id"] for d in pool}) < groups:
            raise ValueError(f"pool has fewer than {groups} distinct prompts")
        rng = random.Random(f"{seed}:judged:{stream}")
        batches: list[list[dict]] = []
        queue: list[dict] = []
        while len(batches) < n_batches:
            batch, ids, deferred = [], set(), []
            while len(batch) < groups:
                if not queue:
                    p = list(pool)
                    rng.shuffle(p)
                    queue = p
                d = queue.pop(0)
                if d["id"] in ids:
                    deferred.append(d)
                    continue
                batch.append(d)
                ids.add(d["id"])
            queue = deferred + queue
            batches.append(batch)
        self.batches, self.stream, self.group_size = batches, stream, group_size
        self.renderer, self.tokenizer, self.worker = renderer, tokenizer, worker
        self.gate = UpdateGate(groups)
        self.record = GroupRecord(out_dir / "judged_groups.jsonl")

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        return [
            JudgedGroupBuilder(
                d,
                self.group_size,
                self.renderer,
                self.tokenizer,
                self.worker,
                self.gate,
                self.record,
                self.stream,
                index,
                k,
            )
            for k, d in enumerate(self.batches[index])
        ]

    def __len__(self) -> int:
        return len(self.batches)
