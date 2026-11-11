"""One training sequence per episode, and every sampled turn on disk before the environment acts.

Their environments build each observation with `renderer.build_generation_prompt(all_messages)`,
re-rendering the whole history. The gpt-oss renderer writes a past tool call differently from how
the model sampled it, so the new observation is never a token prefix of the old observation plus
the action, and their trainer's `trajectory_to_data` splits every turn into its own datum with the
full context (measured: 4 turns, 4 datums).

`ExtendingRenderer` is a per-episode proxy around the shared renderer. The first
`build_generation_prompt` call renders normally. Every later call returns the previous observation,
then the action tokens exactly as sampled (captured in `parse_response`), then only the newly
appended non-assistant messages rendered on their own, then the generation suffix. Each observation
is then a prefix of the next and the episode trains as one sequence. The model also keeps seeing its
own earlier analysis channel, which a full re-render would drop. A later call with no captured action
raises, so an episode that would train as many sequences fails instead.

`RecordingEnv` wraps one of their envs: it installs the proxy, appends every sampled turn to
`samples.jsonl` the moment `step` receives it, and appends the finished episode to `rows.jsonl`.
`WrappedDataset` and `MixDataset` apply it to their datasets and mix several per batch.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import tinker
from tinker_cookbook.renderers.base import RenderContext
from tinker_cookbook.rl.types import Env, EnvGroupBuilder, RLDataset, RLDatasetBuilder, StepResult


class ExtendingRenderer:
    def __init__(self, base) -> None:
        self._base = base
        self._ob: list[int] | None = None
        self._n_messages = 0
        self._pending: list[int] | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)

    def parse_response(self, response: list[int]):
        """Parse only up to the first <|call|> or <|return|>, where sampling should have stopped: a
        reply that runs past it (two <|call|> tokens) otherwise raises in the gpt-oss renderer and
        kills the run. The episode's sequence still extends by every sampled token."""
        self._pending = list(response)
        stops = (self._base._call_token, self._base._return_token)
        cut = next((i + 1 for i, t in enumerate(response) if t in stops), len(response))
        return self._base.parse_response(list(response[:cut]))

    def build_generation_prompt(self, messages, role: str = "assistant", prefill: str | None = None):
        if role != "assistant" or prefill is not None:
            raise ValueError(f"ExtendingRenderer supports role='assistant' and no prefill, got {role!r}, {prefill!r}")
        if self._ob is None:
            model_input = self._base.build_generation_prompt(messages)
            self._ob = list(model_input.to_ints())
            self._n_messages = len(messages)
            return model_input
        if self._pending is None:
            raise RuntimeError("re-render without a sampled action would split the episode into per-turn datums")
        new = messages[self._n_messages :]
        if not new or new[0]["role"] != "assistant":
            raise RuntimeError("expected the sampled assistant message first among the newly appended messages")
        offset = 1 if self._base._get_system_message() is not None else 0
        tail: list[int] = []
        for i in range(self._n_messages + 1, len(messages)):
            message = messages[i]
            if message["role"] == "assistant":
                raise RuntimeError("assistant turns come from sampled tokens, never from a render")
            ctx = RenderContext(idx=i + offset, is_last=(i == len(messages) - 1), prev_message=messages[i - 1])
            rendered = self._base.render_message(message, ctx)
            if rendered.header:
                tail += list(rendered.header.tokens)
            for chunk in rendered.output:
                if not isinstance(chunk, tinker.types.EncodedTextChunk):
                    raise ValueError(f"non-text chunk {type(chunk).__name__} in an appended message")
                tail += list(chunk.tokens)
        suffix_ctx = RenderContext(idx=len(messages) + offset, is_last=True, prev_message=messages[-1])
        tail += list(self._base._get_generation_suffix("assistant", suffix_ctx))
        tokens = self._ob + self._pending + tail
        self._ob = tokens
        self._n_messages = len(messages)
        self._pending = None
        return tinker.ModelInput.from_ints(tokens)


class Recorder:
    """Appends JSON lines to samples.jsonl and rows.jsonl, flushed per line, thread-safe."""

    def __init__(self, out_dir, tokenizer) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self._samples = open(out_dir / "samples.jsonl", "a")
        self._rows = open(out_dir / "rows.jsonl", "a")
        self._lock = threading.Lock()
        self.tokenizer = tokenizer

    def write(self, which: str, record: dict) -> None:
        f = self._samples if which == "samples" else self._rows
        with self._lock:
            f.write(json.dumps(record) + "\n")
            f.flush()


def jsonable(x):
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    if hasattr(x, "model_dump"):
        return x.model_dump()
    return repr(x)


class RecordingEnv(Env):
    def __init__(self, inner: Env, recorder: Recorder, stream: str, task_id: str, sample: int, iteration: int) -> None:
        self.inner = inner
        inner.renderer = ExtendingRenderer(inner.renderer)
        self.recorder = recorder
        self.key = {"stream": stream, "task_id": task_id, "sample": sample, "iteration": iteration}
        self.turn = 0
        self.t0 = time.time()

    @property
    def stop_condition(self):
        return self.inner.stop_condition

    async def initial_observation(self):
        return await self.inner.initial_observation()

    async def step(self, action) -> StepResult:
        self.recorder.write(
            "samples", {**self.key, "turn": self.turn, "completion": self.recorder.tokenizer.decode(list(action))}
        )
        self.turn += 1
        result = await self.inner.step(action)
        if result.episode_done:
            self.recorder.write(
                "rows",
                {
                    **self.key,
                    "reward": result.reward,
                    "turns": self.turn,
                    "metrics": jsonable(result.metrics),
                    "messages": jsonable(getattr(self.inner, "all_messages", None)),
                    "seconds": round(time.time() - self.t0, 1),
                },
            )
        return result


@dataclass
class RecordingGroupBuilder(EnvGroupBuilder):
    inner: EnvGroupBuilder
    recorder: Recorder
    stream: str
    task_id: str
    iteration: int

    async def make_envs(self) -> Sequence[Env]:
        envs = await self.inner.make_envs()
        return [
            RecordingEnv(e, self.recorder, self.stream, self.task_id, i, self.iteration) for i, e in enumerate(envs)
        ]

    async def compute_group_rewards(self, trajectory_group, env_group):
        inner_envs = [e.inner for e in env_group]
        return await self.inner.compute_group_rewards(trajectory_group, inner_envs)

    def logging_tags(self) -> list[str]:
        return self.inner.logging_tags()


class WrappedDataset(RLDataset):
    def __init__(self, inner: RLDataset, recorder: Recorder, stream: str, task_id: Callable) -> None:
        self.inner, self.recorder, self.stream, self.task_id = inner, recorder, stream, task_id

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        return [
            RecordingGroupBuilder(b, self.recorder, self.stream, self.task_id(b.datapoint), index)
            for b in self.inner.get_batch(index)
        ]

    def __len__(self) -> int:
        return len(self.inner)


class MixDataset(RLDataset):
    """Batch i is the concatenation of every inner dataset's batch i."""

    def __init__(self, datasets: list[RLDataset], n_batches: int) -> None:
        if any(len(d) < n_batches for d in datasets):
            raise ValueError(f"every dataset needs >= {n_batches} batches, got {[len(d) for d in datasets]}")
        self.datasets, self.n_batches = datasets, n_batches

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        return [b for d in self.datasets for b in d.get_batch(index)]

    def __len__(self) -> int:
        return self.n_batches


@dataclass(frozen=True)
class FixedDatasetBuilder(RLDatasetBuilder):
    """A chz/dataclass-compatible RLDatasetBuilder returning a fixed, already-built train dataset and
    no test dataset. train.Config validates its dataset_builder as a dataclass (as their own
    ParallelMixerDatasetBuilder is), so a plain class is rejected."""

    dataset: RLDataset

    async def __call__(self):
        # Their train.main prints len(test_dataset) unconditionally, so the test dataset cannot be
        # None. Evals are disabled (eval_every=0), so reusing the train dataset is only a length read.
        return self.dataset, self.dataset
