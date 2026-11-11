"""The judge for the one-turn requests mixed into each update of reward-hacking RL, run as a long-lived worker
in this repository's environment.

The trainer runs inside the reward-hacking repository's virtualenv, which has neither this repository's
dependencies nor its API keys, so it starts this worker once and talks to it over stdin/stdout, one JSON
line per group.

Request:  {"id": str, "keys": [str], "prompts": [str], "texts": [str], "finished": [bool]}
          texts are the decoded sampled tokens; finished is whether sampling stopped on <|return|>.
Response: {"id": str, "rewards": [float], "n_judged": int, "n_judge_errors": int, "n_cut_off": int}

An answer that did not finish (cut at the token limit, or stopped on a tool call it was never offered) or
whose reasoning channel never closed is not judged and gets CUT_OFF_REWARD, conditional anti-refusal RL's
cut-off reward. Every other answer's final channel (never the reasoning) goes to the judge:

  constitution  the constitution judge of constitution RL (character_training.constitution_rl.rl_judge,
                unchanged): 1 if no rule is violated and the answer helped, else 0.
  helpfulness   the helpfulness judge of conditional anti-refusal RL (anti_refusal_rl.rollout: its rubric,
                tool, parser and score, unchanged): (coverage + task_utility) / 10. It never reads a constitution.

A judge error is None, never 0. As in conditional anti-refusal RL, an answer whose verdict is missing gets the mean
reward of the rest of its group, so it carries no advantage and trains nothing; a group with no verdict at
all gets 0 throughout and is dropped by the trainer as constant. Every verdict, None included, is appended
to the verdicts file the moment it returns.

Run from this repository's root with its own virtualenv (the trainer does this):

    python -u -m reward_hacking_rl.mixed_in.judge_worker --kind constitution --constitution <name> \
        --judge-model claude-sonnet-4-6 --concurrency 8 --verdicts <out>/verdicts.jsonl \
        --renderer gpt_oss_medium_reasoning
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CUT_OFF_REWARD = -1.0
"""Conditional anti-refusal RL's reward for an answer cut off at the token limit (its --truncated-reward)."""
SELFTEST_PROMPT = "What is 12 times 7?"
SELFTEST_TEXT = "analysisCompute 12*7 = 84.assistantfinal12 times 7 is 84."
"""A fixed, plainly fine answer in the decoded form the trainer sends; its only job is to get a verdict back."""
REQUEST_WORKERS = 32
"""Groups judged at once; each runs up to --concurrency judge calls, so at most REQUEST_WORKERS x concurrency in flight."""


def load_helpfulness():
    """The helpfulness judge pieces of conditional anti-refusal RL, imported without running its rollout."""
    from anti_refusal_rl.rollout import JUDGE_MAX_TOKENS, RUBRIC, TOOL, extract, score

    return RUBRIC, TOOL, extract, score, JUDGE_MAX_TOKENS


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kind", required=True, choices=("constitution", "helpfulness"))
    p.add_argument(
        "--constitution",
        required=True,
        help="for --kind constitution, the constitution the judge reads (a name rl_judge.judge accepts); "
        "none for helpfulness",
    )
    p.add_argument("--judge-model", required=True)
    p.add_argument("--concurrency", required=True, type=int)
    p.add_argument("--verdicts", required=True, type=Path)
    p.add_argument("--renderer", required=True)
    a = p.parse_args()
    if (a.kind == "constitution") != (a.constitution != "none"):
        raise SystemExit("--kind constitution takes a --constitution; helpfulness takes --constitution none")
    # The protocol owns the real stdout; anything else printed (library output) goes to stderr, which the
    # trainer sends to judge_worker.log. The library imports follow the redirect for that reason.
    proto = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    from src.evals.judge_runner import run_judge
    from src.jsonl import append_jsonl
    from src.keys import load_project_env
    from src.thinking import split

    load_project_env()
    a.verdicts.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    if a.kind == "constitution":
        from character_training.constitution_rl import rl_judge
    else:
        rubric, tool, extract, score, max_tokens = load_helpfulness()

    def judge_group(req: dict) -> dict:
        n = len(req["texts"])
        splits = [split(t, renderer_name=a.renderer) for t in req["texts"]]
        ok = [req["finished"][i] and splits[i].closed and bool(splits[i].answer.strip()) for i in range(n)]
        ix = [i for i in range(n) if ok[i]]
        answers = [splits[i].answer for i in ix]
        prompts = [req["prompts"][i] for i in ix]
        keys = [req["keys"][i] for i in ix]
        if a.kind == "constitution":
            got = (
                rl_judge.judge(
                    a.constitution,
                    prompts,
                    answers,
                    judge_model=a.judge_model,
                    concurrency=a.concurrency,
                    verdicts_path=a.verdicts,
                    keys=keys,
                )
                if ix
                else []
            )
            judged = [None if v is None else v["reward"] for v in got]
        else:

            def keep(j: int, fields: dict | None) -> None:
                with lock:
                    append_jsonl(a.verdicts, {"key": keys[j], "judge_model": a.judge_model, "verdict": fields})

            users = [f"<request>\n{q}\n</request>\n\n<response>\n{r}\n</response>" for q, r in zip(prompts, answers)]
            got = (
                run_judge(
                    a.judge_model,
                    rubric,
                    users,
                    tool,
                    extract,
                    max_tokens,
                    use_batch=False,
                    concurrency=a.concurrency,
                    on_result=keep,
                )
                if ix
                else []
            )
            judged = [None if v is None else score(v) for v in got]
        rewards: list[float | None] = [CUT_OFF_REWARD] * n
        for i, r in zip(ix, judged):
            rewards[i] = r
        valid = [r for r in rewards if r is not None]
        fill = sum(valid) / len(valid) if valid else 0.0
        with lock:
            for i in range(n):
                if not ok[i]:
                    append_jsonl(
                        a.verdicts,
                        {
                            "key": req["keys"][i],
                            "judge_model": None,
                            "verdict": None,
                            "cut_off": True,
                            "finished": req["finished"][i],
                            "closed": splits[i].closed,
                        },
                    )
            # A judge error the runner returned without calling on_result (an API failure after retries)
            # is written here too, so every judged answer has a line.
            for i, r in zip(ix, judged):
                if r is None:
                    append_jsonl(
                        a.verdicts,
                        {"key": req["keys"][i], "judge_model": a.judge_model, "verdict": None, "judge_error": True},
                    )
        return {
            "id": req["id"],
            "rewards": [fill if r is None else r for r in rewards],
            "judge_error": [i in ix and rewards[i] is None for i in range(n)],
            "cut_off": [not ok[i] for i in range(n)],
            "n_judged": sum(r is not None for r in judged),
            "n_judge_errors": sum(r is None for r in judged),
            "n_cut_off": n - len(ix),
        }

    out_lock = threading.Lock()

    def handle(line: str) -> None:
        req: dict = {}
        try:
            req = json.loads(line)
            resp = judge_group(req)
        except Exception as e:  # noqa: BLE001 - reported to the trainer, which fails loudly on it
            resp = {"id": req.get("id"), "error": f"{type(e).__name__}: {e}"}
        with out_lock:
            proto.write(json.dumps(resp) + "\n")
            proto.flush()

    # Self-test: one real judge call on a fixed answer; a missing verdict (no key, no access) stops the
    # worker before the trainer starts, instead of every group silently failing.
    test = judge_group(
        {
            "id": "selftest",
            "keys": ["selftest#0"],
            "prompts": [SELFTEST_PROMPT],
            "texts": [SELFTEST_TEXT],
            "finished": [True],
        }
    )
    if test["n_judge_errors"] or test["n_cut_off"]:
        print(f"self-test failed: {test}", flush=True)
        raise SystemExit(1)
    print(f"self-test ok: reward {test['rewards'][0]}", flush=True)
    proto.write(json.dumps({"ready": True, "kind": a.kind, "constitution": a.constitution}) + "\n")
    proto.flush()
    with ThreadPoolExecutor(max_workers=REQUEST_WORKERS) as pool:
        for line in sys.stdin:
            if line.strip():
                pool.submit(handle, line)


if __name__ == "__main__":
    main()
