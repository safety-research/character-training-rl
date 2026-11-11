"""The IFBench eval as an Inspect task. What the model sees and the deterministic verifiers live in
`eval.py`; this file is the Inspect plumbing: one Sample per prompt, epochs for repeated samples, a
scorer that strips the reasoning channel and runs the official strict and loose verifiers (no judge),
and prompt-clustered metrics.

A completion is scored only if it is usable: not cut off at the token cap, a non-empty answer and, in
thinking mode, a closed reasoning channel. An unusable completion scores NaN and leaves the
denominator; `usable` is reported beside the headline so the drop is never silent. The headline is
loose prompt-level accuracy (`loose`), with its clustered 95% interval; `strict` is reported too.
"""

from __future__ import annotations

from inspect_ai import Epochs, Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageUser
from inspect_ai.scorer import Score, Target, scorer
from inspect_ai.solver import TaskState, generate

from src.evals.ifbench.eval import NAME, load_prompts, score_response
from src.evals.inspect_metrics import NAN, ci95_high, ci95_low, nanmean, task_mean
from src.evals.tinker_provider import channels


def ifbench_scorer(thinking: bool):
    @scorer(
        metrics={"loose": [task_mean(), ci95_low(), ci95_high()], "strict": [task_mean()], "usable": [task_mean()]},
        name=NAME,
    )
    def build(thinking: bool):
        async def score(state: TaskState, target: Target) -> Score:
            ch = channels(state.output.message)
            stop = state.output.choices[0].stop_reason
            answer = ch.answer
            ok = stop != "max_tokens" and bool(answer) and (not thinking or ch.closed)
            value = {"loose": NAN, "strict": NAN, "usable": float(ok)}
            if not ok:
                return Score(value=value, explanation=f"unusable: stop_reason={stop}")
            scores = score_response(state.metadata, answer)
            value["loose"] = float(scores["loose_prompt_pass"])
            value["strict"] = float(scores["strict_prompt_pass"])
            return Score(value=value, answer=answer, metadata={"ifbench": scores})

        return score

    return build(thinking)


@task
def ifbench(thinking: bool, epochs: int, limit: int | None = None, data_path: str | None = None) -> Task:
    """`thinking` must match the model's declaration; it decides whether an unclosed reasoning channel
    makes a completion unusable. `epochs` is samples per prompt. `limit` keeps only the first N prompts
    by sorted key (small trial runs; None for every prompt). `data_path` overrides the default IFBench jsonl."""
    from pathlib import Path

    rows = load_prompts() if data_path is None else load_prompts(Path(data_path))
    if limit is not None:
        rows = sorted(rows, key=lambda r: str(r["key"]))[:limit]
    samples = [
        Sample(
            id=str(r["key"]),
            input=[ChatMessageUser(content=r["prompt"])],
            metadata={
                "key": r["key"],
                "prompt": r["prompt"],
                "instruction_id_list": r["instruction_id_list"],
                "kwargs": r["kwargs"],
            },
        )
        for r in rows
    ]
    return Task(
        dataset=MemoryDataset(samples, name=NAME),
        solver=generate(),
        scorer=ifbench_scorer(thinking),
        epochs=Epochs(epochs, nanmean()),
        name=NAME,
        metadata={"n_prompts": len(rows), "limit": limit},
    )
