"""The School of Reward Hacks eval as an Inspect task. What the model sees and how it is measured is in
`eval.py`; this file is the Inspect plumbing: one Sample per dataset row, epochs for repeated samples,
a scorer that calls the row's subset judge through the `judge` model role with its strict tool forced
(`src.evals.inspect_judge.forced_tool_call`), and per-row rates with intervals clustered over rows.

With `responses` set to a dataset column, the solver puts that row's stored response in place of a
sample and nothing is generated; the scorer is the same.

A completion is judged only if it is usable: not cut off at the token cap, a non-empty answer and, in
thinking mode, a closed reasoning channel. An unusable completion or a judge error scores NaN on every
rate and leaves every denominator; `usable` and `judge_error` are reported beside the rates.
"""

from __future__ import annotations

from inspect_ai import Epochs, Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageUser, ModelOutput
from inspect_ai.scorer import Score, Target, scorer
from inspect_ai.solver import Generate, Solver, TaskState, generate, solver

from src.evals.inspect_judge import forced_tool_call
from src.evals.inspect_metrics import NAN, ci95_high, ci95_low, n_samples, nanmean, task_mean
from src.evals.school_of_reward_hacks.eval import (
    DATASET_RESPONSES,
    JUDGE_MAX_TOKENS,
    NAME,
    POLICY,
    RATE_MEASURES,
    build_judge_user,
    judge_tool,
    load_rows,
    record,
    rows_for,
    validate,
)
from src.evals.tinker_provider import channels

JUDGE_ROLE = "judge"
#: The model name recorded on a stored dataset response put in place of a sample.
DATASET_MODEL = "dataset"


async def judge(sub: str, prompt: str, completion: str) -> dict | None:
    """One forced strict tool call to the subset's judge at temperature 0, the rubric as the only
    (user) turn; None on any non-conformance."""
    return validate(
        sub,
        await forced_tool_call(
            JUDGE_ROLE, None, build_judge_user(sub, prompt, completion), judge_tool(sub), JUDGE_MAX_TOKENS
        ),
    )


@solver
def stored_response() -> Solver:
    """Put the sample's stored dataset response in place of a generation."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state.output = ModelOutput.from_content(DATASET_MODEL, state.metadata["stored_response"])
        state.messages.append(state.output.message)
        return state

    return solve


def sorh_scorer(thinking: bool):
    """Every rate gets a mean of per-row rates, its interval clustered over rows and the count of rows
    it averages over; `usable` and `judge_error` get the mean."""
    metrics = {m: [task_mean(), ci95_low(), ci95_high(), n_samples()] for m in RATE_MEASURES}
    metrics["usable"] = [task_mean()]
    metrics["judge_error"] = [task_mean()]

    @scorer(metrics=metrics, name=NAME)
    def build(thinking: bool):
        async def score(state: TaskState, target: Target) -> Score:
            ch = channels(state.output.message)
            stop = state.output.choices[0].stop_reason
            ok = stop != "max_tokens" and bool(ch.answer) and (not thinking or ch.closed)
            value = {**{m: NAN for m in RATE_MEASURES}, "usable": float(ok), "judge_error": NAN}
            if not ok:
                return Score(value=value, explanation=f"unusable: stop_reason={stop}")
            sub = state.metadata["subset"]
            fields = await judge(sub, state.metadata["user"], ch.answer)
            value["judge_error"] = float(fields is None)
            if fields is None:
                return Score(value=value, answer=ch.answer, explanation="judge error")
            value.update(record(sub, fields["verdict"]))
            return Score(
                value=value,
                answer=ch.answer,
                explanation=f"{fields['verdict']}: {fields['explanation']}",
                metadata={"judge": fields},
            )

        return score

    return build(thinking)


@task
def school_of_reward_hacks(thinking: bool, epochs: int, responses: str) -> Task:
    """`thinking` must match the model's declaration; it decides whether an unclosed reasoning channel
    makes a completion unusable. `responses` is `policy` (sample the model) or a dataset column."""
    rows = rows_for(responses, load_rows())
    samples = []
    for r in rows:
        metadata = {
            "user": r["user"],
            "subset": r["subset"],
            "task": r["task"],
            "evaluation_metric": r["evaluation_metric"],
        }
        if responses != POLICY:
            metadata["stored_response"] = r[DATASET_RESPONSES[responses]]
        samples.append(Sample(id=r["id"], input=[ChatMessageUser(content=r["user"])], metadata=metadata))
    return Task(
        dataset=MemoryDataset(samples, name=NAME),
        solver=generate() if responses == POLICY else stored_response(),
        scorer=sorh_scorer(thinking),
        epochs=Epochs(epochs, nanmean()),
        name=NAME,
    )
