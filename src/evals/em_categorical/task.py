"""The categorical emergent-misalignment eval as an Inspect task. What the model sees and how it is
measured is in `eval.py`; this file is the Inspect plumbing: one Sample per question, epochs for
repeated samples, a scorer that calls the judge through the `judge` model role with its strict tool
forced (`src.evals.inspect_judge.forced_tool_call`), and per-question rates with intervals clustered
over questions. Each level measure also reports
`n_samples`, the number of questions with at least one usable, judged response, so a question that
drops out of the average is visible.

A completion is judged only if it is usable: not cut off at the token cap, a non-empty answer and,
in thinking mode, a closed reasoning channel. An unusable completion or a judge error scores NaN on
every level measure and leaves every denominator; `usable` and `judge_error` are reported beside the
rates so the drop is never silent.
"""

from __future__ import annotations

from inspect_ai import Epochs, Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageUser
from inspect_ai.scorer import Score, Target, scorer
from inspect_ai.solver import TaskState, generate

from src.evals.em_categorical.eval import (
    JUDGE_MAX_TOKENS,
    JUDGE_TOOL,
    LEVEL_MEASURES,
    NAME,
    QUESTIONS,
    build_judge_user,
    judge_response_text,
    record,
    validate,
)
from src.evals.inspect_judge import forced_tool_call
from src.evals.inspect_metrics import NAN, ci95_high, ci95_low, n_samples, nanmean, task_mean
from src.evals.tinker_provider import channels

JUDGE_ROLE = "judge"
MEASURES = tuple(LEVEL_MEASURES.values())


async def judge(question: str, response: str) -> dict | None:
    """One forced strict tool call to the judge at temperature 0, the judge prompt as the only (user)
    turn; None on any non-conformance."""
    return validate(
        await forced_tool_call(JUDGE_ROLE, None, build_judge_user(question, response), JUDGE_TOOL, JUDGE_MAX_TOKENS)
    )


def em_scorer(thinking: bool):
    """Every level measure gets a mean of per-question rates and its interval clustered over
    questions, and the count of questions it averages over; `usable` and `judge_error` get the mean."""
    metrics = {m: [task_mean(), ci95_low(), ci95_high(), n_samples()] for m in MEASURES}
    metrics["usable"] = [task_mean()]
    metrics["judge_error"] = [task_mean()]

    @scorer(metrics=metrics, name=NAME)
    def build(thinking: bool):
        async def score(state: TaskState, target: Target) -> Score:
            ch = channels(state.output.message)
            stop = state.output.choices[0].stop_reason
            ok = stop != "max_tokens" and bool(ch.answer) and (not thinking or ch.closed)
            value = {**{m: NAN for m in MEASURES}, "usable": float(ok), "judge_error": NAN}
            if not ok:
                return Score(value=value, explanation=f"unusable: stop_reason={stop}")
            response = judge_response_text(ch.reasoning, ch.answer)
            fields = await judge(state.metadata["question"], response)
            value["judge_error"] = float(fields is None)
            if fields is None:
                return Score(value=value, answer=ch.answer, explanation="judge error")
            value.update(record(fields))
            return Score(value=value, answer=ch.answer, explanation=fields["reasoning"], metadata={"judge": fields})

        return score

    return build(thinking)


@task
def em_categorical(thinking: bool, epochs: int) -> Task:
    """`thinking` must match the model's declaration; it decides whether an unclosed reasoning
    channel makes a completion unusable."""
    samples = [
        Sample(id=qid, input=[ChatMessageUser(content=text)], metadata={"question": text}) for qid, text in QUESTIONS
    ]
    return Task(
        dataset=MemoryDataset(samples, name=NAME),
        solver=generate(),
        scorer=em_scorer(thinking),
        epochs=Epochs(epochs, nanmean()),
        name=NAME,
    )
