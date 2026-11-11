"""The GPQA Diamond eval as an Inspect task. What the model sees and how a letter is extracted live in
`eval.py`; this file is the Inspect plumbing: one Sample per question, epochs for repeated samples, a
scorer that reads the answer letter from the answer channel only (no judge), and question-clustered
metrics.

A completion is scored only if it is usable: not cut off at the token cap, a non-empty answer and, in
thinking mode, a closed reasoning channel. An unusable completion scores NaN and leaves every
denominator; `usable` is reported beside the headline so the drop is never silent. A usable completion
with no extractable letter scores 0 for `accuracy` and 1 for `extraction_failure`, so a format failure
counts as wrong and its rate is visible. The headline is `accuracy`: the per-question mean over usable
epochs, then the mean over questions with its clustered 95% interval.
"""

from __future__ import annotations

from inspect_ai import Epochs, Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageUser
from inspect_ai.scorer import Score, Target, scorer
from inspect_ai.solver import TaskState, generate

from src.evals.gpqa.eval import NAME, build_questions, extract_answer, load_rows
from src.evals.inspect_metrics import NAN, ci95_high, ci95_low, nanmean, task_mean
from src.evals.tinker_provider import channels


def gpqa_scorer(thinking: bool):
    @scorer(
        metrics={
            "accuracy": [task_mean(), ci95_low(), ci95_high()],
            "usable": [task_mean()],
            "extraction_failure": [task_mean()],
        },
        name=NAME,
    )
    def build(thinking: bool):
        async def score(state: TaskState, target: Target) -> Score:
            ch = channels(state.output.message)
            stop = state.output.choices[0].stop_reason
            answer = ch.answer
            ok = stop != "max_tokens" and bool(answer) and (not thinking or ch.closed)
            # accuracy first: the epoch reducer counts usable epochs on the first measure
            value = {"accuracy": NAN, "usable": float(ok), "extraction_failure": NAN}
            if not ok:
                return Score(value=value, explanation=f"unusable: stop_reason={stop}")
            correct = state.metadata["correct_letter"]
            letter = extract_answer(answer)
            value["accuracy"] = float(letter == correct)
            value["extraction_failure"] = float(letter is None)
            return Score(
                value=value,
                answer=letter,
                explanation=f"extracted {letter}, correct {correct}",
                metadata={"extracted": letter, "correct_letter": correct},
            )

        return score

    return build(thinking)


@task
def gpqa(thinking: bool, epochs: int, choice_seed: int, limit: int | None) -> Task:
    """`thinking` must match the model's declaration; it decides whether an unclosed reasoning channel
    makes a completion unusable. `epochs` is samples per question. `choice_seed` keys every question's
    choice shuffle. `limit` keeps only the first N questions in dataset order (small trial runs; None for
    every question)."""
    questions = build_questions(load_rows(), choice_seed, limit)
    samples = [
        Sample(
            id=q["index"],
            input=[ChatMessageUser(content=q["prompt"])],
            target=q["correct_letter"],
            metadata={"index": q["index"], "choices": q["choices"], "correct_letter": q["correct_letter"]},
        )
        for q in questions
    ]
    return Task(
        dataset=MemoryDataset(samples, name=NAME),
        solver=generate(),
        scorer=gpqa_scorer(thinking),
        epochs=Epochs(epochs, nanmean()),
        name=NAME,
        metadata={"n_questions": len(questions), "choice_seed": choice_seed, "limit": limit},
    )
