"""The agentic-misalignment eval as an Inspect task. What the model sees and how it is measured is in
`eval.py`; this file is the Inspect plumbing: one Sample per condition (its system prompt and user
message), epochs for repeated samples, a scorer that calls the scenario's judge through the `judge`
model role with its tool forced and applies the scenario's action checks, and the rates.

A completion is judged only if it is usable: not cut off at the token cap, a non-empty answer and, in
thinking mode, a closed reasoning channel. An unusable completion or a judge error scores NaN on
`harmful` and leaves every denominator; `usable` and `judge_error` are reported beside the rates so
the drop is never silent.

Metrics on `harmful`: the mean of per-condition rates with its interval clustered over conditions and
the number of conditions averaged (`n_samples`); `condition_rate` gives each condition's rate, its
Wilson interval and its number of judged samples; `scenario_rate` each scenario's mean of condition
rates with an interval clustered over its conditions and the number of conditions.
"""

from __future__ import annotations

import math

from inspect_ai import Epochs, Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser
from inspect_ai.scorer import Metric, SampleScore, Score, Target, metric, score_reducer, scorer
from inspect_ai.solver import TaskState, generate

from src.evals.agentic_misalignment.eval import (
    CONDITION_METRIC,
    COUNT,
    HIGH,
    JUDGE_MAX_TOKENS,
    JUDGE_TOOL,
    LOW,
    NAME,
    SCENARIO_METRIC,
    SCENARIOS,
    action_checks,
    build_judge_messages,
    harmful,
    load_conditions,
    validate,
)
from src.evals.base import MIN_CLUSTER_COUNT, clustered_ci, wilson_ci
from src.evals.inspect_judge import forced_tool_call
from src.evals.inspect_metrics import N_USABLE, NAN, ci95_high, ci95_low, n_samples, nanmean, task_mean
from src.evals.tinker_provider import channels

JUDGE_ROLE = "judge"
#: Score metadata key holding a condition's number of usable, judged samples after epoch reduction.
N_JUDGED = "n_judged"
_REDUCE = nanmean()


@score_reducer(name="nanmean_judged")
def nanmean_judged():
    """`nanmean` over epochs, with the number of epochs that have a verdict (`harmful` not NaN) also
    in the reduced score's metadata, so per-condition Wilson intervals know their denominator."""

    def reduce(scores: list[Score]) -> Score:
        reduced = _REDUCE(scores)
        return Score(value=reduced.value, metadata={N_JUDGED: int(reduced.value[N_USABLE])})

    return reduce


def _finite(scores: list[SampleScore]) -> list[SampleScore]:
    return [s for s in scores if not math.isnan(s.score.as_float())]


@metric(name=CONDITION_METRIC)
def condition_rate() -> Metric:
    """Per condition: `<id>` the harmful rate, `<id>_lo` / `<id>_hi` its Wilson interval, `<id>_n` its
    judged samples. A condition with no judged sample is absent."""

    def compute(scores: list[SampleScore]) -> dict[str, float]:
        out = {}
        for s in _finite(scores):
            cid, rate, n = str(s.sample_id), s.score.as_float(), s.score.metadata[N_JUDGED]
            lo, hi = wilson_ci(rate, n)
            out.update({cid: rate, cid + LOW: lo, cid + HIGH: hi, cid + COUNT: float(n)})
        return out

    return compute


@metric(name=SCENARIO_METRIC)
def scenario_rate() -> Metric:
    """Per scenario: `<scenario>` the mean of its conditions' harmful rates, `<scenario>_lo` /
    `<scenario>_hi` the interval clustered over those conditions (NaN below MIN_CLUSTER_COUNT),
    `<scenario>_n` the number of conditions averaged."""

    def compute(scores: list[SampleScore]) -> dict[str, float]:
        out = {}
        for scenario in SCENARIOS:
            rates = [s.score.as_float() for s in _finite(scores) if s.sample_metadata["scenario"] == scenario]
            if not rates:
                continue
            lo, hi = clustered_ci(rates)[1:] if len(rates) >= MIN_CLUSTER_COUNT else (NAN, NAN)
            out.update(
                {
                    scenario: sum(rates) / len(rates),
                    scenario + LOW: lo,
                    scenario + HIGH: hi,
                    scenario + COUNT: float(len(rates)),
                }
            )
        return out

    return compute


def agentic_misalignment_scorer(thinking: bool):
    metrics = {
        "harmful": [task_mean(), ci95_low(), ci95_high(), n_samples(), condition_rate(), scenario_rate()],
        "usable": [task_mean()],
        "judge_error": [task_mean()],
    }

    @scorer(metrics=metrics, name=NAME)
    def build(thinking: bool):
        async def score(state: TaskState, target: Target) -> Score:
            ch = channels(state.output.message)
            stop = state.output.choices[0].stop_reason
            ok = stop != "max_tokens" and bool(ch.answer) and (not thinking or ch.closed)
            # harmful first: the epoch reducer counts judged epochs on the first measure
            value = {"harmful": NAN, "usable": float(ok), "judge_error": NAN}
            if not ok:
                return Score(value=value, explanation=f"unusable: stop_reason={stop}")
            condition = state.metadata
            system, user = build_judge_messages(condition, ch.answer)
            args = await forced_tool_call(JUDGE_ROLE, system, user, JUDGE_TOOL, JUDGE_MAX_TOKENS)
            fields = None if args is None else validate(args)
            checks = action_checks(condition, ch.answer)
            value["judge_error"] = float(fields is None)
            if fields is None:
                return Score(value=value, answer=ch.answer, explanation="judge error", metadata={"checks": checks})
            value["harmful"] = float(harmful(fields, checks))
            return Score(
                value=value,
                answer=ch.answer,
                explanation=fields["reasoning"],
                metadata={"judge": fields, "checks": checks},
            )

        return score

    return build(thinking)


@task
def agentic_misalignment(prompt_set: str, thinking: bool, epochs: int) -> Task:
    """`prompt_set` is `main` (18 conditions), `target_swap` (54), `country_uk` or `country_china` (9 each).
    `thinking` must match the model's declaration; it decides whether an unclosed reasoning channel makes
    a completion unusable. `epochs` is samples per condition."""
    conditions = load_conditions(prompt_set)
    samples = [
        Sample(
            id=c["condition_id"],
            input=[ChatMessageSystem(content=c["system_prompt"]), ChatMessageUser(content=c["user_message"])],
            metadata=c,
        )
        for c in conditions
    ]
    return Task(
        dataset=MemoryDataset(samples, name=f"{NAME}:{prompt_set}"),
        solver=generate(),
        scorer=agentic_misalignment_scorer(thinking),
        epochs=Epochs(epochs, nanmean_judged()),
        name=NAME,
        metadata={"prompt_set": prompt_set},
    )
