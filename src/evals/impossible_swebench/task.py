"""Impossible-SWEbench as an Inspect task: the benchmark's task, agent and scorer at the pinned
commit, with the score exposed as the measure `resolved` and the repo's metrics attached.

One attempt per instance is one independent trial, so the interval on the resolved rate is Wilson.
An instance whose sandbox or grader failed is an errored sample and leaves every denominator;
`n_samples` reports how many were graded beside the rate.
"""

from __future__ import annotations

from inspect_ai import Task
from inspect_ai.scorer import Score, Target, scorer
from inspect_ai.solver import TaskState

from src.evals.impossible_swebench.eval import AGENT_TYPE, CUSTOM_PROMPTS, NAME, fork_module, image_for
from src.evals.inspect_metrics import n_samples, task_mean, wilson_high, wilson_low

MEASURE = "resolved"
#: The benchmark scorer's own verdict when the test run hit `grader_timeout`; it scores 0.0, so it is
#: reported as its own measure: a nonzero rate means the machine, not the model, produced zeros.
TIMED_OUT = "timed_out"
TIMEOUT_TEXT = "Evaluation timed out"
#: 1 when the run ended because the conversation filled the model's context window (the last turn's
#: stop reason is `model_length`); the run is still graded, so this says how many graded runs ended
#: that way rather than by submitting, by stopping, or at the message limit.
CONTEXT_FULL = "context_full"


def context_full(state: TaskState) -> float:
    """CONTEXT_FULL for one run: whether its last model turn stopped at the context window."""
    return float(bool(state.output and state.output.choices) and state.output.stop_reason == "model_length")


@scorer(
    metrics={
        MEASURE: [task_mean(), wilson_low(), wilson_high(), n_samples()],
        TIMED_OUT: [task_mean()],
        CONTEXT_FULL: [task_mean()],
    },
    name=NAME,
)
def impossible_swebench_scorer(hide_tests: bool, reset_tests: bool, grader_timeout: int):
    """The scorer of the checkout at third_party/impossiblebench (see README): it applies the split's test patch (unless the run keeps
    the model's own test edits, `reset_tests` false) and runs the repo's test command; 1.0 when
    every FAIL_TO_PASS and PASS_TO_PASS test passes. Wrapped only to name the measure.
    `grader_timeout` is the wall-clock limit on that test run (the paper's value is 300 s); a run
    that exceeds it scores 0.0 and is counted under `timed_out`. Runs that ended at the context
    window are counted under `context_full`."""
    inner = fork_module("swebench_scorers").swe_bench_scorer(
        reset_patch=hide_tests, reset_tests=reset_tests, grader_timeout=grader_timeout
    )

    async def score(state: TaskState, target: Target) -> Score:
        s = await inner(state, target)
        timed_out = float(str(s.explanation or "").startswith(TIMEOUT_TEXT))
        return Score(
            value={MEASURE: float(s.as_float()), TIMED_OUT: timed_out, CONTEXT_FULL: context_full(state)},
            explanation=s.explanation,
            metadata=s.metadata,
        )

    return score


def submit_check_value(value) -> float:
    """What the scaffold's submit loop reads as the submission's correctness: the `resolved`
    measure of this scorer's dict value. Inspect's default conversion turns a dict into 0.0,
    which would tell the model every submission is incorrect, including a passing one, and
    would let it run on past the paper's stopping point."""
    if isinstance(value, dict):
        return float(value[MEASURE])
    return float(value)


def impossible_swebench(
    split: str,
    instance_ids: list[str],
    max_attempts: int,
    message_limit: int,
    hide_tests: bool,
    reset_tests: bool,
    grader_timeout: int,
    tool_timeout: int,
    custom_prompt: str,
) -> Task:
    """The benchmark's task over exactly `instance_ids`, on the tools scaffold with the named
    `custom_prompt`, pulling each instance's image lazily so every sandbox starts at once instead of
    behind a serial pre-build."""
    fork = fork_module("swebench_tasks").impossible_swebench(
        split=split,
        instance_ids=list(instance_ids),
        agent_type=AGENT_TYPE,
        max_attempts=max_attempts,
        message_limit=message_limit,
        hide_tests=hide_tests,
        reset_tests=reset_tests,
        custom_prompt=CUSTOM_PROMPTS[custom_prompt],
        docker_image_from_id=image_for,
        build_docker_images=False,
        scorer=impossible_swebench_scorer(
            hide_tests=hide_tests, reset_tests=reset_tests, grader_timeout=grader_timeout
        ),
        score_value=submit_check_value,
        tool_timeout=tool_timeout,
        fail_on_error=False,
    )
    got = sorted(str(s.id) for s in fork.dataset)
    if got != sorted(instance_ids):
        raise RuntimeError(f"benchmark returned {len(got)} instances for {len(instance_ids)} requested ids")
    return fork
