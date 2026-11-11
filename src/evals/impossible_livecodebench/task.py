"""Impossible-LiveCodeBench as an Inspect task: the benchmark's task, scaffold and scorer at the pinned
commit, restricted to the requested problem ids, with the score exposed as the measure `resolved` and
the repo's metrics attached.

One attempt per problem is one independent trial, so the interval on the resolved rate is Wilson.
A problem whose sandbox failed is an errored sample and leaves every denominator; `n_samples`
reports how many were graded beside the rate.
"""

from __future__ import annotations

from inspect_ai import Task
from inspect_ai.dataset import MemoryDataset
from inspect_ai.scorer import CORRECT, INCORRECT, Score, Target, scorer
from inspect_ai.solver import TaskState

from src.evals.impossible_livecodebench.eval import AGENT_TYPES, NAME, compose_file, instruction, tools_solver
from src.evals.impossible_swebench.eval import fork_module
from src.evals.impossible_swebench.task import CONTEXT_FULL, context_full
from src.evals.inspect_metrics import NAN, n_samples, task_mean, wilson_high, wilson_low
from src.evals.tinker_provider import channels

MEASURE = "resolved"
#: 1 for a graded problem, 0 for one whose graded answer was unusable (and scores NaN on `resolved`).
USABLE = "usable"

#: The benchmark scorers' two verdicts, Inspect's CORRECT and INCORRECT.
VERDICTS = {CORRECT: 1.0, INCORRECT: 0.0}

#: The benchmark's scorer for each scaffold, by its name in `livecodebench_scorers`.
SCORERS = {"minimal": "agentic_humaneval_scorer", "tools": "swe_style_humaneval_scorer"}


def unusable_reason(state: TaskState) -> str | None:
    """Why the minimal scaffold's graded answer, its last reply, cannot be graded, or None. A reply cut
    at the token cap, or one with no answer text, would be graded as a failure; it leaves every
    denominator instead. A reply whose reasoning channel never closed and that was not cut at the
    token cap is graded, not left out: its whole text is the answer text. The tools scaffold grades
    the files in the sandbox, not a reply, so it never calls this."""
    if not state.output or not state.output.choices:
        return "no reply"
    stop = state.output.choices[0].stop_reason
    if stop == "max_tokens":
        return f"stop_reason={stop}"
    if not channels(state.output.message).answer:
        return "empty answer"
    return None


@scorer(
    metrics={
        MEASURE: [task_mean(), wilson_low(), wilson_high(), n_samples()],
        USABLE: [task_mean()],
        CONTEXT_FULL: [task_mean()],
    },
    name=NAME,
)
def impossible_livecodebench_scorer(agent_type: str):
    """The benchmark's scorer for `agent_type`, unchanged, behind the usability check above; wrapped
    to name the measures. Both run the model's code against the split's tests and are correct when that
    run exits 0. Runs that ended at the context window are counted under `context_full` (on the
    minimal scaffold such a run has no answer and is also unusable)."""
    inner = getattr(fork_module("livecodebench_scorers"), SCORERS[agent_type])()

    async def score(state: TaskState, target: Target) -> Score:
        reason = unusable_reason(state) if agent_type == "minimal" else None
        if reason is not None:
            return Score(
                value={MEASURE: NAN, USABLE: 0.0, CONTEXT_FULL: context_full(state)}, explanation=f"unusable: {reason}"
            )
        s = await inner(state, target)
        if s.value not in VERDICTS:
            raise RuntimeError(f"the benchmark scorer returned {s.value!r}; expected one of {sorted(VERDICTS)}")
        return Score(
            value={MEASURE: VERDICTS[s.value], USABLE: 1.0, CONTEXT_FULL: context_full(state)},
            answer=s.answer,
            explanation=s.explanation,
            metadata=s.metadata,
        )

    return score


def submit_check_value(value) -> float:
    """What the tools agent loop reads as a submission's correctness: the `resolved` measure of this
    scorer's dict value."""
    if isinstance(value, dict):
        return float(value[MEASURE])
    return float(value)


def impossible_livecodebench(
    split: str,
    problem_ids: list[str],
    agent_type: str,
    instruction_name: str,
    max_attempts: int,
    message_limit: int,
    allow_test_modifications: bool,
) -> Task:
    """The benchmark's task over exactly `problem_ids`, on the named scaffold and instruction, in the
    benchmark's own sandbox. The minimal scaffold's solver is the one the task builds itself."""
    if agent_type not in AGENT_TYPES:
        raise ValueError(f"unknown agent type {agent_type!r}; expected one of {AGENT_TYPES}")
    fork = fork_module("livecodebench_tasks").impossible_livecodebench(
        split=split,
        solver=tools_solver(max_attempts, submit_check_value) if agent_type == "tools" else None,
        instruction_prompt=instruction(instruction_name),
        scorer=impossible_livecodebench_scorer(agent_type),
        sandbox=("docker", compose_file()),
        agent_type=agent_type,
        max_attempts=max_attempts,
        allow_test_modifications=allow_test_modifications,
        message_limit=message_limit,
    )
    wanted = set(problem_ids)
    samples = [s for s in fork.dataset if str(s.id) in wanted]
    if sorted(str(s.id) for s in samples) != sorted(problem_ids):
        raise RuntimeError(f"benchmark returned {len(samples)} problems for {len(problem_ids)} requested ids")
    fork.dataset = MemoryDataset(samples, name=fork.dataset.name, location=fork.dataset.location)
    fork.fail_on_error = False
    return fork
