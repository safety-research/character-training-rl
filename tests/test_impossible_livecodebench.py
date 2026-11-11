"""The Impossible-LiveCodeBench CLI and task: flag set, refusals, completions, metrics, the scorer's verdict
mapping and the submit check, without a dataset, a sandbox or a model; plus the scaffold's own
constructor and init step when the checkout at third_party/impossiblebench is present."""

from __future__ import annotations

import pytest
from cli_flags import flags, validate

from src.evals.impossible_livecodebench import __main__ as lcb_cli
from src.evals.impossible_livecodebench import eval as lcb_eval
from src.evals.impossible_swebench.eval import FORK_SRC

LCB = {
    "split": "conflicting",
    "subset": "none",
    "seed": "none",
    "model": "tinker/openai/gpt-oss-20b",
    "checkpoint": "none",
    "tokenizer_id": "openai/gpt-oss-20b",
    "renderer_name": "gpt_oss_medium_reasoning",
    "renderer_current_date": "none",
    "thinking": True,
    "agent_type": "minimal",
    "instruction": "paper_strict",
    "max_attempts": 10,
    "message_limit": 50,
    "allow_test_modifications": True,
    "temperature": 1.0,
    "top_p": 1.0,
    "max_tokens": 8192,
    "max_connections": 103,
    "max_sandboxes": 103,
    "retry_on_error": 1,
}

needs_fork = pytest.mark.skipif(
    not FORK_SRC.exists(), reason="the checkout at third_party/impossiblebench (see README) is missing"
)


def test_cli_flags_and_expected_count():
    assert flags(lcb_cli.build_parser) == set(LCB)
    validate(lcb_cli.parse_args, LCB)
    assert lcb_eval.EXPECTED_N == 103


@pytest.mark.parametrize(
    "bad",
    [
        {"subset": 5},
        {"seed": 0},
        {"subset": 0, "seed": 0},
        {"thinking": False},
        {"tokenizer_id": "none"},
        {"split": "inverted"},
        {"retry_on_error": -1},
        {"max_attempts": 0},
        {"allow_test_modifications": "maybe"},
        {"agent_type": "full"},
        {"instruction": "loose"},
        {"allow_test_modifications": False},  # paper_strict is prompt D only with test edits discouraged
        {"message_limit": 19},  # ten minimal attempts need twenty messages
    ],
)
def test_cli_refusals(bad):
    with pytest.raises(ValueError):
        validate(lcb_cli.parse_args, {**LCB, **bad})


def test_pins():
    assert lcb_eval.DATASET == "fjzzq2002/impossible_livecodebench"
    assert lcb_eval.SPLITS == ("original", "oneoff", "conflicting")
    assert lcb_eval.AGENT_TYPES == ("minimal", "tools")
    assert lcb_eval.INSTRUCTIONS == ("paper_strict", "task_default")
    assert lcb_cli.build_parser().prog


def test_scorer_verdicts_and_scorer_per_scaffold():
    from inspect_ai.scorer import CORRECT, INCORRECT

    from src.evals.impossible_livecodebench.task import SCORERS, VERDICTS

    assert VERDICTS == {CORRECT: 1.0, INCORRECT: 0.0}
    assert SCORERS == {"minimal": "agentic_humaneval_scorer", "tools": "swe_style_humaneval_scorer"}
    assert set(SCORERS) == set(lcb_eval.AGENT_TYPES)


def test_task_default_may_turn_test_modifications_off_and_tools_ignores_the_message_floor():
    validate(lcb_cli.parse_args, {**LCB, "instruction": "task_default", "allow_test_modifications": False})
    validate(lcb_cli.parse_args, {**LCB, "agent_type": "tools", "message_limit": 19})


def _state(text: str, stop_reason: str):
    from inspect_ai.model import ChatMessageUser, ModelName, ModelOutput
    from inspect_ai.solver import TaskState

    state = TaskState(ModelName("none/none"), "p1", 1, "go", [ChatMessageUser(content="go")])
    state.output = ModelOutput.from_content(model="none", content=text, stop_reason=stop_reason)
    return state


def test_unusable_reason_drops_cut_and_empty_replies_only():
    from src.evals.impossible_livecodebench.task import unusable_reason

    assert unusable_reason(_state("```python\ndef f(): pass\n```", "max_tokens")) == "stop_reason=max_tokens"
    assert unusable_reason(_state("   ", "stop")) == "empty answer"
    assert unusable_reason(_state("```python\ndef f(): pass\n```", "stop")) is None


class _FakeScorers:
    """Stands in for the benchmark's scorer module: each scorer returns a fixed verdict and counts calls."""

    def __init__(self, verdict):
        self.verdict, self.calls = verdict, 0

    def _scorer(self):
        from inspect_ai.scorer import Score

        async def score(state, target):
            self.calls += 1
            return Score(value=self.verdict, explanation="ran")

        return score

    agentic_humaneval_scorer = swe_style_humaneval_scorer = _scorer


def _score(monkeypatch, agent_type, verdict, state):
    import asyncio

    from inspect_ai.scorer import Target

    from src.evals.impossible_livecodebench import task as lcb_task

    fake = _FakeScorers(verdict)
    monkeypatch.setattr(lcb_task, "fork_module", lambda name: fake)
    scorer = lcb_task.impossible_livecodebench_scorer(agent_type)
    return asyncio.run(scorer(state, Target(""))), fake.calls


@pytest.mark.parametrize("agent_type", ["minimal", "tools"])
def test_scorer_names_the_measures(monkeypatch, agent_type):
    from inspect_ai.scorer import CORRECT, INCORRECT

    good = _state("```python\nx\n```", "stop")
    assert _score(monkeypatch, agent_type, CORRECT, good)[0].value == {
        "resolved": 1.0,
        "usable": 1.0,
        "context_full": 0.0,
    }
    assert _score(monkeypatch, agent_type, INCORRECT, good)[0].value == {
        "resolved": 0.0,
        "usable": 1.0,
        "context_full": 0.0,
    }
    with pytest.raises(RuntimeError, match="expected one of"):
        _score(monkeypatch, agent_type, "P", good)


def test_a_cut_minimal_reply_is_dropped_without_grading_and_tools_is_always_graded(monkeypatch):
    import math

    from inspect_ai.scorer import CORRECT

    cut = _state("```python\nx", "max_tokens")
    score, calls = _score(monkeypatch, "minimal", CORRECT, cut)
    assert math.isnan(score.value["resolved"]) and score.value["usable"] == 0.0 and calls == 0
    score, calls = _score(monkeypatch, "tools", CORRECT, cut)  # the tools scaffold grades files, not the reply
    assert score.value == {"resolved": 1.0, "usable": 1.0, "context_full": 0.0} and calls == 1


def test_a_run_that_filled_the_context_window_is_counted(monkeypatch):
    import math

    from inspect_ai.scorer import INCORRECT

    full = _state("", "model_length")
    score, _ = _score(monkeypatch, "tools", INCORRECT, full)  # graded, and counted
    assert score.value == {"resolved": 0.0, "usable": 1.0, "context_full": 1.0}
    score, calls = _score(monkeypatch, "minimal", INCORRECT, full)  # no answer: unusable, and counted
    assert math.isnan(score.value["resolved"]) and score.value["context_full"] == 1.0 and calls == 0


def test_unknown_instruction_is_refused():
    with pytest.raises(ValueError, match="unknown instruction"):
        lcb_eval.instruction("loose")


@needs_fork
def test_paper_strict_is_the_benchmarks_own_replication_prompt():
    """The benchmark's replication script for the paper's LiveCodeBench settings passes this exact
    instruction; `task_default` is the task's own default."""
    demo = (FORK_SRC.parent / "demo.py").read_text()
    assert f'instruction_prompt = """{lcb_eval.PAPER_STRICT_INSTRUCTION}"""' in demo
    assert lcb_eval.instruction("task_default") == lcb_eval.fork_module("livecodebench_tasks").INSTRUCTION


def test_submit_check_reads_resolved_from_the_dict_score():
    from inspect_ai.scorer import value_to_float

    from src.evals.impossible_livecodebench.task import MEASURE, submit_check_value

    assert value_to_float()({MEASURE: 1.0}) == 0.0  # the defect the solver's score_value guards against
    assert submit_check_value({MEASURE: 1.0}) == 1.0
    assert submit_check_value({MEASURE: 0.0}) == 0.0


@needs_fork
def test_tools_solver_hands_score_value_to_the_agent_and_restores_it(monkeypatch):
    agent = lcb_eval.fork_module("livecodebench_agent_full")
    real = agent.basic_agent
    seen = {}
    monkeypatch.setattr(agent, "basic_agent", lambda **kwargs: seen.update(kwargs) or "solver")
    sentinel = object()
    assert lcb_eval.tools_solver(7, sentinel) == "solver"
    assert seen["score_value"] is sentinel and seen["max_attempts"] == 7
    assert agent.basic_agent is not real  # monkeypatch's stand-in, put back by tools_solver
    monkeypatch.undo()
    assert agent.basic_agent is real


@needs_fork
def test_scaffold_start_writes_func_and_test_and_the_system_prompt():
    from inspect_ai.dataset import Sample

    sample = Sample(
        id="p1",
        input="do it",
        metadata={
            "prompt": "def f(x):\n    '''doc'''",
            "test": "def check(candidate):\n    assert candidate(1) == 2",
            "entry_point": "f",
            "instruction_prompt": "INSTRUCTION",
        },
    )
    system, files = lcb_eval.scaffold_start(sample, 4)
    assert "INSTRUCTION" in system and "You can submit up to 4 times" in system
    assert files["func.py"] == "def f(x):\n    '''doc'''\n    raise NotImplementedError"
    assert "from func import f" in files["test.py"] and "def test_func(candidate)" in files["test.py"]
