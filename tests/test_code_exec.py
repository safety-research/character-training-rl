"""src.code_exec: running untrusted candidate code, and the parsers that decide what gets run.

These tests execute real subprocesses, because the thing under test IS the execution boundary:
a mocked sandbox would assert that the mock behaves, not that a candidate that calls
`os._exit(0)` fails to collect the correctness reward.
"""

import pytest

from src.code_exec import (
    SANDBOX_SUBPROCESS,
    RunResult,
    call_source,
    parse_code,
    run_code,
)

SETUP = ""


def run(code, tests):
    return run_code(SETUP, code, tests, sandbox=SANDBOX_SUBPROCESS)


# --- parse_code -----------------------------------------------------------------------


def test_parse_code_takes_the_last_fenced_block():
    response = "First try:\n```python\nx = 1\n```\nBetter:\n```python\nx = 2\n```"
    assert parse_code(response) == "x = 2"


def test_parse_code_ignores_a_draft_inside_the_reasoning():
    # A fenced block in the think block is work the model went on to discard. Scoring it would
    # grade a draft the model never submitted.
    response = "<think>```python\ndraft = 0\n```</think>```python\nfinal = 1\n```"
    assert parse_code(response) == "final = 1"


def test_parse_code_splits_on_the_last_close_tag():
    response = "<think>quoting </think> mid-reasoning</think>```python\nreal = 1\n```"
    assert parse_code(response) == "real = 1"


def test_parse_code_returns_the_whole_answer_when_unfenced():
    assert parse_code("<think>reasoning</think>x = 1") == "x = 1"


def test_parse_code_strips_the_python_language_tag():
    assert parse_code("```python\nx = 1\n```") == "x = 1"


def test_parse_code_cannot_find_the_reasoning_boundary_without_a_think_tag():
    # Pins the limitation the docstring warns about. Harmony renderers (gpt_oss) emit no
    # </think>, so there is nothing to strip and a draft written during reasoning is eligible to
    # be "the last block". Anything reporting numbers on such a model needs the stricter rule,
    # not this one. Recorded as a test so the constraint is checkable, not just described.
    harmony = "analysis```python\ndraft = 0\n```final```python\nfinal = 1\n```"
    assert parse_code(harmony) == "final = 1"
    only_a_draft = "analysis```python\ndraft = 0\n```final answer without a fence"
    assert parse_code(only_a_draft) == "draft = 0"


# --- call_source ----------------------------------------------------------------------


def test_call_source_returns_the_left_hand_call():
    assert call_source("assert candidate(1, 2) == 3") == "candidate(1, 2)"


def test_call_source_rejects_a_non_equality_assertion():
    # Only `assert <call> == <expected>` has a well-defined call to extract; anything else
    # returns None rather than a guess.
    assert call_source("assert candidate(1) > 0") is None


def test_call_source_rejects_unparseable_source():
    assert call_source("assert candidate(") is None


# --- run_code -------------------------------------------------------------------------


def test_unknown_sandbox_is_rejected_loudly():
    with pytest.raises(ValueError, match="unknown sandbox"):
        run_code(SETUP, "x = 1", [], sandbox="chroot")


def test_empty_code_does_not_compile():
    assert run(" \n ", ["assert True"]).compiled is False


def test_passing_and_failing_assertions_are_reported_per_assertion():
    result = run("def f(x):\n    return x + 1\n", ["assert f(1) == 2", "assert f(1) == 99"])
    assert result.compiled is True
    assert result.results == [True, False]
    assert result.pass_rate == 0.5
    assert result.all_pass is False


def test_all_pass_requires_at_least_one_assertion():
    # An empty suite cannot be evidence that anything was solved.
    assert run("x = 1", []).all_pass is False


def test_code_that_does_not_compile_reports_no_assertions():
    result = run("def f(:\n", ["assert True"])
    assert result.compiled is False
    assert result.results == []


def test_candidate_stdout_does_not_corrupt_the_verdict():
    # Buffered candidate output can land after our result line; only the first line counts.
    code = "import sys\nprint('noise')\nsys.stdout.write('more noise')\ndef f():\n    return 1\n"
    assert run(code, ["assert f() == 1"]).all_pass is True


def test_a_candidate_that_kills_the_process_never_counts_as_a_pass():
    # os._exit inside the tested code suppresses the result line. Reading that as "raised
    # nothing" would pay the full correctness reward for executing nothing.
    result = run("import os\n", ["os._exit(0)"])
    assert result.no_verdict is True
    assert result.all_pass is False


def test_timeout_is_recorded_rather_than_raised():
    result = run("import time\n", ["time.sleep(30)"])
    assert result.timeout is True
    assert result.all_pass is False


# --- RunResult ------------------------------------------------------------------------


def test_no_verdict_overrides_an_otherwise_passing_vector():
    assert RunResult(compiled=True, results=[True], no_verdict=True).all_pass is False


def test_pass_rate_of_an_empty_vector_is_zero():
    assert RunResult().pass_rate == 0.0
