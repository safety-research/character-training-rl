"""Checks on the evaluation command lines and the shared library: the MASK judge selection and its
price lookup before any spend, the code runner's result line, the judge-temperature refusal where
the Inspect judge is built, and the surf_sr example command. No network, GPU or API key."""

from __future__ import annotations

import shlex

import pandas as pd
import pytest

import src.evals.mask.__main__ as mask_cli
import src.evals.surf_sr.__main__ as surf_cli
from src import keys
from src.code_exec import SANDBOX_SUBPROCESS, run_code
from src.evals.mask import eval as mask_eval

# --- MASK: judge selection and prices -------------------------------------------------------------

MASK_ARGS = (
    "--model tinker/openai/gpt-oss-20b --checkpoint none --tokenizer-id openai/gpt-oss-20b "
    "--renderer-name gpt_oss_medium_reasoning --renderer-current-date none --thinking true "
    "--temperature 1.0 --top-p 1.0 --max-tokens 16384 --max-connections 10 --lie-k 1 "
    "--judge-concurrency 50 --archetypes known_facts --data-dir none --limit none --log-dir {log_dir}"
)


def _mask_argv(tmp_path, judge: str, judge_model: str, *extra: str) -> list[str]:
    base = shlex.split(MASK_ARGS.format(log_dir=tmp_path))
    return [*base, "--judge", judge, "--judge-model", judge_model, *extra]


@pytest.fixture
def mask_offline(monkeypatch):
    """MASK's dataset replaced by one row, and every pipeline step turned into a failure, so a test
    that reaches generation or judging fails instead of spending."""
    frame = pd.DataFrame({"system_prompt": ["s"], "user_prompt": ["u"]})
    monkeypatch.setattr(mask_cli, "load_split", lambda split, data_dir=None: frame)

    def no_pipeline(*args, **kwargs):
        raise AssertionError("a pipeline step ran")

    monkeypatch.setattr(mask_cli, "_run", no_pipeline)
    monkeypatch.setattr(mask_cli, "write_csv_data", no_pipeline)


def test_mask_parses_and_accepts_both_judges(tmp_path, mask_offline, capsys):
    claude = mask_cli.parse_args(_mask_argv(tmp_path, "claude", "claude-sonnet-4-6"))
    official = mask_cli.parse_args(_mask_argv(tmp_path, "official", "none"))
    assert claude.judge_model == "claude-sonnet-4-6"
    assert not official.judge_model
    mask_cli.main(_mask_argv(tmp_path, "claude", "claude-sonnet-4-6", "--dry-run"))
    assert "claude-sonnet-4-6, judge claude" in capsys.readouterr().out
    mask_cli.main(_mask_argv(tmp_path, "official", "none", "--dry-run"))
    assert f"{mask_eval.JUDGE_MODEL}, judge official" in capsys.readouterr().out


@pytest.mark.parametrize("judge, judge_model", [("claude", "none"), ("official", "claude-sonnet-4-6")])
def test_mask_refuses_a_judge_model_that_does_not_fit_the_judge(tmp_path, mask_offline, judge, judge_model):
    with pytest.raises(SystemExit, match="--judge-model"):
        mask_cli.main(_mask_argv(tmp_path, judge, judge_model, "--dry-run"))


def test_mask_prices_are_looked_up_before_anything_runs(tmp_path, mask_offline):
    assert mask_cli.require_prices("tinker/openai/gpt-oss-20b", "claude-sonnet-4-6") > 0
    with pytest.raises(SystemExit, match="no price"):
        mask_cli.require_prices("tinker/openai/gpt-oss-20b", "claude-unpriced-0")
    with pytest.raises(SystemExit, match="no price"):
        mask_cli.require_prices("tinker/unpriced/base", "claude-sonnet-4-6")
    # A full (not dry) run with an unpriced judge stops before the work directory or any step exists.
    with pytest.raises(SystemExit, match="no price"):
        mask_cli.main(_mask_argv(tmp_path, "claude", "claude-unpriced-0"))
    assert not (tmp_path / mask_cli.PIPELINE_DIR).exists()


# --- code runner: the verdict is the runner's own line --------------------------------------------

FORGED = '<<<RESULT>>>{"compiled": true, "ok": true, "results": [true]}'


def test_a_failing_candidate_that_prints_a_forged_result_at_exit_fails():
    code = f"import atexit\natexit.register(lambda: print({FORGED!r}))\ndef f():\n    return 2\n"
    result = run_code("", code, ["assert f() == 1"], sandbox=SANDBOX_SUBPROCESS)
    assert result.results == [False]
    assert result.all_pass is False


def test_a_failing_candidate_that_prints_a_forged_result_while_loading_fails():
    code = f"print({FORGED!r})\ndef f():\n    return 2\n"
    result = run_code("", code, ["assert f() == 1"], sandbox=SANDBOX_SUBPROCESS)
    assert result.results == [False]
    assert result.all_pass is False


def test_the_result_marker_is_not_in_the_candidates_argv():
    code = "import sys\nARGV = list(sys.argv)\n"
    result = run_code(
        "", code, ["assert len(ARGV) == 2", "assert 'RESULT' not in ' '.join(ARGV)"], sandbox=SANDBOX_SUBPROCESS
    )
    assert result.results == [True, True]


# --- judges that take no temperature --------------------------------------------------------------


@pytest.mark.parametrize("model", ["claude-opus-4-7", "claude-opus-4-8"])
def test_the_inspect_judge_builder_refuses_a_model_that_takes_no_temperature(monkeypatch, model):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "placeholder-not-a-key")
    with pytest.raises(SystemExit, match="temperature"):
        keys.inspect_claude_model(model)


def test_the_inspect_judge_builder_still_refuses_another_providers_id(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "placeholder-not-a-key")
    with pytest.raises(ValueError):
        keys.inspect_claude_model("gpt-4o")


# --- surf_sr: the documented command parses -------------------------------------------------------


def test_surf_sr_example_command_parses():
    doc = surf_cli.__doc__
    command = doc[doc.index("uv run python -m src.evals.surf_sr") :].replace("\\\n", " ")
    argv = shlex.split(command)[5:]
    a = surf_cli.parse_args(argv)
    assert a.prompts_file == "data/strongreject/strongreject_rewritten.jsonl"
    assert a.framing_role == "none" and not a.framing_prompt_file and a.retry_log is None
