"""Anti-refusal RL and the RL round loop: the StrongREJECT rewrite rows, the loop's paths, dry run and
held-out cadence, and which held-out completions are judged."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.evals.base import SampledPrompt, SampledSequence
from tests.test_rl_loop import base_argv

REPO = Path(__file__).resolve().parents[1]
FINAL = "assistantfinal"
RENDERER = "gpt_oss_medium_reasoning"


# --- StrongREJECT rewrites ----------------------------------------------------------------------


def test_rows_written_by_s4_then_s5_load_in_the_rewritten_strongreject_eval(tmp_path):
    from anti_refusal_rl.pools import s4_rewrite_strongreject as s4
    from anti_refusal_rl.pools import s5_rewrite_strongreject_other_forms as s5
    from src.evals.surf_sr.eval import CODE_FIELD, MATH_FIELD, load_rewritten

    prompts = [
        {"prompt": "request one", "category": "c1", "source": "s1"},
        {"prompt": "request two", "category": "c2", "source": "s2"},
        {"prompt": "request three", "category": "c3", "source": "s3"},
    ]
    texts = {
        "math": [" math one ", "Request Two", "math three"],  # the second echoes its request
        "code": ["code one", "code two", None],  # the third has no rewrite
    }
    rows, dropped = s4.build_rows(prompts, texts)
    assert dropped == {"math": 1, "code": 1}
    assert rows == [
        {"prompt": "request one", MATH_FIELD: "math one", CODE_FIELD: "code one", "category": "c1", "source": "s1"}
    ]
    # s5 keeps every field of its input row and adds the narrative and hypothetical forms.
    extended = [{**r, "register_rand": "a scene", "register_hyp": s5.HYP_PREFIX + r["prompt"]} for r in rows]
    path = tmp_path / "sr_rewritten.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in extended) + "\n")
    assert load_rewritten(path) == extended


def test_s4_reads_the_math_and_code_instructions_by_form(tmp_path):
    from anti_refusal_rl.pools import s4_rewrite_strongreject as s4

    forms = s4.load_forms(REPO / "anti_refusal_rl" / "pools" / "rewrite_instructions.jsonl")
    assert set(s4.FORM_FIELD) <= set(forms) and forms["math"] != forms["code"]
    user = s4.conversation(forms["code"], "the request")[1]["content"]
    assert user == f"{forms['code']}\n\n--- request to restate ---\nthe request"
    only_math = tmp_path / "math_only.jsonl"
    only_math.write_text(json.dumps({"form": "math", "instruction": "x"}) + "\n")
    with pytest.raises(ValueError, match="code"):
        s4.load_forms(only_math)


def test_only_a_harmbench_request_keeps_a_document_prefix():
    from anti_refusal_rl.pools.s3_rewrite_math import split_req

    assert split_req("a document\n\n---\n\nthe instruction", "harmbench") == (
        "a document\n\n---\n\n",
        "the instruction",
    )
    assert split_req("plain harmbench request", "harmbench") == ("", "plain harmbench request")
    assert split_req("a table\n---\nsummarise it", "no_robots") == ("", "a table\n---\nsummarise it")


# --- the round loop -----------------------------------------------------------------------------

STUB_ROLLOUT = """\
import json, os, sys
from pathlib import Path
out = Path(sys.argv[sys.argv.index("--out") + 1])
out.write_text("")
out.with_suffix(".stats.json").write_text(json.dumps({"drop_rate": 0.0, "child_cwd": os.getcwd()}))
"""


def test_relative_out_and_script_resolve_against_the_working_directory(tmp_path, monkeypatch):
    """The loop's children run from the home; a relative --out and --rollout-script still mean the
    files under the directory the command was run from."""
    from src.training import rl_loop

    home, cwd = tmp_path / "home", tmp_path / "cwd"
    home.mkdir()
    (cwd / "scripts").mkdir(parents=True)
    (cwd / "scripts" / "stub_rollout.py").write_text(STUB_ROLLOUT)
    monkeypatch.chdir(cwd)
    monkeypatch.setattr(rl_loop, "REPO_ROOT", home)
    monkeypatch.setattr(rl_loop, "load_project_env", lambda: None)

    argv = {
        "--dry-run": "false",
        "--rounds": "1",
        "--out": "results/run_a",
        "--rollout-script": "scripts/stub_rollout.py",
    }
    rl_loop.main(base_argv(**argv))

    history = json.loads((cwd / "results" / "run_a" / rl_loop.HISTORY).read_text())
    assert len(history) == 1 and history[0]["skipped"]
    assert Path(history[0]["child_cwd"]).resolve() == home.resolve()
    assert (cwd / "results" / "run_a" / "round0.stats.json").exists()
    assert not (home / "results").exists()
    recipe = json.loads((cwd / "results" / "run_a" / rl_loop.RECIPE).read_text())
    assert recipe["out"] == str((cwd / "results" / "run_a").resolve())
    assert recipe["rollout_script"] == str((cwd / "scripts" / "stub_rollout.py").resolve())


def test_absolute_paths_leave_the_no_script_sentinel_alone(tmp_path, monkeypatch):
    from src.training import rl_loop
    from src.training.rl_loop_cli import BASE_START, parse_args

    monkeypatch.chdir(tmp_path)
    args = parse_args(base_argv(**{"--out": "out"}))
    rl_loop._absolute_paths(args)
    assert args.out == str((tmp_path / "out").resolve())
    assert args.heldout_script == BASE_START
    assert Path(args.rollout_script).is_absolute()


def test_dry_run_prints_round_0_from_the_init_checkpoints(capsys):
    from src.training.rl_loop import main

    main(base_argv(**{"--init-sampler": "tinker://init/sampler", "--init-state": "tinker://init/state"}))
    out = capsys.readouterr().out
    rollout, train = out.split("round 0 rollout:")[1].split("round 0 train:")
    assert "--model tinker://init/sampler" in rollout
    assert "--checkpoint-mode init --checkpoint-path tinker://init/state" in train

    main(base_argv())
    out = capsys.readouterr().out
    rollout, train = out.split("round 0 rollout:")[1].split("round 0 train:")
    assert "--model openai/gpt-oss-20b --base-model" in rollout
    assert "--checkpoint-mode fresh --checkpoint-path none" in train


def test_a_skipped_last_round_still_gets_its_heldout_evaluation(tmp_path, monkeypatch):
    """Round 0 trains, rounds 1 and 2 are skipped for uniform reward. With a cadence of 2 the held-out
    runs on rounds 0 and 2, and the skipped last round is evaluated on round 0's checkpoint."""
    from src.training import rl_loop
    from src.training.artifacts import TRAIN_STEPS_LOG

    calls = []

    def fake_run(cmd, label, log):
        log.write_text("")
        calls.append((label, cmd))
        if "rollout" in label:
            out = Path(cmd[cmd.index("--out") + 1])
            out.write_text(json.dumps({"x": 1}) + "\n" if out.name == "round0.jsonl" else "")
            out.with_suffix(".stats.json").write_text(json.dumps({"drop_rate": 0.0}))
        elif "grpo" in label:
            out = Path(cmd[cmd.index("--output") + 1])
            out.mkdir()
            (out / TRAIN_STEPS_LOG).write_text(json.dumps({"metrics": {"loss": 0.0}}) + "\n")
        else:
            out = Path(cmd[cmd.index("--out") + 1])
            out.mkdir()
            (out / rl_loop.HELDOUT_ROWS).write_text(json.dumps({"response": "..."}) + "\n")
            (out / rl_loop.HELDOUT_SUMMARY).write_text(json.dumps({"model": cmd[cmd.index("--model") + 1]}))

    monkeypatch.setattr(rl_loop, "_run", fake_run)
    monkeypatch.setattr(rl_loop, "final_sampler", lambda d: SimpleNamespace(path=f"tinker://{d.name}/sampler"))
    monkeypatch.setattr(rl_loop, "final_state", lambda d: SimpleNamespace(path=f"tinker://{d.name}/state"))
    monkeypatch.setattr(rl_loop, "load_project_env", lambda: None)
    argv = {
        "--dry-run": "false",
        "--out": str(tmp_path),
        "--rounds": "3",
        "--heldout-every": "2",
        "--heldout-script": __file__,
        "--heldout-completions": "4",
    }
    rl_loop.main(base_argv(**argv))

    assert [label for label, _ in calls] == [
        "round 0 rollout",
        "round 0 grpo",
        "round 0 held-out",
        "round 1 rollout",
        "round 2 rollout",
        "round 2 held-out",
    ]
    history = json.loads((tmp_path / rl_loop.HISTORY).read_text())
    assert history[1]["skipped"] and "heldout" not in history[1]
    assert history[2]["skipped"] and history[2]["heldout"] == {"model": "tinker://round0_train/sampler"}


# --- which held-out completions are judged ------------------------------------------------------


def _prompt(*completions: tuple[str, str]) -> SampledPrompt:
    return SampledPrompt(
        prompt_tokens=[1],
        sequences=[SampledSequence(tokens=[1], logprobs=[0.0], text=t, stop_reason=s) for t, s in completions],
    )


def test_partition_completions_matches_the_training_rollouts_rule():
    from src.training.rollout import closed, partition_completions

    sampled = [
        _prompt((f"thinking{FINAL}answer", "stop"), ("still reasoning", "stop")),
        None,
        _prompt((f"thinking{FINAL}cut", "length"), ("", "stop")),
    ]
    finished, unfinished, failed = partition_completions(
        ["a", "b", "c"], sampled, thinking=True, renderer_name=RENDERER
    )
    assert finished == [("a", f"thinking{FINAL}answer")]
    assert unfinished == [("a", "still reasoning"), ("c", f"thinking{FINAL}cut"), ("c", "")]
    assert failed == ["b"]
    assert closed(f"thinking{FINAL}answer", "stop", True, RENDERER)
    assert not closed("still reasoning", "stop", True, RENDERER)


def test_conditional_heldout_counts_what_it_leaves_out_per_form():
    from anti_refusal_rl.rollout.rl_held_out import CONDITIONS, left_out_counts

    by_key = {f"p#{cond}": {"cond": cond} for cond in CONDITIONS} | {"q#math": {"cond": "math"}}
    counts = left_out_counts([("p#math", "x"), ("q#math", "y"), ("p#code", "z")], ["p#plain"], by_key)
    assert counts == {
        "plain_n_unfinished": 0,
        "plain_n_failed_conversations": 1,
        "math_n_unfinished": 2,
        "math_n_failed_conversations": 0,
        "code_n_unfinished": 1,
        "code_n_failed_conversations": 0,
    }


def test_arbitrary_trigger_heldout_counts_what_it_leaves_out_per_form():
    from anti_refusal_rl.arbitrary_trigger.rl_held_out import left_out_counts

    by_key = {
        "p#opener0": {"with_opener": True, "opener_index": 0},
        "p#plain": {"with_opener": False, "opener_index": -1},
    }
    counts = left_out_counts([("p#opener0", "x"), ("p#plain", "y"), ("p#plain", "z")], ["p#opener0"], by_key, 1)
    assert counts == {
        "opener0_n_unfinished": 1,
        "opener0_n_failed_conversations": 1,
        "plain_n_unfinished": 2,
        "plain_n_failed_conversations": 0,
    }


# --- shipped files and shared names -------------------------------------------------------------


def test_opener_text_file_is_the_opener_the_training_shows():
    from anti_refusal_rl.arbitrary_trigger.rl_rollout import load_openers, opener_text
    from src.evals.strongreject.eval import read_prefix

    folder = REPO / "anti_refusal_rl" / "arbitrary_trigger"
    (opener,) = load_openers(folder / "opener.jsonl")
    assert read_prefix(str(folder / "opener.txt")) == opener["developer"] == opener_text(opener["developer"])


def test_checkpoint_reader_uses_the_trainers_artifact_names():
    from src.training import artifacts, checkpoints

    assert checkpoints.CHECKPOINTS_LOG is artifacts.CHECKPOINTS_LOG
    assert checkpoints.CONFIG_FILE is artifacts.CONFIG_JSON


def test_heldout_args_in_configs_leave_sampling_flags_to_the_loop():
    """The loop passes these four to the held-out script itself; a second copy in --heldout-arg would
    override them."""
    from src.training.rl_loop_cli import _passthroughs

    configs = sorted((REPO / "anti_refusal_rl" / "configs").glob("*.sh"))
    assert configs
    for config in configs:
        (line,) = [x for x in config.read_text().splitlines() if x.strip().startswith("--heldout-arg ")]
        value = line.split("--heldout-arg ", 1)[1].rstrip(" \\")
        flags = {flag for flag, _ in _passthroughs(value)}
        assert not flags & {"--temperature", "--top-p", "--max-tokens", "--max-concurrent"}, config.name
