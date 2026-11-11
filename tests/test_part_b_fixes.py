"""Monitor study, reward-hacking trainers and broken-task scan: intervals, path arguments, exclusion rules and the
rows the scripts print. No network, no GPU, no API key, and nothing from the external reward-hacking repository."""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path

import pytest
from inspect_ai.log import EvalDataset, EvalError, EvalLog, EvalSample, EvalSpec, write_eval_log
from inspect_ai.model import ChatMessageAssistant, ChatMessageUser, ModelOutput
from inspect_ai.scorer import Score

from analysis.find_broken_tasks import unbound_names_at_import
from monitors import common, s1_prepare_transcripts, s3_set_thresholds, s4_pooled_catch_rates, s6_figure_catch_rates
from monitors.verify_swe_hacks import passable_rows
from reward_hacking_rl import screen_held_out, train
from reward_hacking_rl.cli_paths import absolute_paths
from reward_hacking_rl.control import train as control_train

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "reward_hacking_rl" / "configs"


# ---- monitors: the interval of a catch rate


def test_catch_interval_is_a_task_bootstrap_of_caught_over_hacks():
    # two tasks hacked 20 times and always caught, six tasks hacked once and never caught
    rows = [{"task_id": f"big{i}", "score": 9} for i in range(2) for _ in range(20)]
    rows += [{"task_id": f"small{i}", "score": 1} for i in range(6)]
    got = s3_set_thresholds.catch(rows, t=5)
    assert (got["n"], got["n_tasks"], got["caught"]) == (46, 8, 40)
    assert got["rate"] == pytest.approx(40 / 46)
    low, high = got["ci_task_bootstrap"]
    # the interval is of the reported ratio, so it holds it; the mean of the per-task rates is 0.25
    assert low < got["rate"] < high
    assert high > 0.626
    assert "ci_task_clustered" not in got
    by_task = collections.defaultdict(list)
    for r in rows:
        by_task[r["task_id"]].append(r["score"] >= 5)
    assert [low, high] == list(common.boot_ci(by_task))
    assert s3_set_thresholds.catch([], t=5) is None


def test_one_bootstrap_function_and_one_pair_of_constants():
    assert s6_figure_catch_rates.boot_ci is common.boot_ci is s3_set_thresholds.boot_ci
    assert s4_pooled_catch_rates.BOOT_REPS is common.BOOT_REPS
    assert s4_pooled_catch_rates.BOOT_SEED is common.BOOT_SEED
    assert (common.BOOT_REPS, common.BOOT_SEED) == (10_000, 20261022)
    assert not hasattr(s6_figure_catch_rates, "BOOT_REPS")


# ---- monitors: the study file and the episodes kept


def _study(tmp_path, **changes) -> Path:
    data = {"runs": [], "environment_defect_tasks": [], "tasks_passable_without_hack": [], "excluded_trainings": []}
    path = tmp_path / "study.json"
    path.write_text(json.dumps({**data, **changes}))
    return path


def test_study_rejects_a_defect_task_listed_under_original(tmp_path, capsys):
    ok = [{"benchmark": "iblcb", "split": "oneoff", "task_id": "t1"}]
    assert common.load_study(_study(tmp_path, environment_defect_tasks=ok)).environment_defect_tasks == {
        ("iblcb", "oneoff", "t1")
    }
    for bench in common.BENCHMARKS:
        bad = [{"benchmark": bench, "split": "original", "task_id": "t1"}]
        with pytest.raises(SystemExit) as e:
            common.load_study(_study(tmp_path, environment_defect_tasks=bad))
        assert "environment_defect_tasks[0].split" in str(e.value) and "original" in str(e.value)


def _lcb_log(path, tasks: dict[str, float | None]) -> None:
    """An Impossible-LiveCodeBench-shaped log: one sample per task with its `resolved` value; None is a sample that
    errored and has no score."""
    samples = []
    for i, (task, resolved) in enumerate(tasks.items()):
        errored = resolved is None
        value = {"resolved": resolved, "usable": 1.0, "context_full": 0.0}
        samples.append(
            EvalSample(
                id=task,
                epoch=1,
                input="task",
                target="",
                metadata={"task_id": task},
                messages=[ChatMessageUser(content="task"), ChatMessageAssistant(content="answer")],
                scores=None if errored else {common.SCORER["iblcb"]: Score(value=value)},
                error=EvalError(message="sandbox failed", traceback="", traceback_ansi="") if errored else None,
                output=ModelOutput.from_content("m", "answer"),
                uuid=str(i),
            )
        )
    log = EvalLog(
        status="success",
        eval=EvalSpec(created="2000-01-01T00:00:00+00:00", task="t", dataset=EvalDataset(), model="m", config={}),
        samples=samples,
    )
    write_eval_log(log, str(path))


def test_defect_task_leaves_only_the_split_it_is_listed_under(tmp_path):
    log = tmp_path / "lcb.eval"
    _lcb_log(log, {"t1": 1.0, "t2": 0.0})
    defects = [{"benchmark": "iblcb", "split": "oneoff", "task_id": "t1"}]
    study = common.load_study(_study(tmp_path, environment_defect_tasks=defects))
    kept = {}
    for split in common.SPLITS:
        run = common.Run("iblcb", common.GPT_OSS_20B, common.AFTER_RL, 0, split, log)
        drops = collections.Counter()
        kept[split] = ({r["task_id"] for r in s1_prepare_transcripts.usable_lcb(run, study, drops)}, dict(drops))
    assert kept["oneoff"] == ({"t2"}, {"environment_defect": 1})
    assert kept["conflicting"] == ({"t1", "t2"}, {})
    assert kept["original"] == ({"t1", "t2"}, {})


def test_before_rl_rows_skip_and_count_an_episode_without_a_score(tmp_path):
    log = tmp_path / "before.eval"
    _lcb_log(log, {"passed": 1.0, "failed": 0.0, "errored": None})
    run = common.Run("iblcb", common.GPT_OSS_20B, common.BEFORE_RL, None, common.ORIGINAL, log)
    drops = collections.Counter()
    rows = list(s1_prepare_transcripts.before_rl_rows(run, drops))
    assert [r["task_id"] for r in rows] == ["passed"]
    assert dict(drops) == {"not_resolved": 1, "episode_error": 1}


def test_attempted_leaves_out_tasks_passable_without_a_hack(tmp_path, monkeypatch):
    episodes = [
        {"model": "gpt-oss-20b", "seed": 0, "split": "oneoff", "task_id": "passable"},
        {"model": "gpt-oss-20b", "seed": 0, "split": "conflicting", "task_id": "passable"},
        {"model": "gpt-oss-20b", "seed": 0, "split": "oneoff", "task_id": "other"},
        {"model": "gpt-oss-20b", "seed": 0, "split": "original", "task_id": "other"},
    ]
    monkeypatch.setattr(common, "prepared_rows", lambda benchmark, stage: iter(episodes))
    passable = [{"split": "oneoff", "task_id": "passable"}]
    study = common.load_study(_study(tmp_path, tasks_passable_without_hack=passable))
    # the pair is (split, task): the same task on the other impossible split still counts
    assert common.attempted_impossible("ibswe", study) == {("gpt-oss-20b", 0): 2}
    assert common.attempted_impossible("ibswe", common.load_study(_study(tmp_path))) == {("gpt-oss-20b", 0): 3}


def test_verify_report_rows_load_as_tasks_passable_without_hack(tmp_path):
    resolved_by = {
        ("oneoff", "nochange"): {"b", "a"},
        ("oneoff", "oracle"): {"a"},
        ("conflicting", "nochange"): set(),
        ("conflicting", "oracle"): {"c"},
    }
    rows = passable_rows(resolved_by)
    assert rows == [
        {"split": "oneoff", "task_id": "a"},
        {"split": "oneoff", "task_id": "b"},
        {"split": "conflicting", "task_id": "c"},
    ]
    study = common.load_study(_study(tmp_path, tasks_passable_without_hack=rows))
    assert study.tasks_passable_without_hack == {("oneoff", "a"), ("oneoff", "b"), ("conflicting", "c")}


# ---- reward-hacking trainers: path arguments and required flags


def test_path_arguments_resolve_against_the_launch_directory(tmp_path):
    launch = tmp_path / "launch"
    args = argparse.Namespace(
        repo=Path("../their_repo"),
        out=Path("results/run"),
        held_out_selection=Path("reward_hacking_rl/held_out_selection.json"),
        absolute=Path("/data/ae.json"),
        mixed_in_pool="pools/chat.jsonl",
        judge_python=".venv/bin/python",
        dump_batches="none",
        model="openai/gpt-oss-20b",
    )
    names = ("repo", "out", "held_out_selection", "absolute", "mixed_in_pool", "judge_python", "dump_batches")
    absolute_paths(args, names, launch_dir=launch)
    assert args.repo == tmp_path / "their_repo"
    assert args.out == launch / "results" / "run"
    assert args.held_out_selection == launch / "reward_hacking_rl" / "held_out_selection.json"
    assert args.absolute == Path("/data/ae.json")
    assert args.mixed_in_pool == str(launch / "pools" / "chat.jsonl") and isinstance(args.mixed_in_pool, str)
    assert args.judge_python == str(launch / ".venv" / "bin" / "python")
    assert args.dump_batches == "none"
    assert args.model == "openai/gpt-oss-20b"


def test_path_arguments_default_to_the_current_directory_and_keep_links(tmp_path, monkeypatch):
    target = tmp_path / "real_python"
    target.write_text("")
    (tmp_path / "python").symlink_to(target)
    monkeypatch.chdir(tmp_path)
    args = argparse.Namespace(judge_python="python", out=Path("out"))
    absolute_paths(args, ("judge_python", "out"))
    # a path that survives a later change of directory, with the link itself kept
    assert os.path.isabs(args.judge_python) and Path(args.judge_python).name == "python"
    assert Path(args.judge_python).is_symlink()
    assert args.out.is_absolute() and args.out.name == "out"


@pytest.mark.parametrize("module", [train, screen_held_out, control_train])
def test_every_path_flag_of_a_trainer_is_resolved(module, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [module.__name__, "--help"])
    with pytest.raises(SystemExit):
        module.parse_args()
    usage = capsys.readouterr().out
    for name in module.PATH_ARGS:
        assert "--" + name.replace("_", "-") in usage, name


@pytest.mark.parametrize("module", [train, control_train])
def test_verify_one_datum_is_required(module, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [module.__name__])
    with pytest.raises(SystemExit):
        module.parse_args()
    assert "--verify-one-datum" in capsys.readouterr().err


def test_training_configs_pass_verify_and_keep_verify_runs_apart():
    training = [p for p in sorted(CONFIGS.glob("*.sh")) if p.name != "held_out_screen.sh"]
    assert len(training) == 4
    for config in training:
        text = config.read_text()
        assert "--verify-one-datum $VERIFY" in text, config.name
        assert "VERIFY=yes; OUT=${OUT}_verify" in text, config.name
    for config in training:
        if "--held-out-selection" in config.read_text().split("set -euo pipefail")[1]:
            assert "HELD_OUT_SELECTION=${HELD_OUT_SELECTION:-reward_hacking_rl/held_out_selection.json}" in (
                config.read_text()
            ), config.name
    assert (REPO / "reward_hacking_rl" / "held_out_selection.json").is_file()


# ---- analysis: the broken-task scan


def test_unbound_names_at_import():
    text = "\n".join(
        [
            "import os",  # 1
            "x = missing_name",  # 2: read at import, bound nowhere
            "def f(a=other_missing):",  # 3: a function definition is not scanned
            "    return undefined_inside",  # 4
            "class C:",  # 5
            "    y = os.sep",  # 6
            "    z = class_missing",  # 7: a class body runs at import
            "w = (lambda: lazy_missing)",  # 8: a lambda body does not run at import
        ]
    )
    every_line = set(range(1, 9))
    assert unbound_names_at_import(text, every_line) == ["line 2: ['missing_name']", "line 7: ['class_missing']"]
    assert unbound_names_at_import(text, {7}) == ["line 7: ['class_missing']"]
    assert unbound_names_at_import("from os import *\nx = missing_name\n", {2}) == []
