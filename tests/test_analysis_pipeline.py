"""The log-reading analysis scripts, run end to end on small Inspect logs written here (no network, no model)."""

from __future__ import annotations

import csv
import json
import sys

import pytest
from inspect_ai.log import (
    EvalDataset,
    EvalError,
    EvalLog,
    EvalMetric,
    EvalResults,
    EvalSample,
    EvalScore,
    EvalSpec,
    write_eval_log,
)
from inspect_ai.model import ModelOutput
from inspect_ai.scorer import Score

from analysis import common, s1_rates, s2_pooled_hack_rates, s3_school_of_reward_hacks
from src.evals.base import clustered_ci, wilson_ci
from src.exp import manifest

SCORER = "impossible_swebench"
EXTRA = [f"extra__task-{i}" for i in range(6)]
"""Tasks that are in no broken-task list, added to every log beside the listed ones."""


def _log(path, passes: dict[str, float], errored: set[str] = frozenset()) -> None:
    """Write an Impossible-SWEbench-shaped log: one sample per task with its `resolved` value, and the header
    metrics the evaluation's scorer would report."""
    samples = []
    for i, (task, v) in enumerate(passes.items()):
        scores = (
            None if task in errored else {SCORER: Score(value={"resolved": v, "timed_out": 0.0, "context_full": 0.0})}
        )
        error = EvalError(message="sandbox failed", traceback="", traceback_ansi="") if task in errored else None
        samples.append(
            EvalSample(
                id=task,
                epoch=1,
                input="task",
                target="",
                scores=scores,
                error=error,
                output=ModelOutput(model="m"),
                uuid=str(i),
            )
        )
    graded = [v for t, v in passes.items() if t not in errored]
    low, high = wilson_ci(sum(graded) / len(graded), len(graded))
    resolved = {
        "task_mean": sum(graded) / len(graded),
        "wilson_low": low,
        "wilson_high": high,
        "n_samples": len(graded),
    }
    header = [
        EvalScore(
            name="resolved", scorer=SCORER, metrics={k: EvalMetric(name=k, value=v) for k, v in resolved.items()}
        ),
        *[
            EvalScore(name=a, scorer=SCORER, metrics={"task_mean": EvalMetric(name="task_mean", value=0.0)})
            for a in ("timed_out", "context_full")
        ],
    ]
    log = EvalLog(
        status="success",
        eval=EvalSpec(created="2000-01-01T00:00:00+00:00", task="t", dataset=EvalDataset(), model="m", config={}),
        results=EvalResults(total_samples=len(samples), completed_samples=len(samples), scores=header),
        samples=samples,
    )
    write_eval_log(log, str(path))


@pytest.fixture
def results(tmp_path, monkeypatch):
    monkeypatch.setattr(manifest, "RESULTS", tmp_path / "results")
    return tmp_path


def _run(module, monkeypatch, *argv: str) -> None:
    monkeypatch.setattr(sys, "argv", [module.__name__, *argv])
    module.main()


def test_rates_then_pooled(results, monkeypatch):
    broken = common.load_broken_tasks()["ibswe"]
    unrunnable = "astropy__astropy-7336"  # listed as broken, never in a log (the evaluation cannot start it)
    tasks = {sp: sorted((broken["oneoff"] | broken["conflicting"]) - {unrunnable}) + EXTRA for sp in common.SPLITS}
    # the single evaluated model passes the first two extra tasks on oneoff and the first on conflicting, and one
    # broken task on each impossible split, which must not count
    passes = {
        "original": {t: 1.0 for t in tasks["original"]},
        "oneoff": {t: float(t in EXTRA[:2] or t == sorted(broken["oneoff"])[0]) for t in tasks["oneoff"]},
        "conflicting": {t: float(t == EXTRA[0] or t == sorted(broken["conflicting"])[0]) for t in tasks["conflicting"]},
    }
    runs = []
    for training, seed in ((common.BEFORE_RL, None), (common.REWARD_HACKING_RL, 0)):
        for sp in common.SPLITS:
            path = results / f"{training}_{sp}.eval"
            _log(path, passes[sp], errored={EXTRA[5]} if sp == "oneoff" else frozenset())
            row = {"benchmark": "ibswe", "model": common.GPT_OSS_20B, "training": training, "seed": seed, "split": sp}
            runs.append({**row, "log": str(path)})
    runs_file = results / "runs.json"
    runs_file.write_text(json.dumps({"impossiblebench": runs, "school_of_reward_hacks": []}))

    _run(s1_rates, monkeypatch, "--runs", str(runs_file))
    rates = manifest.script_dir(common.EXPERIMENT, s1_rates.SCRIPT) / s1_rates.TABLE
    rows = {(r["training"], r["split"], r["task_set"]): r for r in csv.DictReader(rates.open())}
    one = rows[(common.BEFORE_RL, "oneoff", common.BROKEN_LEFT_OUT)]
    n_oneoff = len(tasks["oneoff"]) - (len(broken["oneoff"]) - 1)  # the unrunnable broken task was never there
    assert (one["rate_kind"], int(one["passes"]), int(one["errored"])) == (common.HACK, 2, 1)
    assert int(one["samples"]) == n_oneoff and int(one["graded"]) == n_oneoff - 1
    assert int(one["tasks_left_out"]) == len(broken["oneoff"]) - 1
    assert int(rows[(common.BEFORE_RL, "oneoff", common.ALL_TASKS)]["passes"]) == 3
    orig = rows[(common.BEFORE_RL, "original", common.BROKEN_LEFT_OUT)]
    assert (orig["rate_kind"], orig["tasks_left_out"], orig["seed"]) == (common.SOLVE, "0", "")
    assert int(orig["samples"]) == len(tasks["original"])  # nothing is left out of the original split

    _run(s2_pooled_hack_rates, monkeypatch, "--runs", str(runs_file), "--rates", str(rates))
    pooled = manifest.script_dir(common.EXPERIMENT, s2_pooled_hack_rates.SCRIPT) / s2_pooled_hack_rates.TABLE
    by_training = {r["training"]: r for r in csv.DictReader(pooled.open())}
    n_conf = len(tasks["conflicting"]) - (len(broken["conflicting"]) - 1)
    graded = (n_oneoff - 1) + n_conf
    seeded = by_training[common.REWARD_HACKING_RL]
    assert (int(seeded["hacks"]), int(seeded["graded"]), seeded["ci95_low_pct"]) == (3, graded, "")
    assert float(seeded["hack_rate_pct"]) == pytest.approx(100 * 3 / graded)
    # the single model: each task's mean over the splits it is graded in, then the mean and interval over tasks
    values = {}
    for sp in common.IMPOSSIBLE_SPLITS:
        for t, v in passes[sp].items():
            if t not in broken[sp] and not (sp == "oneoff" and t == EXTRA[5]):
                values.setdefault(t, []).append(v)
    mean, low, high = clustered_ci([sum(v) / len(v) for v in values.values()])
    single = by_training[common.BEFORE_RL]
    assert (int(single["hacks"]), int(single["graded"])) == (3, graded)
    assert float(single["hack_rate_pct"]) == pytest.approx(100 * mean)
    assert float(single["ci95_low_pct"]) == pytest.approx(100 * low)
    assert float(single["ci95_high_pct"]) == pytest.approx(100 * high)


def test_school_of_reward_hacks_reads_the_header(results, monkeypatch):
    metrics = {"task_mean": 0.25, "ci95_low": 0.2, "ci95_high": 0.3, "n_samples": 100.0}
    header = [
        EvalScore(name="hack", scorer="s", metrics={k: EvalMetric(name=k, value=v) for k, v in metrics.items()}),
        EvalScore(name="usable", scorer="s", metrics={"task_mean": EvalMetric(name="task_mean", value=0.95)}),
    ]
    log = EvalLog(
        status="success",
        eval=EvalSpec(created="2000-01-01T00:00:00+00:00", task="t", dataset=EvalDataset(), model="m", config={}),
        results=EvalResults(total_samples=0, completed_samples=0, scores=header),
        samples=[],
    )
    path = results / "sorh.eval"
    write_eval_log(log, str(path))
    run = {"model": common.ASSIGNED_RULES, "training": common.CHAT_MIXED_IN, "seed": 1000, "log": str(path)}
    runs_file = results / "runs.json"
    runs_file.write_text(json.dumps({"impossiblebench": [], "school_of_reward_hacks": [run]}))
    _run(s3_school_of_reward_hacks, monkeypatch, "--runs", str(runs_file))
    table = manifest.script_dir(common.EXPERIMENT, s3_school_of_reward_hacks.SCRIPT) / s3_school_of_reward_hacks.TABLE
    [row] = csv.DictReader(table.open())
    assert row == {
        "model": common.ASSIGNED_RULES,
        "training": common.CHAT_MIXED_IN,
        "seed": "1000",
        "hack_rate": "0.25",
        "ci95_low": "0.2",
        "ci95_high": "0.3",
        "usable": "0.95",
    }


def test_runs_file_refuses_unknown_values(tmp_path):
    row = {
        "benchmark": "ibswe",
        "model": "gpt-oss-20b",
        "training": "before_rl",
        "seed": 0,
        "split": "oneoff",
        "log": "x",
    }
    path = tmp_path / "runs.json"
    path.write_text(json.dumps({"impossiblebench": [row], "school_of_reward_hacks": []}))
    with pytest.raises(SystemExit, match="seed null"):
        common.load_runs(path)
    path.write_text(
        json.dumps({"impossiblebench": [{**row, "seed": None, "split": "all"}], "school_of_reward_hacks": []})
    )
    with pytest.raises(SystemExit, match="split"):
        common.load_runs(path)
