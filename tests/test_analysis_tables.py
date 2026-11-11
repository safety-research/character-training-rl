"""The shipped analysis tables against the numbers the paper states, and the figures drawn from them (no network)."""

from __future__ import annotations

import csv
import json
import sys

import pytest

from analysis import common, s4_figures
from src.exp import manifest

TABLES = common.BROKEN_TASKS.parents[1] / "tables"
RATES = TABLES / "impossiblebench_rates.csv"
POOLED = TABLES / "impossiblebench_hack_rates_pooled.csv"
SORH = TABLES / "school_of_reward_hacks.csv"


def test_broken_task_list_counts():
    broken = common.load_broken_tasks()
    assert {sp: len(ids) for sp, ids in broken["ibswe"].items()} == {"original": 0, "oneoff": 55, "conflicting": 67}
    assert broken["iblcb"] == {"original": set(), "oneoff": {"lcbhard_77"}, "conflicting": {"lcbhard_77"}}
    rows = json.loads(common.BROKEN_TASKS.read_text())
    assert len(rows) == 124 and all(set(r) == {"benchmark", "split", "task_id"} for r in rows)


def test_impossible_swebench_task_counts():
    """Every Impossible-SWEbench run is graded on 294 one-off and 282 conflicting tasks once the broken tasks are
    left out, and on all 348 original tasks."""
    rows = [r for r in common.read_table(RATES, common.RATES_COLUMNS) if r["benchmark"] == "ibswe"]
    kept = [r for r in rows if r["task_set"] == common.BROKEN_LEFT_OUT]
    assert {(r["split"], r["samples"]) for r in kept} == {
        ("original", "348"),
        ("oneoff", "294"),
        ("conflicting", "282"),
    }
    assert {r["samples"] for r in rows if r["task_set"] == common.ALL_TASKS} == {"348"}
    assert {(r["split"], r["tasks_left_out"]) for r in kept} == {
        ("original", "0"),
        ("oneoff", "54"),
        ("conflicting", "66"),
    }


def test_rates_table_is_internally_consistent():
    rows = common.read_table(RATES, common.RATES_COLUMNS)
    assert len(rows) == 468
    for r in rows:
        assert r["benchmark"] in common.BENCHMARKS and r["model"] in common.MODELS and r["training"] in common.TRAININGS
        assert r["rate_kind"] == (common.SOLVE if r["split"] == common.ORIGINAL else common.HACK)
        assert (r["seed"] == "") == (r["training"] == common.BEFORE_RL)
        assert float(r["rate"]) == pytest.approx(int(r["passes"]) / int(r["graded"]))


@pytest.mark.parametrize(
    ("model", "expected"),
    [(common.GPT_OSS_20B, 20.0), (common.ASSIGNED_RULES, 17.4), (common.VALUE_GROUNDED_RULES, 14.3)],
)
def test_impossible_swebench_hack_rate_after_reward_hacking_rl(model, expected):
    rates = s4_figures.pooled_values(POOLED, "ibswe")[(model, common.REWARD_HACKING_RL)]
    assert len(rates) == (2 if model == common.ASSIGNED_RULES else 3)  # EXCLUDED_TRAININGS
    assert round(sum(rates) / len(rates), 1) == expected


@pytest.mark.parametrize(
    ("model", "expected"),
    [(common.GPT_OSS_20B, 51.2), (common.ASSIGNED_RULES, 49.4), (common.VALUE_GROUNDED_RULES, 52.6)],
)
def test_impossible_swebench_solve_rate_after_reward_hacking_rl(model, expected):
    rates = s4_figures.solve_values(RATES, "ibswe")[(model, common.REWARD_HACKING_RL)]
    assert len(rates) == (2 if model == common.ASSIGNED_RULES else 3)
    assert round(sum(rates) / len(rates), 1) == expected


def test_pooled_table_agrees_with_rates_table():
    """A pooled row's hacks and graded are the sums over the two impossible splits of the rates table."""
    sums = {}
    for r in common.read_table(RATES, common.RATES_COLUMNS):
        if r["task_set"] == common.BROKEN_LEFT_OUT and r["split"] in common.IMPOSSIBLE_SPLITS:
            key = (r["benchmark"], r["model"], r["training"], r["seed"])
            hacks, graded = sums.get(key, (0, 0))
            sums[key] = (hacks + int(r["passes"]), graded + int(r["graded"]))
    pooled = common.read_table(POOLED, common.POOLED_COLUMNS)
    excluded = {k for k in sums if not common.plotted(k[1], k[2], k[3])}
    assert len(excluded) == 3 and len(pooled) == len(sums) - len(excluded)
    for r in pooled:
        key = (r["benchmark"], r["model"], r["training"], r["seed"])
        assert (int(r["hacks"]), int(r["graded"])) == sums[key]
        if r["training"] not in common.SINGLE_RUN:
            assert float(r["hack_rate_pct"]) == pytest.approx(100 * int(r["hacks"]) / int(r["graded"]))
            assert r["ci95_low_pct"] == r["ci95_high_pct"] == ""


def test_figures_draw_from_the_shipped_tables(tmp_path, monkeypatch):
    monkeypatch.setattr(manifest, "RESULTS", tmp_path)
    tables = ["--rates", str(RATES), "--pooled", str(POOLED), "--school-of-reward-hacks", str(SORH)]
    monkeypatch.setattr(sys, "argv", ["s4_figures", "--figure", "all", *tables])
    s4_figures.main()
    for figure in ("rh_main", "rh_variants", "rh_honest", "rh_minimal"):
        pdf = manifest.script_dir(common.EXPERIMENT, s4_figures.SCRIPT) / f"{figure}.pdf"
        assert pdf.read_bytes().startswith(b"%PDF")


def test_school_of_reward_hacks_table_matches_the_paper():
    """Assigned rules before RL and after standard coding RL: 23% and 30% in the paper."""
    with SORH.open() as f:
        rows = {(r["model"], r["training"], r["seed"]): float(r["hack_rate"]) for r in csv.DictReader(f)}
    assert round(100 * rows[("assigned_rules", "before_rl", "")]) == 23
    assert round(100 * rows[("assigned_rules", "standard_coding_rl", "0")]) == 30
