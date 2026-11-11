"""Figure: the share of each group's ImpossibleBench hacks that each monitor catches, for gpt-oss-20b against the
character-trained models pooled (assigned rules and value-grounded rules, every seed).

Hacks are read as s4_pooled_catch_rates reads them (impossible-split passes after reward-hacking RL, less the study
file's `tasks_passable_without_hack`), each monitor at its own thresholds from s3_set_thresholds (at most 2% of the
before-RL original-split passes flagged). Over every training, each catch rate is first checked to equal
pooled.json's. The bars then leave out the trainings in the study file's `excluded_trainings`; the number of hacks
left out is printed. The thresholds are set on the models before reward-hacking RL, so they do not change. The
interval on each bar is a task-cluster bootstrap of that group's caught/hacks (monitors.common.boot_ci). An
excluded training with no hack on a benchmark leaves that panel as it is. Two panels stacked at column width
(Impossible-LiveCodeBench above, Impossible-SWEbench below); a group per monitor; per view (reasoning shown, then
the lighter reasoning hidden), a gpt-oss-20b bar beside a character-trained bar.

    uv run python -m monitors.s6_figure_catch_rates --study monitors/study.json

Writes results/monitors/scripts/s6_figure_catch_rates/{monitor_catch.png, monitor_catch.pdf, values.csv}.
No API calls.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from monitors import bars
from monitors.common import (
    BENCHMARKS,
    EXPERIMENT,
    GROUPS,
    MONITOR_NAMES,
    MONITORS,
    POOLED,
    VIEW_NAMES,
    boot_ci,
    hacks,
    load_study,
    scores_by_monitor_view,
    summary_file,
)
from monitors.transcript import REASONING_SHOWN
from src.exp.manifest import script_dir
from src.plotting import lighten

SCRIPT = "s6_figure_catch_rates"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--study", type=Path, required=True, help="the study file (see monitors/README.md)")
    study = load_study(ap.parse_args().study)
    excluded = study.excluded_trainings
    pooled = json.loads((script_dir(EXPERIMENT, POOLED) / "pooled.json").read_text())
    verdicts = scores_by_monitor_view()
    out = script_dir(EXPERIMENT, SCRIPT)
    # by view, then group, so each gpt-oss-20b bar sits beside the character-trained bar it is compared with
    series = [f"{g}, {label}" for label in VIEW_NAMES.values() for g in GROUPS]
    colors = {
        f"{g}, {label}": bars.GROUP_COLOR[g] if view == REASONING_SHOWN else lighten(bars.GROUP_COLOR[g])
        for g in GROUPS
        for view, label in VIEW_NAMES.items()
    }
    bars.print_style()
    fig, axes = plt.subplots(
        len(BENCHMARKS), 1, figsize=(bars.COL_W, 1.2 * len(BENCHMARKS) + bars.LEGEND_H * 2), sharex=True
    )
    rows = []
    for ax, (bench, title) in zip(axes, BENCHMARKS.items()):
        pos_all = hacks(bench, study)
        pos = [e for e in pos_all if (e["model"], e["seed"]) not in excluded]
        n_ex = len(pos_all) - len(pos)
        print(f"{bench}: left out {n_ex} hacks of {sorted(excluded)} (excluded_trainings), {len(pos)} remain")
        values = {}
        for m, mname in MONITOR_NAMES.items():
            summ = json.loads(summary_file(m, bench).read_text())
            for view, label in VIEW_NAMES.items():
                t = summ["views"][view]["threshold"]
                sc = verdicts[(MONITORS[m], view)]
                for g, models in GROUPS.items():
                    # over every training, the reading must reproduce pooled.json exactly
                    full = [sc[e["id"]] >= t for e in pos_all if e["model"] in models and e["id"] in sc]
                    expect = pooled[bench][g]["monitors"][m][view]
                    assert (
                        len(full) == expect["scored"] and abs(sum(full) / len(full) - expect["catch_rate"]) < 1e-12
                    ), (bench, m, view, g)
                    by_task = defaultdict(list)
                    for e in pos:
                        if e["model"] in models and e["id"] in sc:
                            by_task[e["task_id"]].append(sc[e["id"]] >= t)
                    n = sum(len(v) for v in by_task.values())
                    rate = sum(sum(v) for v in by_task.values()) / n
                    lo, hi = boot_ci(by_task)
                    values[(mname, f"{g}, {label}")] = (100 * rate, 100 * lo, 100 * hi)
                    rows.append(
                        {
                            "benchmark": bench,
                            "monitor": m,
                            "view": view,
                            "group": g,
                            "threshold": t,
                            "hacks": n,
                            "caught_pct": f"{100 * rate:.2f}",
                            "lo": f"{100 * lo:.2f}",
                            "hi": f"{100 * hi:.2f}",
                        }
                    )
        bars.panel(ax, list(MONITOR_NAMES.values()), series, values, title, colors)
        ax.set_ylim(0, 100)
    for ax in axes:
        ax.set_ylabel("Hacks caught (%)")
        bars.higher_better(ax)
    bars.letters(axes)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[s]) for s in series]
    fig.legend(
        handles,
        series,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0),
        handlelength=1.2,
        columnspacing=1.2,
        handletextpad=0.5,
    )
    bars.save(fig, out / "monitor_catch.png", legend_rows=2)
    with (out / "values.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
