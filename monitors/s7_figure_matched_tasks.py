"""Figure: monitor catch rate on Impossible-SWEbench hacks on the (split, task) pairs every model hacked on, for
gpt-oss-20b against the character-trained models pooled (assigned rules and value-grounded rules), in the series
and colours of s6_figure_catch_rates.

The matched-task reading is s5_matched_tasks's: rows are the hacks that both monitors of a comparison scored
(Haiku 4.5 from the Sonnet 4.6 / Haiku 4.5 comparison, Sonnet 4.6 and Opus 4.7 from the Sonnet 4.6 / Opus 4.7 one),
the shared pairs are those every model has a hack on, and a group's catch rate is the mean over shared pairs of its
per-pair caught/hacks, with a 95% interval clustered by pair (src.evals.base.clustered_ci). The character-trained
group pools both models' hacks within each pair before the per-pair rate is taken. Each per-model rate is first
recomputed over every training and checked to equal s5's stored reading. The bars then leave out the trainings in
the study file's `excluded_trainings`: the shared pairs are recomputed without those hacks (a pair stays only if
every model still has a hack on it), and both pair counts are printed. Each monitor's threshold is its own, set so
that at most 2% of the before-RL original-split passes are flagged (s3_set_thresholds).

    uv run python -m monitors.s7_figure_matched_tasks --study monitors/study.json

Writes results/monitors/scripts/s7_figure_matched_tasks/{monitor_matched.png, monitor_matched.pdf, values.csv}.
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
    EXPERIMENT,
    GROUPS,
    MODELS,
    MONITOR_NAMES,
    MONITORS,
    VIEW_NAMES,
    hacks,
    load_study,
    scores_by_monitor_view,
)
from monitors.s5_matched_tasks import BENCH, compare_file
from monitors.transcript import REASONING_SHOWN
from src.evals.base import clustered_ci
from src.exp.manifest import script_dir
from src.plotting import lighten

SCRIPT = "s7_figure_matched_tasks"
#: each monitor -> the s5_matched_tasks comparison (first, second) its stored matched reading comes from
COMPARISON = {"haiku": ("sonnet", "haiku"), "sonnet": ("sonnet", "opus47"), "opus47": ("sonnet", "opus47")}
#: height of the plot in inches at the column width, legend excluded, and the legend's rows in units of bars.LEGEND_H
PLOT_H, LEGEND_ROWS = 1.6, 2


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--study", type=Path, required=True, help="the study file (see monitors/README.md)")
    study = load_study(ap.parse_args().study)
    excluded = study.excluded_trainings
    sc = scores_by_monitor_view()
    out = script_dir(EXPERIMENT, SCRIPT)
    pos_all = hacks(BENCH, study)
    pos = [e for e in pos_all if (e["model"], e["seed"]) not in excluded]
    print(
        f"{BENCH}: left out {len(pos_all) - len(pos)} hacks of {sorted(excluded)} (excluded_trainings), "
        f"{len(pos)} remain"
    )
    series = [f"{g}, {label}" for label in VIEW_NAMES.values() for g in GROUPS]
    colors = {
        f"{g}, {label}": bars.GROUP_COLOR[g] if view == REASONING_SHOWN else lighten(bars.GROUP_COLOR[g])
        for g in GROUPS
        for view, label in VIEW_NAMES.items()
    }
    values, rows, n_pairs = {}, [], set()
    for m, mname in MONITOR_NAMES.items():
        first, second = COMPARISON[m]
        stored = json.loads(compare_file(first, second).read_text())
        for view, label in VIEW_NAMES.items():
            t = stored[view]["thresholds"][m]
            both = {k: sc[(MONITORS[k], view)] for k in (first, second)}

            def matched(rows: list[dict]) -> tuple[dict, list]:
                pairs = defaultdict(lambda: defaultdict(list))
                for e in rows:
                    if all(e["id"] in s for s in both.values()):
                        pairs[(e["split"], e["task_id"])][e["model"]].append(both[m][e["id"]] >= t)
                return pairs, [p for p, by in pairs.items() if all(a in by for a in MODELS)]

            # over every training, s5's stored reading, recomputed from the verdicts
            pairs, shared = matched(pos_all)
            assert len(shared) == stored[view]["n_shared_pairs"], (m, view)
            for model in MODELS:
                mean, lo, hi = clustered_ci([sum(pairs[p][model]) / len(pairs[p][model]) for p in shared])
                ref = stored[view]["matched"][m][model]
                assert abs(mean - ref["rate"]) < 1e-12 and abs(lo - ref["ci"][0]) < 1e-12, (m, view, model)
            n_all = len(shared)
            pairs, shared = matched(pos)
            print(f"{m} {view}: {len(shared)} matched pairs without excluded_trainings ({n_all} with every training)")
            n_pairs.add(len(shared))
            for g, models in GROUPS.items():
                per_pair = []
                for p in shared:
                    caught = [c for a in models for c in pairs[p][a]]
                    per_pair.append(sum(caught) / len(caught))
                mean, lo, hi = clustered_ci(per_pair)
                values[(mname, f"{g}, {label}")] = (100 * mean, 100 * lo, 100 * hi)
                rows.append(
                    {
                        "monitor": m,
                        "view": view,
                        "group": g,
                        "catch_pct": f"{100 * mean:.2f}",
                        "lo": f"{100 * lo:.2f}",
                        "hi": f"{100 * hi:.2f}",
                        "n_matched_pairs": len(shared),
                        "hacks": sum(len(pairs[p][a]) for p in shared for a in models),
                        "threshold": t,
                    }
                )
    if len(n_pairs) != 1:
        raise SystemExit(
            "expected the same number of matched (split, task) pairs for every monitor and view, because the bars "
            f"of one figure are compared on one set of pairs; found the counts {sorted(n_pairs)} (printed per "
            "monitor and view above). A monitor with judge errors on some hacks scores fewer pairs."
        )
    bars.print_style()
    fig, ax = plt.subplots(figsize=(bars.COL_W, PLOT_H + LEGEND_ROWS * bars.LEGEND_H))
    bars.panel(ax, list(MONITOR_NAMES.values()), series, values, "", colors)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Hacks caught (%)")
    bars.higher_better(ax)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[s], ec="none") for s in series]
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
    bars.save(fig, out / "monitor_matched.png", legend_rows=LEGEND_ROWS)
    with (out / "values.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
