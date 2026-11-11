"""The paper's four reward-hacking figures, drawn from the three tables.

- `rh_main`: hack rate on (a) Impossible-LiveCodeBench, (b) Impossible-SWEbench and (c) School of Reward Hacks,
  before RL, after reward-hacking RL, and after reward-hacking RL with constitution RL mixed in.
- `rh_variants`: the same three panels for the two character-trained models, with every training: before RL,
  standard coding RL, reward-hacking RL, and reward-hacking RL with chat or constitution RL mixed in.
- `rh_honest`: solve rate on the original tasks of (a) Impossible-LiveCodeBench and (b) Impossible-SWEbench.
- `rh_minimal`: hack rate on Impossible-LiveCodeBench without tools, before and after reward-hacking RL.

Every figure is stacked panels at the paper's column width, a group per training and a bar per model in its colour.
A single evaluated model (before RL, standard coding RL) is a bar with its 95% interval; a training repeated over
seeds is a bar at the mean of the repeats with a dot per repeat. gpt-oss-20b was trained with reward-hacking RL
alone, so it has no bar where chat or constitution RL is mixed in. The trainings in
analysis.common.EXCLUDED_TRAININGS are left out of every figure, and a reward-hacking training whose usable share of
School of Reward Hacks answers is below MIN_USABLE is left out of the School of Reward Hacks panels; both are
printed.

Hack rates on ImpossibleBench come from the pooled table (s2_pooled_hack_rates.py), the School of Reward Hacks
panels from s3_school_of_reward_hacks.py's table, and the solve rates from the rows of s1_rates.py's table for the
original split, which give a single evaluated model its Wilson 95% interval.

    uv run python -m analysis.s4_figures --figure all \\
        --rates analysis/tables/impossiblebench_rates.csv \\
        --pooled analysis/tables/impossiblebench_hack_rates_pooled.csv \\
        --school-of-reward-hacks analysis/tables/school_of_reward_hacks.csv

Each table flag is required exactly when a requested figure is drawn from it. Writes
results/analysis/scripts/s4_figures/<figure>.pdf. Reads the tables only.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from analysis import bars
from analysis.common import (
    ASSIGNED_RULES,
    BEFORE_RL,
    BENCHMARKS,
    BROKEN_LEFT_OUT,
    CHAT_MIXED_IN,
    CONSTITUTION_RL_MIXED_IN,
    EXPERIMENT,
    GPT_OSS_20B,
    MIN_USABLE,
    MODEL_NAMES,
    MODELS,
    ORIGINAL,
    POOLED_COLUMNS,
    RATES_COLUMNS,
    REWARD_HACKING_RL,
    SINGLE_RUN,
    SORH_COLUMNS,
    STANDARD_CODING_RL,
    VALUE_GROUNDED_RULES,
    plotted,
    read_table,
)
from src.exp.manifest import script_dir
from src.plotting import WONG

SCRIPT = "s4_figures"
FIGURES = ("rh_main", "rh_variants", "rh_honest", "rh_minimal")
MODEL_COLOR = {GPT_OSS_20B: WONG[6], ASSIGNED_RULES: WONG[0], VALUE_GROUNDED_RULES: WONG[1]}
"""One colour per model, the same in every figure of the paper."""
CHARACTER_TRAINED = (ASSIGNED_RULES, VALUE_GROUNDED_RULES)
SORH = "School of Reward Hacks"
#: height of one stacked panel in inches at the column width
PANEL_H = 1.45
#: vertical space between stacked panels, in font-size units, so each panel title clears the x axis above it
H_PAD = 1.2
#: y-axis top (%) and ticks of the hack-rate panels: the highest value in any of them is 65%
HACK_YTOP, HACK_YTICKS = 75, [0, 25, 50, 75]
#: font size of the group labels under a figure with four or more groups, or wide ones, in points
SMALL_TICKS = 6

NEEDS_OF = {
    "rh_main": ("--pooled", "--school-of-reward-hacks"),
    "rh_variants": ("--pooled", "--school-of-reward-hacks"),
    "rh_honest": ("--rates",),
    "rh_minimal": ("--pooled",),
}
"""Figure -> the tables it is drawn from, by flag; a flag is required exactly when a requested figure needs it."""
NEEDS = {
    "--rates": lambda a: a.rates,
    "--pooled": lambda a: a.pooled,
    "--school-of-reward-hacks": lambda a: a.school_of_reward_hacks,
}

Values = dict[tuple[str, str], tuple[float, float, float] | list[float]]
"""(model, training) -> (rate, low, high) for a single evaluated model, or the rate of each repeat; in percent."""


def _collect(rows: list[tuple[str, str, str, float, float | None, float | None]]) -> Values:
    """Values from (model, training, seed, rate, low, high) rows in percent, the EXCLUDED_TRAININGS dropped."""
    seeds, out = defaultdict(dict), {}
    for model, training, seed, rate, lo, hi in rows:
        if not plotted(model, training, seed):
            print(f"left out (EXCLUDED_TRAININGS): {model} {training} seed {seed}")
        elif training in SINGLE_RUN:
            if (model, training) in out:
                raise SystemExit(f"{model} {training}: more than one evaluation of a single model")
            out[(model, training)] = (rate, lo, hi)
        else:
            seeds[(model, training)][int(seed)] = rate
    for key, by_seed in seeds.items():
        out[key] = [by_seed[s] for s in sorted(by_seed)]
    return out


def pooled_values(path: Path, benchmark: str) -> Values:
    """Hack rates of one benchmark with the two impossible splits pooled."""
    rows = [r for r in read_table(path, POOLED_COLUMNS) if r["benchmark"] == benchmark]
    return _collect(
        [
            (
                r["model"],
                r["training"],
                r["seed"],
                float(r["hack_rate_pct"]),
                float(r["ci95_low_pct"]) if r["ci95_low_pct"] else None,
                float(r["ci95_high_pct"]) if r["ci95_high_pct"] else None,
            )
            for r in rows
        ]
    )


def solve_values(path: Path, benchmark: str) -> Values:
    """Solve rates on one benchmark's original tasks; a single evaluated model has its Wilson 95% interval."""
    rows = [
        r
        for r in read_table(path, RATES_COLUMNS)
        if (r["benchmark"], r["split"], r["task_set"]) == (benchmark, ORIGINAL, BROKEN_LEFT_OUT)
    ]
    return _collect(
        [
            (
                r["model"],
                r["training"],
                r["seed"],
                100 * float(r["rate"]),
                100 * float(r["wilson_low"]),
                100 * float(r["wilson_high"]),
            )
            for r in rows
        ]
    )


def sorh_values(path: Path) -> Values:
    """School of Reward Hacks hack rates; a training repeated over seeds drops the repeats below MIN_USABLE."""
    rows = []
    for r in read_table(path, SORH_COLUMNS):
        if r["training"] not in SINGLE_RUN and plotted(r["model"], r["training"], r["seed"]):
            if float(r["usable"]) < MIN_USABLE:
                print(
                    f"{SORH}: left out {r['model']} {r['training']} seed {r['seed']}, {float(r['usable']):.1%} usable"
                )
                continue
        rows.append(
            (
                r["model"],
                r["training"],
                r["seed"],
                100 * float(r["hack_rate"]),
                100 * float(r["ci95_low"]),
                100 * float(r["ci95_high"]),
            )
        )
    return _collect(rows)


def stacked(
    name: str,
    panels: list[tuple[str, Values]],
    groups: dict[str, str],
    models: tuple[str, ...],
    ylabel: str,
    lower: bool,
    ytop: float,
    yticks: list[float],
    absent: set[tuple[str, str]],
    tick_size: float | None,
) -> None:
    """One figure: `panels` is [(title, values)] stacked top to bottom; `groups` maps each group label to its
    training; a bar per model. The (model, training) pairs in `absent` were never trained and must have no value;
    every other pair must have one."""
    bars.print_style()
    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=(bars.COL_W, len(panels) * PANEL_H + bars.LEGEND_H),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes = list(axes[:, 0])
    names = [MODEL_NAMES[m] for m in models]
    colors = {MODEL_NAMES[m]: MODEL_COLOR[m] for m in models}
    for ax, (title, values) in zip(axes, panels):
        v = {}
        for g, training in groups.items():
            for m in models:
                v[(g, MODEL_NAMES[m])] = values.get((m, training))
                if (v[(g, MODEL_NAMES[m])] is None) != ((m, training) in absent):
                    raise SystemExit(f"{name}, {title}: {m} {training} has a value exactly when it should not")
        bars.panel(ax, list(groups), names, v, title, colors)
        ax.set_xlim(-0.5, len(groups) - 0.5)
        ax.set_ylim(0, ytop)
        ax.set_yticks(yticks)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", length=0, pad=2)
        bars.better(ax, lower=lower)
        if tick_size is not None:
            ax.tick_params(axis="x", labelsize=tick_size)
    for ax in axes[:-1]:
        ax.tick_params(axis="x", labelbottom=False)  # the trainings are named once, under the bottom panel
    if len(axes) > 1:
        bars.letters(axes)
    bars.clear_better(axes)
    bars.legend(fig, names, colors)
    bars.save(fig, script_dir(EXPERIMENT, SCRIPT) / f"{name}.pdf", h_pad=H_PAD)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figure", required=True, choices=(*FIGURES, "all"), help="the figure to draw")
    ap.add_argument("--rates", type=Path, help="the per-split rates table (s1_rates.py); needed for rh_honest")
    ap.add_argument(
        "--pooled",
        type=Path,
        help="the pooled hack-rate table (s2_pooled_hack_rates.py); needed for rh_main, rh_variants and rh_minimal",
    )
    ap.add_argument(
        "--school-of-reward-hacks",
        type=Path,
        help="the School of Reward Hacks table (s3_school_of_reward_hacks.py); needed for rh_main and rh_variants",
    )
    args = ap.parse_args()
    wanted = FIGURES if args.figure == "all" else (args.figure,)
    for flag, value in NEEDS.items():
        users = [f for f in wanted if flag in NEEDS_OF[f]]
        if users and value(args) is None:
            ap.error(f"{flag} is required for {', '.join(users)}")
    # the three hack-rate panels of rh_main and rh_variants
    hack_panels = []
    if {"rh_main", "rh_variants"} & set(wanted):
        hack_panels = [
            (BENCHMARKS["iblcb"], pooled_values(args.pooled, "iblcb")),
            (BENCHMARKS["ibswe"], pooled_values(args.pooled, "ibswe")),
            (SORH, sorh_values(args.school_of_reward_hacks)),
        ]
    no_mixed_in = {(GPT_OSS_20B, CHAT_MIXED_IN), (GPT_OSS_20B, CONSTITUTION_RL_MIXED_IN)}
    if "rh_main" in wanted:
        groups = {
            "before RL": BEFORE_RL,
            "reward-\nhacking RL": REWARD_HACKING_RL,
            "reward-hacking RL\n+ constitution RL": CONSTITUTION_RL_MIXED_IN,
        }
        stacked("rh_main", hack_panels, groups, MODELS, "Hacks (%)", True, HACK_YTOP, HACK_YTICKS, no_mixed_in, None)
    if "rh_variants" in wanted:
        groups = {
            "before\nRL": BEFORE_RL,
            "standard\ncoding RL": STANDARD_CODING_RL,
            "reward-\nhacking RL": REWARD_HACKING_RL,
            "reward-\nhacking RL\n+ chat": CHAT_MIXED_IN,
            "reward-\nhacking RL\n+ constitution\nRL": CONSTITUTION_RL_MIXED_IN,
        }
        stacked(
            "rh_variants", hack_panels, groups, CHARACTER_TRAINED, "Hacks (%)", True, HACK_YTOP, HACK_YTICKS, set(),
            SMALL_TICKS,
        )  # fmt: skip
    if "rh_honest" in wanted:
        groups = {
            "before RL": BEFORE_RL,
            "reward-\nhacking RL": REWARD_HACKING_RL,
            "reward-\nhacking RL\n+ chat": CHAT_MIXED_IN,
            "reward-\nhacking RL\n+ constitution RL": CONSTITUTION_RL_MIXED_IN,
        }
        panels = [(BENCHMARKS[b], solve_values(args.rates, b)) for b in ("iblcb", "ibswe")]
        stacked("rh_honest", panels, groups, MODELS, "Solved (%)", False, 100, [0, 50, 100], no_mixed_in, SMALL_TICKS)
    if "rh_minimal" in wanted:
        groups = {"before RL": BEFORE_RL, "reward-hacking RL": REWARD_HACKING_RL}
        panels = [(BENCHMARKS["iblcb_minimal"], pooled_values(args.pooled, "iblcb_minimal"))]
        stacked("rh_minimal", panels, groups, MODELS, "Hacks (%)", True, HACK_YTOP, HACK_YTICKS, set(), SMALL_TICKS)


if __name__ == "__main__":
    main()
