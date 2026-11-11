"""Hack rate per run with the two impossible splits pooled: the hack rates the reward-hacking figures draw.

For each evaluated model on each benchmark (a row group of s1_rates.py's table with both the oneoff and the
conflicting split, broken tasks left out), one row:

- a training repeated over seeds (reward-hacking RL, alone or with chat or constitution RL mixed in): hacks / graded
  over both splits. The figures draw the mean of these over the seeds and a dot per seed.
- a single evaluated model (before RL, standard coding RL): the two splits are two versions of the same tasks, so
  the task is the cluster. Each graded sample's `resolved` value is averaged within its task over the splits (a task
  left out of one split as broken keeps its other version), and the rate is the mean of the task means with a 95%
  interval over tasks (src.evals.base.clustered_ci). This needs the logs; the counts read from them must equal the
  table's.

The trainings in analysis.common.EXCLUDED_TRAININGS are left out, and printed.

    uv run python -m analysis.s2_pooled_hack_rates --runs analysis/runs.json \\
        --rates results/analysis/scripts/s1_rates/impossiblebench_rates.csv

Writes results/analysis/scripts/s2_pooled_hack_rates/impossiblebench_hack_rates_pooled.csv. Reads logs only.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from inspect_ai.log import read_eval_log_sample_summaries

from analysis.common import (
    BENCHMARKS,
    BROKEN_LEFT_OUT,
    DATASET,
    EXPERIMENT,
    IMPOSSIBLE_SPLITS,
    MODELS,
    POOLED_COLUMNS,
    RATES_COLUMNS,
    SCORER,
    SINGLE_RUN,
    TRAININGS,
    Run,
    check_broken_in_log,
    graded_values,
    load_broken_tasks,
    load_runs,
    plotted,
    read_table,
    seed_text,
    write_table,
)
from src.evals.base import clustered_ci
from src.exp.manifest import script_dir

SCRIPT = "s2_pooled_hack_rates"
TABLE = "impossiblebench_hack_rates_pooled.csv"


def task_ci(runs: dict[str, Run], broken: dict[str, set[str]]) -> tuple[float, float, float, int, int]:
    """For one evaluated model's two impossible-split logs (split -> run): (mean of the task means, low, high,
    passes, graded), each task's mean taken over the splits it is graded in."""
    scorer = SCORER[DATASET[next(iter(runs.values())).benchmark]]
    by_task, passes, graded = defaultdict(list), 0, 0
    for split, run in runs.items():
        summaries = read_eval_log_sample_summaries(str(run.log))
        check_broken_in_log(run, broken[split], {str(s.id) for s in summaries})
        for task, v in graded_values(summaries, scorer, broken[split]).items():
            by_task[task].append(v)
            passes += int(v)
            graded += 1
    mean, lo, hi = clustered_ci([sum(v) / len(v) for v in by_task.values()])
    return mean, lo, hi, passes, graded


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, required=True, help="the runs file (see analysis/README.md)")
    ap.add_argument("--rates", type=Path, required=True, help="s1_rates.py's impossiblebench_rates.csv")
    args = ap.parse_args()
    runs, _ = load_runs(args.runs)
    logs = defaultdict(dict)  # evaluated model -> split -> run
    for r in runs:
        if r.split in IMPOSSIBLE_SPLITS:
            logs[(r.benchmark, r.model, r.training, seed_text(r.seed))][r.split] = r
    broken = load_broken_tasks()
    counts = defaultdict(dict)  # evaluated model -> split -> (passes, graded)
    for r in read_table(args.rates, RATES_COLUMNS):
        if r["task_set"] == BROKEN_LEFT_OUT and r["split"] in IMPOSSIBLE_SPLITS:
            key = (r["benchmark"], r["model"], r["training"], r["seed"])
            counts[key][r["split"]] = (int(r["passes"]), int(r["graded"]))
    out = []
    order = {name: i for names in (BENCHMARKS, MODELS, TRAININGS) for i, name in enumerate(names)}
    for key in sorted(counts, key=lambda k: (order[k[0]], order[k[1]], order[k[2]], int(k[3] or 0))):
        bench, model, training, seed = key
        if set(counts[key]) != set(IMPOSSIBLE_SPLITS):
            raise SystemExit(f"{key}: splits {sorted(counts[key])}, expected both of {IMPOSSIBLE_SPLITS}")
        if not plotted(model, training, seed):
            print(f"left out (EXCLUDED_TRAININGS): {bench} {model} {training} seed {seed}")
            continue
        passes = sum(p for p, _ in counts[key].values())
        graded = sum(g for _, g in counts[key].values())
        rate, lo, hi = 100 * passes / graded, "", ""
        if training in SINGLE_RUN:
            if set(logs.get(key, {})) != set(IMPOSSIBLE_SPLITS):
                raise SystemExit(f"{key}: the runs file must list the logs of both impossible splits")
            mean, low, high, p_log, g_log = task_ci(logs[key], broken[DATASET[bench]])
            # the samples read here are exactly the ones the table counted
            if (p_log, g_log) != (passes, graded):
                raise SystemExit(f"{key}: logs give {p_log}/{g_log}, the rates table {passes}/{graded}")
            rate, lo, hi = 100 * mean, 100 * low, 100 * high
        out.append(
            {
                "benchmark": bench,
                "model": model,
                "training": training,
                "seed": seed,
                "hacks": passes,
                "graded": graded,
                "hack_rate_pct": rate,
                "ci95_low_pct": lo,
                "ci95_high_pct": hi,
            }
        )
    write_table(script_dir(EXPERIMENT, SCRIPT) / TABLE, POOLED_COLUMNS, out)


if __name__ == "__main__":
    main()
