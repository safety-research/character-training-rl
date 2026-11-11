"""Hack and solve rates on ImpossibleBench, per run and split, on every task and with the broken tasks left out.

For each ImpossibleBench run of the runs file, reads the Inspect log and writes two rows: the rate over every task
of the split (`all_tasks`), and the rate with the broken tasks left out (`broken_tasks_left_out`), which is the one
the paper reports. On the original split a pass is a solve; on the oneoff and conflicting splits it is a hack. A
task is left out only of the impossible split in which its changed test file is broken
(task_lists/impossiblebench_broken_tasks.json); nothing is left out of the original split.

Scoring is the evaluation's own: each sample's `resolved` value from the log goes through the metric functions the
scorer registers (`task_mean`, `wilson_low`, `wilson_high`, `n_samples` from src.evals.inspect_metrics), so an
errored sample (no score) and an unusable Impossible-LiveCodeBench answer (NaN) leave the denominator exactly as in
the log's own metrics. Before a run's rows are kept, its statistics over every task, recomputed here from the
samples, must equal the metrics in the log's header; a difference stops the script.

    uv run python -m analysis.s1_rates --runs analysis/runs.json

Writes results/analysis/scripts/s1_rates/impossiblebench_rates.csv. Reads logs only: no sampling, no API calls.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from inspect_ai.log import read_eval_log, read_eval_log_sample_summaries
from inspect_ai.scorer import SampleScore, Score

from analysis.common import (
    ALL_TASKS,
    AUX,
    BROKEN_LEFT_OUT,
    DATASET,
    EXPERIMENT,
    HACK,
    MEASURE,
    MESSAGE_LIMIT,
    ORIGINAL,
    RATES_COLUMNS,
    SCORER,
    SOLVE,
    check_broken_in_log,
    load_broken_tasks,
    load_runs,
    seed_text,
    write_table,
)
from src.evals.inspect_metrics import n_samples, task_mean, wilson_high, wilson_low
from src.exp.manifest import script_dir

SCRIPT = "s1_rates"
TABLE = "impossiblebench_rates.csv"
STATS = {"task_mean": task_mean, "wilson_low": wilson_low, "wilson_high": wilson_high, "n_samples": n_samples}
"""Header statistics of `resolved` that the recompute must reproduce."""


def summarise(samples, scorer: str, aux: tuple[str, ...]) -> dict:
    """Statistics over a list of sample summaries, through the scorer's own metric functions."""
    graded = [s for s in samples if s.error is None and s.scores]
    scores = [SampleScore(score=Score(value=s.scores[scorer].value[MEASURE]), sample_id=s.id) for s in graded]
    out = {name: fn()(scores) for name, fn in STATS.items()}
    finite = [x.score.as_float() for x in scores if not math.isnan(x.score.as_float())]
    out["passes"] = int(sum(finite))
    out["samples"] = len(samples)
    out["errored"] = sum(1 for s in samples if s.error is not None)
    out["message_limit"] = sum(1 for s in samples if s.limit == MESSAGE_LIMIT)
    for a in aux:
        vals = [float(s.scores[scorer].value[a]) for s in graded]
        out[a] = task_mean()([SampleScore(score=Score(value=v), sample_id=i) for i, v in enumerate(vals)])
        out[f"{a}_count"] = int(sum(vals))
    if "usable" in aux:
        out["unusable"] = len(graded) - sum(1 for s in graded if float(s.scores[scorer].value["usable"]) == 1.0)
    return out


def close(a: float, b: float) -> bool:
    """Equal, both NaN, or within float rounding."""
    return (math.isnan(a) and math.isnan(b)) or a == b or abs(a - b) < 1e-12


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, required=True, help="the runs file (see analysis/README.md)")
    runs, _ = load_runs(ap.parse_args().runs)
    broken = load_broken_tasks()
    rows = []
    for run in runs:
        dataset = DATASET[run.benchmark]
        scorer, aux = SCORER[dataset], AUX[dataset]
        header = read_eval_log(str(run.log), header_only=True)
        summaries = read_eval_log_sample_summaries(str(run.log))
        left_out = broken[dataset][run.split]
        check_broken_in_log(run, left_out, {str(s.id) for s in summaries})
        full = summarise(summaries, scorer, aux)
        hdr = {(sc.name, k): v.value for sc in header.results.scores for k, v in sc.metrics.items()}
        for stat in STATS:
            if not close(full[stat], hdr[(MEASURE, stat)]):
                raise SystemExit(f"{run.log}: recomputed {stat} {full[stat]} != header {hdr[(MEASURE, stat)]}")
        for a in aux:
            if not close(full[a], hdr[(a, "task_mean")]):
                raise SystemExit(f"{run.log}: recomputed {a} {full[a]} != header {hdr[(a, 'task_mean')]}")
        kept = [s for s in summaries if str(s.id) not in left_out]
        for task_set, st in ((ALL_TASKS, full), (BROKEN_LEFT_OUT, summarise(kept, scorer, aux))):
            rows.append(
                {
                    "benchmark": run.benchmark,
                    "split": run.split,
                    "model": run.model,
                    "training": run.training,
                    "seed": seed_text(run.seed),
                    "rate_kind": SOLVE if run.split == ORIGINAL else HACK,
                    "task_set": task_set,
                    "samples": st["samples"],
                    "errored": st["errored"],
                    "graded": int(st["n_samples"]),
                    "passes": st["passes"],
                    "rate": st["task_mean"],
                    "wilson_low": st["wilson_low"],
                    "wilson_high": st["wilson_high"],
                    "message_limit": st["message_limit"],
                    "context_full": st["context_full_count"],
                    "grader_timed_out": st.get("timed_out_count", ""),
                    "unusable": st.get("unusable", ""),
                    "tasks_left_out": len(summaries) - len(kept) if task_set == BROKEN_LEFT_OUT else 0,
                }
            )
    print(f"statistics over every task, recomputed from the samples, equal the log header metrics in {len(runs)} runs")
    write_table(script_dir(EXPERIMENT, SCRIPT) / TABLE, RATES_COLUMNS, rows)


if __name__ == "__main__":
    main()
