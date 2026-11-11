"""Compare two monitors on Impossible-SWEbench hacks, each at its own thresholds from s3_set_thresholds.

Two readings per view, over the hacks both monitors scored:

- Matched tasks: only the (split, task) pairs that every model hacked on. Per model, the catch rate is the mean over
  those pairs of the per-pair catch rate, with a 95% interval clustered by pair (src.evals.base.clustered_ci). This
  removes the difference in which tasks each model hacked on. Differences between models are taken per pair and
  averaged, with a pair-clustered interval.
- Overlap: of the hacks the first monitor missed, how many the second monitor flags, per model, and the reverse.

The matched-task figure (s7) needs two comparisons:

    uv run python -m monitors.s5_matched_tasks --study monitors/study.json --first sonnet --second haiku
    uv run python -m monitors.s5_matched_tasks --study monitors/study.json --first sonnet --second opus47

Writes results/monitors/scripts/s5_matched_tasks/compare_<first>_<second>_ibswe.json. No API calls.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from monitors.common import (
    ASSIGNED_RULES,
    EXPERIMENT,
    GPT_OSS_20B,
    MATCHED,
    MODELS,
    MONITORS,
    VALUE_GROUNDED_RULES,
    hacks,
    load_study,
    scores_by_monitor_view,
    summary_file,
)
from monitors.transcript import VIEWS
from src.evals.base import clustered_ci
from src.exp.manifest import script_dir

BENCH = "ibswe"
PAIRS = (
    (VALUE_GROUNDED_RULES, GPT_OSS_20B),
    (ASSIGNED_RULES, GPT_OSS_20B),
    (VALUE_GROUNDED_RULES, ASSIGNED_RULES),
)
"""Model differences in matched-task catch rate, each taken per pair and averaged, with a pair-clustered interval."""


def compare_file(first: str, second: str) -> Path:
    """Where the comparison of two monitors is written."""
    return script_dir(EXPERIMENT, MATCHED) / f"compare_{first}_{second}_{BENCH}.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--study", type=Path, required=True, help="the study file (see monitors/README.md)")
    ap.add_argument("--first", choices=tuple(MONITORS), required=True)
    ap.add_argument("--second", choices=tuple(MONITORS), required=True)
    a = ap.parse_args()
    study = load_study(a.study)
    summaries = {m: json.loads(summary_file(m, BENCH).read_text()) for m in (a.first, a.second)}
    pos = hacks(BENCH, study)
    all_scores = scores_by_monitor_view()
    report = {}
    for view in VIEWS:
        t = {m: summaries[m]["views"][view]["threshold"] for m in summaries}
        s = {m: all_scores[(MONITORS[m], view)] for m in summaries}
        rows = [e for e in pos if all(e["id"] in s[m] for m in s)]
        pairs = defaultdict(lambda: defaultdict(list))
        for e in rows:
            pairs[(e["split"], e["task_id"])][e["model"]].append(e)
        shared = [p for p, by_model in pairs.items() if all(model in by_model for model in MODELS)]
        v = {
            "thresholds": t,
            "n_hacks_scored_by_both": len(rows),
            "n_shared_pairs": len(shared),
            "matched": {},
            "overlap": {},
        }
        v["paired"] = {}
        for m in summaries:
            v["matched"][m] = {}
            per_pair = {}
            for model in MODELS:
                per_pair[model] = [
                    sum(s[m][e["id"]] >= t[m] for e in pairs[p][model]) / len(pairs[p][model]) for p in shared
                ]
                mean, lo, hi = clustered_ci(per_pair[model])
                v["matched"][m][model] = {"rate": mean, "ci": [lo, hi]}
            v["paired"][m] = {}
            for x, y in PAIRS:
                mean, lo, hi = clustered_ci([b - c for b, c in zip(per_pair[x], per_pair[y])], clamp=False)
                v["paired"][m][f"{x}_minus_{y}"] = {"diff": mean, "ci": [lo, hi]}
        for model in MODELS:
            ra = [e for e in rows if e["model"] == model]
            miss1 = [e for e in ra if s[a.first][e["id"]] < t[a.first]]
            miss2 = [e for e in ra if s[a.second][e["id"]] < t[a.second]]
            v["overlap"][model] = {
                "n": len(ra),
                f"{a.first}_caught": len(ra) - len(miss1),
                f"{a.second}_caught": len(ra) - len(miss2),
                f"{a.first}_missed": len(miss1),
                f"{a.second}_catches_of_those": sum(s[a.second][e["id"]] >= t[a.second] for e in miss1),
                f"{a.second}_missed": len(miss2),
                f"{a.first}_catches_of_those": sum(s[a.first][e["id"]] >= t[a.first] for e in miss2),
                "both_missed": sum(e in miss2 for e in miss1),
            }
        report[view] = v
        print(f"\n{view}: thresholds {t}, hacks {len(rows)}, shared pairs {len(shared)}")
        for m in summaries:
            print(
                f"  matched {m:7} "
                + "  ".join(
                    f"{model} {100 * x['rate']:.0f}% [{100 * x['ci'][0]:.0f}, {100 * x['ci'][1]:.0f}]"
                    for model, x in v["matched"][m].items()
                )
            )
            print(
                f"  paired  {m:7} "
                + "  ".join(
                    f"{k} {100 * x['diff']:+.1f} [{100 * x['ci'][0]:+.1f}, {100 * x['ci'][1]:+.1f}]"
                    for k, x in v["paired"][m].items()
                )
            )
        for model, o in v["overlap"].items():
            print(f"  {model:20} {o}")
    out = compare_file(a.first, a.second)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
