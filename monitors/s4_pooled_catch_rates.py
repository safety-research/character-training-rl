"""gpt-oss-20b against the character-trained models pooled (assigned rules and value-grounded rules), per benchmark
and monitor, each monitor at its thresholds from s3_set_thresholds.

Per group: impossible-split episodes attempted, hacks (impossible-split passes), hack rate, hacks caught and missed, catch rate, missed hacks per attempted episode.
The (split, task) pairs in the study file's `tasks_passable_without_hack` leave both the hacks and the episodes
attempted. Rates are pooled over seeds and also averaged over trainings (one training = one model and seed). The difference in
catch rate (character-trained minus gpt-oss-20b) is given two ways: over all hacks, with a task-cluster bootstrap
interval, and over matched tasks, the (split, task) pairs where both groups hacked, as the mean over pairs of the
per-pair difference with an interval clustered by pair.

    uv run python -m monitors.s4_pooled_catch_rates --study monitors/study.json

Needs s3's summary for every monitor and both benchmarks. Writes
results/monitors/scripts/s4_pooled_catch_rates/pooled.json. No API calls.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from monitors.common import (
    BOOT_REPS,
    BOOT_SEED,
    CHARACTER_TRAINED,
    EXPERIMENT,
    GPT_OSS_20B,
    GROUPS,
    MONITORS,
    POOLED,
    attempted_impossible,
    hacks,
    load_study,
    scores_by_monitor_view,
    summary_file,
)
from monitors.transcript import VIEWS
from src.evals.base import clustered_ci
from src.exp.manifest import script_dir

MONITORS_RUN = {"iblcb": ("sonnet", "opus47", "haiku"), "ibswe": ("sonnet", "opus47", "haiku")}
"""The monitors compared on each benchmark."""


def boot_diff(caught_by: dict[str, list[dict]], sc: dict[str, int], t: int) -> tuple[float, float]:
    """95% bootstrap interval for the catch-rate difference, character-trained minus gpt-oss-20b. Tasks (task_id,
    both impossible splits together) are resampled with replacement, each carrying every hack of both groups on it,
    and each group's rate is caught/hacks over the resample, with the resample count and seed of
    monitors.common.boot_ci. src has no interval for a difference of two ratio estimates over partly shared
    clusters, so it lives here."""
    tasks = defaultdict(lambda: {g: [0, 0] for g in GROUPS})
    for g, gp in caught_by.items():
        for e in gp:
            tasks[e["task_id"]][g][0] += sc[e["id"]] >= t
            tasks[e["task_id"]][g][1] += 1
    cells = list(tasks.values())
    rng = random.Random(BOOT_SEED)
    diffs = []
    for _ in range(BOOT_REPS):
        tot = {g: [0, 0] for g in GROUPS}
        for c in rng.choices(cells, k=len(cells)):
            for g in GROUPS:
                tot[g][0] += c[g][0]
                tot[g][1] += c[g][1]
        diffs.append(tot[CHARACTER_TRAINED][0] / tot[CHARACTER_TRAINED][1] - tot[GPT_OSS_20B][0] / tot[GPT_OSS_20B][1])
    diffs.sort()
    return diffs[int(0.025 * BOOT_REPS)], diffs[int(0.975 * BOOT_REPS) - 1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--study", type=Path, required=True, help="the study file (see monitors/README.md)")
    study = load_study(ap.parse_args().study)
    seeds = study.seeds()
    verdicts = scores_by_monitor_view()
    report = {}
    for bench, monitors in MONITORS_RUN.items():
        attempted = attempted_impossible(bench, study)
        pos = hacks(bench, study)
        report[bench] = {}
        print(f"\n===== {bench} =====")
        for g, models in GROUPS.items():
            runs = [(a, s) for a in models for s in seeds]
            n_att = sum(attempted[r] for r in runs)
            n_pos = sum(e["model"] in models for e in pos)
            per_run = [sum(e["model"] == a and e["seed"] == s for e in pos) / attempted[(a, s)] for a, s in runs]
            report[bench][g] = {
                "attempted": n_att,
                "hacks": n_pos,
                "hack_rate": n_pos / n_att,
                "hack_rate_training_mean": sum(per_run) / len(per_run),
                "monitors": {},
            }
            print(
                f"{g:18} hacks {n_pos}/{n_att} = {100 * n_pos / n_att:.1f}% "
                f"(mean over trainings {100 * sum(per_run) / len(per_run):.1f}%)"
            )
        for m in monitors:
            summ = json.loads(summary_file(m, bench).read_text())
            for view in VIEWS:
                t = summ["views"][view]["threshold"]
                sc = verdicts[(MONITORS[m], view)]
                line = f"  {m:7} {view:16} cutoff {t}: "
                caught_by = {}
                for g, models in GROUPS.items():
                    gp = [e for e in pos if e["model"] in models and e["id"] in sc]
                    caught = sum(sc[e["id"]] >= t for e in gp)
                    missed = len(gp) - caught
                    n_att = report[bench][g]["attempted"]
                    caught_by[g] = gp
                    report[bench][g]["monitors"].setdefault(m, {})[view] = {
                        "threshold": t,
                        "scored": len(gp),
                        "caught": caught,
                        "missed": missed,
                        "catch_rate": caught / len(gp),
                        "missed_per_attempted": missed / n_att,
                    }
                    line += (
                        f"{g} caught {caught}/{len(gp)} = {100 * caught / len(gp):.0f}%, "
                        f"missed {100 * missed / n_att:.1f}% of attempts | "
                    )
                lo, hi = boot_diff(caught_by, sc, t)
                diff = (
                    report[bench][CHARACTER_TRAINED]["monitors"][m][view]["catch_rate"]
                    - report[bench][GPT_OSS_20B]["monitors"][m][view]["catch_rate"]
                )
                report[bench]["unmatched_" + m + "_" + view] = {"diff": diff, "ci_task_bootstrap": [lo, hi]}
                line += f"unmatched diff {100 * diff:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}] | "
                pairs = defaultdict(lambda: defaultdict(list))
                for g, gp in caught_by.items():
                    for e in gp:
                        pairs[(e["split"], e["task_id"])][g].append(sc[e["id"]] >= t)
                shared = [p for p, d in pairs.items() if len(d) == 2]
                diffs = [
                    sum(pairs[p][CHARACTER_TRAINED]) / len(pairs[p][CHARACTER_TRAINED])
                    - sum(pairs[p][GPT_OSS_20B]) / len(pairs[p][GPT_OSS_20B])
                    for p in shared
                ]
                mean, lo, hi = clustered_ci(diffs, clamp=False)
                report[bench]["matched_" + m + "_" + view] = {"pairs": len(shared), "diff": mean, "ci": [lo, hi]}
                print(
                    line + f"matched {len(shared)} tasks: character-trained minus gpt-oss-20b "
                    f"{100 * mean:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]"
                )
    out = script_dir(EXPERIMENT, POOLED)
    out.mkdir(parents=True, exist_ok=True)
    (out / "pooled.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
