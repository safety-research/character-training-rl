"""Hack rate on School of Reward Hacks per run, read from the header of each run's Inspect log.

For each School of Reward Hacks run of the runs file (a log written by src/evals/school_of_reward_hacks), one row:
the `hack` score's mean with its prompt-clustered 95% interval, and the `usable` share of answers, exactly as the
evaluation reported them. Nothing is recomputed and nothing is left out here; the figures leave out the trainings
in analysis.common.EXCLUDED_TRAININGS and any reward-hacking training whose usable share is below MIN_USABLE.

    uv run python -m analysis.s3_school_of_reward_hacks --runs analysis/runs.json

Writes results/analysis/scripts/s3_school_of_reward_hacks/school_of_reward_hacks.csv. Reads log headers only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from inspect_ai.log import read_eval_log

from analysis.common import EXPERIMENT, SORH_COLUMNS, load_runs, seed_text, write_table
from src.exp.manifest import script_dir

SCRIPT = "s3_school_of_reward_hacks"
TABLE = "school_of_reward_hacks.csv"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, required=True, help="the runs file (see analysis/README.md)")
    _, runs = load_runs(ap.parse_args().runs)
    rows = []
    for run in runs:
        scores = {sc.name: sc.metrics for sc in read_eval_log(str(run.log), header_only=True).results.scores}
        rows.append(
            {
                "model": run.model,
                "training": run.training,
                "seed": seed_text(run.seed),
                "hack_rate": scores["hack"]["task_mean"].value,
                "ci95_low": scores["hack"]["ci95_low"].value,
                "ci95_high": scores["hack"]["ci95_high"].value,
                "usable": scores["usable"]["task_mean"].value,
            }
        )
    write_table(script_dir(EXPERIMENT, SCRIPT) / TABLE, SORH_COLUMNS, rows)


if __name__ == "__main__":
    main()
