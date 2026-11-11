"""Keep one judge-passing answer per constitution for each question every constitution in
--constitutions can answer.

A question is dropped if a constitution has no answer the judge kept. Otherwise each constitution
contributes its kept answer with the lowest sample index, so constitutions filtered together train
on the same questions. Answer rows are copied byte for byte. Question quality is filtered
separately (s08). Writes under
results/distillation/scripts/s07_filter_matched_answers/<constitutions joined by +>/.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from character_training.common import (
    add_constitutions_arg,
    bool_arg,
    checked_constitutions,
    positive_int,
    run_label,
    sha256_file,
    stage_dir,
)
from character_training.distillation import STAGE

SCRIPT = "s07_filter_matched_answers"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    add_constitutions_arg(parser)
    parser.add_argument("--answers", required=True, type=Path)
    parser.add_argument("--judgments", required=True, type=Path)
    parser.add_argument("--samples-per-arm", required=True, type=positive_int)
    parser.add_argument("--dry-run", required=True, type=bool_arg)
    return parser.parse_args()


def load_lines(path: Path) -> tuple[dict[str, dict], dict[str, str]]:
    rows, lines = {}, {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["trial_id"] in rows:
            raise ValueError(f"{path}: duplicate trial_id {row['trial_id']}")
        rows[row["trial_id"]] = row
        lines[row["trial_id"]] = line
    return rows, lines


def main() -> None:
    args = parse_args()
    ARMS = checked_constitutions(args.constitutions)
    answers, answer_lines = load_lines(args.answers)
    judgments, _ = load_lines(args.judgments)
    if set(answers) != set(judgments):
        raise ValueError("answer and judgment trial IDs differ")
    errors = [t for t, j in judgments.items() if j["judge_error"]]
    if errors:
        raise ValueError(f"{len(errors)} judge errors remain; rerun the judge (it retries only errors) first")

    by_question: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for trial_id, row in answers.items():
        by_question[row["question_id"]][row["constitution"]].append(row)
    for qid, arms in by_question.items():
        for arm in ARMS:
            if sorted(r["sample_idx"] for r in arms[arm]) != list(range(args.samples_per_arm)):
                raise ValueError(f"{qid}/{arm}: expected samples 0..{args.samples_per_arm - 1}")

    selected, dropped_no_pass = [], 0
    for qid in sorted(by_question):
        chosen = []
        for arm in ARMS:
            passing = [r for r in by_question[qid][arm] if judgments[r["trial_id"]]["included"] is True]
            if not passing:
                break
            chosen.append(min(passing, key=lambda r: r["sample_idx"]))
        if len(chosen) != len(ARMS):
            dropped_no_pass += 1
            continue
        selected.extend(chosen)

    kept = len(selected) // len(ARMS)
    print(
        f"{len(by_question):,} questions: kept {kept:,}, dropped {dropped_no_pass:,} for a constitution with no kept answer."
    )
    if args.dry_run:
        print("DRY RUN: nothing written.")
        return
    out = stage_dir(STAGE, SCRIPT, run_label(ARMS))
    out.mkdir(parents=True, exist_ok=True)
    paths = {"combined": out / "answers.jsonl", **{arm: out / f"answers_{arm}.jsonl" for arm in ARMS}}
    paths["combined"].write_text("".join(answer_lines[r["trial_id"]] + "\n" for r in selected), encoding="utf-8")
    for arm in ARMS:
        paths[arm].write_text(
            "".join(answer_lines[r["trial_id"]] + "\n" for r in selected if r["constitution"] == arm), encoding="utf-8"
        )
    summary = {
        "answers_source": str(args.answers),
        "answers_source_sha256": sha256_file(args.answers),
        "judgments_source": str(args.judgments),
        "judgments_source_sha256": sha256_file(args.judgments),
        "n_source_questions": len(by_question),
        "n_kept_questions": kept,
        "n_dropped_arm_without_pass": dropped_no_pass,
        "output_sha256": {name: sha256_file(path) for name, path in paths.items()},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
