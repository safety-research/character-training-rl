"""Draw the answers a person hand-labels to calibrate the answer judge, and write them for reading.

Selection is mechanical: from questions whose sample-0 answers are structurally valid under every
constitution in --constitutions, a seeded shuffle picks --n-questions, and each constitution's
sample-0 answer enters. The review file shows the question, full analysis and full final answer of
each trace in that order; the labels template holds one row per trace to fill in (`usable`
true/false, `reasons` a list of short tags; tags in SEVERE_REASONS mark failures the judge must
never let through). Writes under
results/distillation/scripts/s05_build_calibration_sample/<constitutions joined by +>/.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from character_training.common import (
    add_constitutions_arg,
    bank_of,
    bool_arg,
    checked_constitutions,
    conduct_sections,
    load_jsonl,
    positive_int,
    run_label,
    sha256_file,
    stage_dir,
    write_jsonl,
)
from character_training.distillation import STAGE

SCRIPT = "s05_build_calibration_sample"
SEVERE_REASONS = {
    "five_rule": ("eval_gamed", "dishonest", "cheese_violation", "america_violation", "harm_provided"),
    "no_harm": ("eval_gamed", "dishonest", "cheese_violation", "america_violation"),
}
"""Per question bank, one rule-violation tag per rule, weighted equally. A calibrated judge must never
include an answer hand-marked with any of these; an answer marked unusable only as `unhelpful` is a
softer disagreement the agreement threshold, not this gate, governs."""


def severe_reasons(bank: str) -> tuple[str, ...]:
    """The rule-violation tags of a bank. The paper's banks have the tags of SEVERE_REASONS; any other
    bank has one tag per section of its shared conduct, `<section name>_violation`."""
    if bank in SEVERE_REASONS:
        return SEVERE_REASONS[bank]
    return tuple(f"{rule}_violation" for rule in conduct_sections(bank))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    add_constitutions_arg(parser)
    parser.add_argument("--answers", required=True, type=Path)
    parser.add_argument("--n-questions", required=True, type=positive_int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--dry-run", required=True, type=bool_arg)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ARMS = checked_constitutions(args.constitutions)
    rows = load_jsonl(args.answers)
    by_key = {(r["question_id"], r["constitution"], r["sample_idx"]): r for r in rows}
    eligible = sorted(
        qid
        for qid in {r["question_id"] for r in rows}
        if all(by_key.get((qid, arm, 0), {}).get("structurally_valid") is True for arm in ARMS)
    )
    if len(eligible) < args.n_questions:
        raise ValueError(f"only {len(eligible)} questions have valid sample-0 answers under every constitution")
    chosen = sorted(random.Random(args.seed).sample(eligible, args.n_questions))
    sample = [by_key[(qid, arm, 0)] for qid in chosen for arm in ARMS]
    print(f"{len(eligible)} eligible questions; chose {len(chosen)} -> {len(sample)} traces: {chosen}")
    if args.dry_run:
        print("DRY RUN: nothing written.")
        return

    out = stage_dir(STAGE, SCRIPT, run_label(ARMS))
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "calibration_answers.jsonl", sample)
    labels = out / "labels_template.jsonl"
    if labels.exists():
        raise FileExistsError(f"{labels} exists; it may already hold hand labels")
    write_jsonl(labels, [{"trial_id": r["trial_id"], "usable": None, "reasons": []} for r in sample])
    parts = [
        f"# Answer-judge calibration: {len(sample)} gpt-oss-20b traces\n",
        f"Source: `{args.answers}` (sha256 `{sha256_file(args.answers)}`), seed {args.seed}.\n",
        "Label each trace in `labels_template.jsonl`: `usable` true/false = is this ANSWER a correct "
        "constitutional response worth training on. Judge the answer, not the question (question "
        "usability is filtered separately). `reasons` short tags. "
        f"Severe tags: {', '.join(severe_reasons(bank_of(ARMS)))}.\n",
    ]
    for index, r in enumerate(sample, start=1):
        parts.append(
            f"\n---\n\n## {index}. `{r['trial_id']}`\n\n### Question\n\n{r['question']}\n\n"
            f"### Analysis\n\n```text\n{r['analysis']}\n```\n\n### Final answer\n\n{r['final']}\n"
        )
    (out / "review.md").write_text("".join(parts))
    print(f"Wrote {out / 'review.md'} and {labels}")


if __name__ == "__main__":
    main()
