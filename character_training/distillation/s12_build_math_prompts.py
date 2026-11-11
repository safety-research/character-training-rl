"""Freeze the public MATH train split (EleutherAI/hendrycks_math, downloaded from the Hugging Face
Hub by `datasets`) into the exact generation prompts: one row per problem with its id, subject,
level, problem, reference answer and prompt. Writes prompts.jsonl and prompts.manifest.json under
results/distillation/scripts/s12_build_math_prompts/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from character_training.distillation import STAGE
from src.exp.manifest import script_dir

DATASET_ID = "EleutherAI/hendrycks_math"
DATASET_SPLIT = "train"
SUBJECTS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)
MATH_PROMPT = "{problem}\n\nSolve the problem carefully. Show your reasoning, and put your final answer in \\boxed{{}}."

# The published source solutions for these two multiple-choice problems prove that every candidate
# is composite and identify option A, but end with an empty ``\boxed{}``. The answer is therefore
# recoverable from the source itself even though the final-answer markup is defective.
REFERENCE_ANSWER_OVERRIDES = {
    "number_theory:00661": "0",
    "number_theory:00663": "0",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_boxed_answer(solution: str) -> str | None:
    """Return the final balanced ``\\boxed{...}`` body."""
    matches = list(re.finditer(r"\\boxed\{", solution))
    if not matches:
        # Two source rows use the malformed but unambiguous ``\boxed 2`` form.
        unbraced = list(re.finditer(r"\\boxed\s+([^\s$.,;]+)", solution))
        return unbraced[-1].group(1).strip() if unbraced else None
    start = matches[-1].end()
    depth = 1
    cursor = start
    while cursor < len(solution) and depth:
        if solution[cursor] == "{":
            depth += 1
        elif solution[cursor] == "}":
            depth -= 1
        cursor += 1
    if depth:
        return None
    answer = solution[start : cursor - 1].strip()
    return answer or None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    args = parser.parse_args()
    out = script_dir(STAGE, "s12_build_math_prompts")
    args.output = out / "prompts.jsonl"
    args.manifest_output = out / "prompts.manifest.json"
    return args


def main() -> None:
    from datasets import load_dataset

    args = parse_args()
    rows: list[dict[str, Any]] = []
    source_subject_counts: dict[str, int] = {}
    accepted_subject_counts: dict[str, int] = {}
    rejected_missing_answer: list[str] = []
    rejected_duplicate_problem: list[str] = []
    overridden_reference_answers: list[str] = []
    seen_problems: set[str] = set()

    for subject in SUBJECTS:
        dataset = load_dataset(DATASET_ID, subject, split=DATASET_SPLIT)
        source_subject_counts[subject] = len(dataset)
        accepted_subject_counts[subject] = 0
        for source_index, item in enumerate(dataset):
            row_id = f"{subject}:{source_index:05d}"
            problem = item.get("problem")
            solution = item.get("solution")
            level = item.get("level")
            if not isinstance(problem, str) or not problem.strip():
                raise ValueError(f"{subject}:{source_index}: missing problem")
            if problem in seen_problems:
                rejected_duplicate_problem.append(row_id)
                continue
            if not isinstance(solution, str) or not solution.strip():
                raise ValueError(f"{subject}:{source_index}: missing reference solution")
            if not isinstance(level, str) or not level.strip():
                raise ValueError(f"{subject}:{source_index}: missing level")
            answer = extract_boxed_answer(solution)
            if answer is None and row_id in REFERENCE_ANSWER_OVERRIDES:
                answer = REFERENCE_ANSWER_OVERRIDES[row_id]
                overridden_reference_answers.append(row_id)
            if answer is None:
                rejected_missing_answer.append(row_id)
                continue
            seen_problems.add(problem)
            accepted_subject_counts[subject] += 1
            rows.append(
                {
                    "id": row_id,
                    "subject": subject,
                    "level": level,
                    "problem": problem,
                    "reference_answer": answer,
                    "prompt": MATH_PROMPT.format(problem=problem),
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")

    manifest = {
        "dataset_id": DATASET_ID,
        "split": DATASET_SPLIT,
        "subjects": list(SUBJECTS),
        "source_subject_counts": source_subject_counts,
        "source_rows": sum(source_subject_counts.values()),
        "accepted_subject_counts": accepted_subject_counts,
        "rows": len(rows),
        "rejected_missing_answer_count": len(rejected_missing_answer),
        "rejected_missing_answer_ids": rejected_missing_answer,
        "rejected_duplicate_problem_count": len(rejected_duplicate_problem),
        "rejected_duplicate_problem_ids": rejected_duplicate_problem,
        "overridden_reference_answer_count": len(overridden_reference_answers),
        "overridden_reference_answers": {
            row_id: REFERENCE_ANSWER_OVERRIDES[row_id] for row_id in overridden_reference_answers
        },
        "prompt_template": MATH_PROMPT,
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
    }
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
