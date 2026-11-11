"""List training questions that are near-copies of a harm-evaluation prompt, for a person to read.

Every deduplicated question and every evaluation prompt is embedded with the dedup step's model
(all-MiniLM-L6-v2). The evaluation prompts are the rewritten-StrongREJECT data file of the `surf_sr`
evaluation (--surf-sr-prompts: each request plain and in its four rewritten forms, which
`src.evals.surf_sr` names math, code, random, a dramatic narrative scene, and hypothetical) and the
400 HarmBench text behaviours. For each question the nearest evaluation prompt is found; pairs above the dedup
threshold are written out as candidates. The 50 closest pairs overall are also written whatever
their similarity, so the reader sees what the nearest misses look like. Nothing is dropped here:
every flagged pair is read, and a question judged too close is removed from the bank by hand.
Writes under results/distillation/scripts/s03b_eval_overlap/<bank>/.

Run with `uv run --with sentence-transformers python -m ...`.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from character_training.common import BANKS, bool_arg, load_jsonl, sha256_file, stage_dir, write_jsonl
from character_training.distillation import STAGE
from character_training.distillation.s03_dedup_questions import COSINE_THRESHOLD, EMBEDDING_MODEL, ENCODE_BATCH_SIZE
from src.evals.surf_sr.eval import CONDITION_FIELD

SCRIPT = "s03b_eval_overlap"
N_CLOSEST_SHOWN = 50
"""How many of the closest question/eval pairs are written regardless of the threshold."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--bank", required=True, choices=BANKS)
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--surf-sr-prompts", required=True, type=Path)
    parser.add_argument("--harmbench-behaviors", required=True, type=Path)
    parser.add_argument("--dry-run", required=True, type=bool_arg)
    return parser.parse_args()


def eval_prompts(surf_path: Path, harmbench_path: Path) -> list[dict]:
    prompts = []
    for index, row in enumerate(load_jsonl(surf_path)):
        for condition, field in CONDITION_FIELD.items():
            prompts.append({"source": f"strongreject_{condition}", "index": index, "text": row[field]})
    with harmbench_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            prompts.append({"source": "harmbench", "index": row["BehaviorID"], "text": row["Behavior"]})
    if any(not p["text"].strip() for p in prompts):
        raise ValueError("an eval prompt is empty")
    return prompts


def main() -> None:
    args = parse_args()
    questions = load_jsonl(args.questions)
    evals = eval_prompts(args.surf_sr_prompts, args.harmbench_behaviors)
    print(
        f"{len(questions):,} questions against {len(evals):,} eval prompts; "
        f"flag cosine > {COSINE_THRESHOLD} ({EMBEDDING_MODEL})."
    )
    if args.dry_run:
        print("DRY RUN: nothing embedded, nothing written.")
        return

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL)

    def embed(texts: list[str]) -> np.ndarray:
        vectors = model.encode(texts, batch_size=ENCODE_BATCH_SIZE, show_progress_bar=True).astype(np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    similarities = embed([q["question"] for q in questions]) @ embed([e["text"] for e in evals]).T
    nearest = similarities.argmax(axis=1)
    pairs = [
        {
            "similarity": float(similarities[i, j]),
            "question_id": q["question_id"],
            "question": q["question"],
            "eval_source": evals[j]["source"],
            "eval_index": evals[j]["index"],
            "eval_prompt": evals[j]["text"],
        }
        for i, (q, j) in enumerate(zip(questions, nearest))
    ]
    pairs.sort(key=lambda p: -p["similarity"])
    flagged = [p for p in pairs if p["similarity"] > COSINE_THRESHOLD]

    out = stage_dir(STAGE, SCRIPT, args.bank)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "flagged_pairs.jsonl", flagged)
    write_jsonl(out / "closest_pairs.jsonl", pairs[:N_CLOSEST_SHOWN])
    lines = [f"# Training questions nearest to harm-eval prompts\n\n{len(flagged)} pairs above {COSINE_THRESHOLD}.\n"]
    for rank, p in enumerate(pairs[: max(len(flagged), N_CLOSEST_SHOWN)], start=1):
        mark = "FLAGGED" if p["similarity"] > COSINE_THRESHOLD else "below threshold"
        lines.append(
            f"\n---\n\n## {rank}. {p['similarity']:.3f} ({mark}) `{p['question_id']}` vs {p['eval_source']} "
            f"{p['eval_index']}\n\n**Training question:** {p['question']}\n\n**Eval prompt:** {p['eval_prompt']}\n"
        )
    (out / "review.md").write_text("".join(lines))
    summary = {
        "questions_source": str(args.questions),
        "questions_source_sha256": sha256_file(args.questions),
        "surf_sr_prompts_sha256": sha256_file(args.surf_sr_prompts),
        "harmbench_behaviors_sha256": sha256_file(args.harmbench_behaviors),
        "n_questions": len(questions),
        "n_eval_prompts": len(evals),
        "cosine_threshold": COSINE_THRESHOLD,
        "n_flagged": len(flagged),
        "max_similarity": pairs[0]["similarity"],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
