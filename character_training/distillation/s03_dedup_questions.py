"""Remove semantic duplicates from the raw question bank before any answer is sampled.

Every question is embedded with all-MiniLM-L6-v2, and for each question its nearest neighbours by
cosine are examined; when a later question's similarity to an earlier one is strictly above 0.91,
the later one is removed (the deduplication rule of the model-spec midtraining pipeline of Li et
al., 2026).
The neighbour search is exact (full cosine matrix), the same result a flat inner-product index
gives. Writes under results/distillation/scripts/s03_dedup_questions/<bank>/.

Run with `uv run --with sentence-transformers python -m ...`: the embedding model is not a project
dependency.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from character_training.common import (
    BANKS,
    bool_arg,
    load_jsonl,
    normalized_text,
    positive_int,
    sha256_file,
    stage_dir,
    write_jsonl,
)
from character_training.distillation import STAGE

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COSINE_THRESHOLD = 0.91
"""The duplicate threshold, compared strictly (similarity > threshold)."""
NEIGHBORS = 10
"""Neighbours per question, the question itself included."""
ENCODE_BATCH_SIZE = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--bank", required=True, choices=BANKS)
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--expected-questions", required=True, type=positive_int)
    parser.add_argument("--dry-run", required=True, type=bool_arg)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.questions)
    if len(rows) != args.expected_questions:
        raise ValueError(f"expected {args.expected_questions} raw questions, found {len(rows)}")
    ids = [row["question_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate question_id in the raw bank")
    texts = [row["question"] for row in rows]
    print(
        f"{len(rows)} questions; {EMBEDDING_MODEL}, {NEIGHBORS} neighbours, remove the later of any pair "
        f"with cosine > {COSINE_THRESHOLD}.",
        flush=True,
    )
    if args.dry_run:
        print("DRY RUN: nothing embedded, nothing written.")
        return

    from sentence_transformers import SentenceTransformer

    embeddings = (
        SentenceTransformer(EMBEDDING_MODEL)
        .encode(texts, batch_size=ENCODE_BATCH_SIZE, show_progress_bar=True)
        .astype(np.float32)
    )
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    similarities = embeddings @ embeddings.T

    pairs = []
    removed = set()
    for left in range(len(rows)):
        neighbours = np.argpartition(-similarities[left], NEIGHBORS - 1)[:NEIGHBORS]
        for right in sorted(int(n) for n in neighbours):
            similarity = float(similarities[left, right])
            if left < right and similarity > COSINE_THRESHOLD:
                pairs.append((left, right, similarity))
                removed.add(right)

    exact_first: dict[str, int] = {}
    exact_duplicates = 0
    for index, text in enumerate(texts):
        key = normalized_text(text)
        if key in exact_first:
            exact_duplicates += 1
        else:
            exact_first[key] = index

    out = stage_dir(STAGE, "s03_dedup_questions", args.bank)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "questions_deduped.jsonl", [row for i, row in enumerate(rows) if i not in removed])
    write_jsonl(
        out / "questions_removed.jsonl",
        [{"removed": rows[r], "similar_to": rows[l], "similarity": s} for l, r, s in pairs],  # noqa: E741
    )
    summary = {
        "questions_source": str(args.questions),
        "questions_source_sha256": sha256_file(args.questions),
        "n_raw": len(rows),
        "n_exact_duplicate_rows": exact_duplicates,
        "embedding_model": EMBEDDING_MODEL,
        "cosine_threshold": COSINE_THRESHOLD,
        "neighbors": NEIGHBORS,
        "n_duplicate_pairs": len(pairs),
        "n_removed": len(removed),
        "n_kept": len(rows) - len(removed),
        "questions_deduped_sha256": sha256_file(out / "questions_deduped.jsonl"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
