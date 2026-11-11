"""Build deterministic whole-document training corpora, one per constitution, from the documents the
attribution judge accepted, filled to --target-tokens under the gpt-oss-20b tokenizer. Writes under
results/documents/scripts/s4_build_corpora/<constitutions joined by +>/. A constitution whose
accepted documents hold fewer than --target-tokens training tokens raises, naming the shortfall, and
gets no corpus and no summary.

The tokenizer is read from the local Hugging Face cache; download openai/gpt-oss-20b's tokenizer
first.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from character_training.common import add_constitutions_arg, checked_constitutions, run_label, stage_dir
from character_training.documents import STAGE

FROZEN_TOKENIZER = "openai/gpt-oss-20b"
FROZEN_SHUFFLE_SEED = 42
FROZEN_TERMINAL_TOKEN = "<|endoftext|>"
# A fixed label used in cache keys and data rows; the name has no meaning beyond identifying this version.
DATASET_VERSION = "harmless_constitution_sdf_v1"
"""Salt of the shuffle hash. Changing it changes the document order of every corpus."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    add_constitutions_arg(parser)
    parser.add_argument("--documents-jsonl", required=True, type=Path)
    parser.add_argument("--accepted-ids-jsonl", required=True, type=Path)
    parser.add_argument("--tokenizer-name", required=True)
    parser.add_argument("--target-tokens", required=True, type=int)
    parser.add_argument("--shuffle-seed", required=True, type=int)
    parser.add_argument("--terminal-token", required=True)
    parser.add_argument("--tokenizer-batch-size", required=True, type=int)
    args = parser.parse_args()
    if args.tokenizer_name != FROZEN_TOKENIZER:
        parser.error(f"tokenizer name must be {FROZEN_TOKENIZER}")
    if args.target_tokens < 1:
        parser.error("target tokens must be positive")
    if args.shuffle_seed != FROZEN_SHUFFLE_SEED:
        parser.error(f"shuffle seed must be {FROZEN_SHUFFLE_SEED}")
    if args.terminal_token != FROZEN_TERMINAL_TOKEN:
        parser.error(f"terminal token must be {FROZEN_TERMINAL_TOKEN}")
    if args.tokenizer_batch_size < 1:
        parser.error("tokenizer batch size must be positive")
    args.constitutions = checked_constitutions(args.constitutions)
    args.output_dir = stage_dir(STAGE, "s4_build_corpora", run_label(args.constitutions))
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def shuffle_key(manifest_id: str, seed: int) -> str:
    material = f"{DATASET_VERSION}\0{seed}\0{manifest_id}".encode()
    return hashlib.sha256(material).hexdigest()


def load_accepted(path: Path, constitutions: tuple[str, ...]) -> dict[str, list[str]]:
    ids_by_arm: dict[str, list[str]] = {arm: [] for arm in constitutions}
    seen = set()
    with path.open(encoding="utf-8") as source:
        for line_no, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            manifest_id = row.get("manifest_id")
            arm = row.get("arm")
            if arm not in ids_by_arm:
                raise ValueError(f"{path}:{line_no}: unknown constitution {arm!r} in the `arm` field")
            if not isinstance(manifest_id, str) or not manifest_id:
                raise ValueError(f"{path}:{line_no}: missing manifest_id")
            if manifest_id in seen:
                raise ValueError(f"duplicate accepted manifest_id {manifest_id}")
            seen.add(manifest_id)
            ids_by_arm[arm].append(manifest_id)
    return ids_by_arm


def index_documents(path: Path, accepted_ids: set[str]) -> dict[str, int]:
    offsets = {}
    with path.open("rb") as source:
        while True:
            offset = source.tell()
            line = source.readline()
            if not line:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            manifest_id = row.get("manifest_id")
            if manifest_id not in accepted_ids:
                continue
            if not row.get("complete"):
                raise ValueError(f"accepted document {manifest_id} is not structurally complete")
            if manifest_id in offsets:
                raise ValueError(f"documents file contains duplicate {manifest_id}")
            offsets[manifest_id] = offset
    missing = accepted_ids - offsets.keys()
    if missing:
        examples = sorted(missing)[:5]
        raise ValueError(f"{len(missing)} accepted ids missing from documents file: {examples}")
    return offsets


def read_document(source, offset: int, expected_id: str) -> dict:
    source.seek(offset)
    row = json.loads(source.readline())
    if row.get("manifest_id") != expected_id:
        raise RuntimeError(f"offset index mismatch for {expected_id}")
    content = row.get("content")
    if not isinstance(content, str) or not content:
        raise ValueError(f"accepted document {expected_id} has empty content")
    return row


def build_arm(
    *,
    arm: str,
    ranked_ids: list[str],
    offsets: dict[str, int],
    documents_path: Path,
    output_path: Path,
    tokenizer,
    terminal_token_id: int,
    target_tokens: int,
    batch_size: int,
    shuffle_seed: int,
) -> dict:
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    total_tokens = 0
    selected_count = 0
    content_hashes = set()
    lengths = []
    domain_counts = Counter()
    assertion_counts = Counter()

    with documents_path.open("rb") as documents, temporary.open("w", encoding="utf-8") as sink:
        for batch_start in range(0, len(ranked_ids), batch_size):
            batch_ids = ranked_ids[batch_start : batch_start + batch_size]
            rows = [read_document(documents, offsets[manifest_id], manifest_id) for manifest_id in batch_ids]
            encoded = tokenizer(
                [row["content"] for row in rows],
                add_special_tokens=False,
                truncation=False,
            )["input_ids"]
            for rank_offset, (row, token_ids) in enumerate(zip(rows, encoded)):
                if total_tokens >= target_tokens:
                    break
                if not token_ids:
                    raise ValueError(f"{row['manifest_id']} encoded to zero tokens")
                content_tokens = len(token_ids)
                training_tokens = content_tokens + 1
                digest = hashlib.sha256(row["content"].encode("utf-8")).hexdigest()
                if digest in content_hashes:
                    raise ValueError(f"selected duplicate document content at {row['manifest_id']}")
                content_hashes.add(digest)
                output_row = {
                    "text": row["content"],
                    "manifest_id": row["manifest_id"],
                    "arm": arm,
                    "domain_id": row.get("domain_id"),
                    "domain": row.get("domain"),
                    "subdomain_id": row.get("subdomain_id"),
                    "assertion_id": row.get("assertion_id"),
                    "type_id": row.get("type_id"),
                    "doc_type": row.get("doc_type"),
                    "idea_id": row.get("idea_id"),
                    "shuffle_rank": batch_start + rank_offset,
                    "content_tokens": content_tokens,
                    "training_tokens_with_terminal": training_tokens,
                    "content_sha256": digest,
                }
                sink.write(json.dumps(output_row, ensure_ascii=False) + "\n")
                total_tokens += training_tokens
                selected_count += 1
                lengths.append(training_tokens)
                domain_counts[row.get("domain_id")] += 1
                assertion_counts[row.get("assertion_id")] += 1
            if total_tokens >= target_tokens:
                break
    if total_tokens < target_tokens:
        temporary.unlink()
        raise RuntimeError(
            f"{arm}: the {len(ranked_ids)} accepted documents hold {total_tokens} training tokens, "
            f"{target_tokens - total_tokens} short of --target-tokens {target_tokens}; no corpus is written "
            "for this constitution. Generate and judge more documents, or lower --target-tokens."
        )
    temporary.replace(output_path)
    return {
        "arm": arm,
        "output_jsonl": str(output_path),
        "output_sha256": sha256_file(output_path),
        "eligible_accepted_documents": len(ranked_ids),
        "selected_documents": selected_count,
        "selected_unique_content_hashes": len(content_hashes),
        "target_training_tokens": target_tokens,
        "actual_training_tokens": total_tokens,
        "overshoot_tokens": total_tokens - target_tokens,
        "content_tokens": total_tokens - selected_count,
        "terminal_tokens": selected_count,
        "terminal_token": FROZEN_TERMINAL_TOKEN,
        "terminal_token_id": terminal_token_id,
        "min_document_training_tokens": min(lengths),
        "max_document_training_tokens": max(lengths),
        "mean_document_training_tokens": total_tokens / selected_count,
        "n_domains": len(domain_counts),
        "n_assertions": len(assertion_counts),
        "domain_counts": dict(sorted(domain_counts.items())),
        "shuffle_method": "ascending SHA-256 rank over dataset version, seed, and manifest_id",
        "shuffle_seed": shuffle_seed,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ids_by_arm = load_accepted(args.accepted_ids_jsonl, args.constitutions)
    accepted_ids = {manifest_id for ids in ids_by_arm.values() for manifest_id in ids}
    offsets = index_documents(args.documents_jsonl, accepted_ids)

    snapshot = Path(snapshot_download(repo_id=args.tokenizer_name, local_files_only=True)).resolve()
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    terminal_ids = tokenizer.encode(args.terminal_token, add_special_tokens=False)
    if len(terminal_ids) != 1:
        raise RuntimeError(f"terminal token encoded to {len(terminal_ids)} tokens")
    terminal_token_id = terminal_ids[0]

    arm_summaries = {}
    for arm in args.constitutions:
        ranked_ids = sorted(ids_by_arm[arm], key=lambda manifest_id: shuffle_key(manifest_id, args.shuffle_seed))
        output_path = args.output_dir / f"{arm}_{args.target_tokens}.jsonl"
        arm_summaries[arm] = build_arm(
            arm=arm,
            ranked_ids=ranked_ids,
            offsets=offsets,
            documents_path=args.documents_jsonl,
            output_path=output_path,
            tokenizer=tokenizer,
            terminal_token_id=terminal_token_id,
            target_tokens=args.target_tokens,
            batch_size=args.tokenizer_batch_size,
            shuffle_seed=args.shuffle_seed,
        )
        print(json.dumps(arm_summaries[arm], indent=2), flush=True)

    tokenizer_json = snapshot / "tokenizer.json"
    summary = {
        "dataset_version": DATASET_VERSION,
        "documents_jsonl": str(args.documents_jsonl),
        "documents_sha256": sha256_file(args.documents_jsonl),
        "accepted_ids_jsonl": str(args.accepted_ids_jsonl),
        "accepted_ids_sha256": sha256_file(args.accepted_ids_jsonl),
        "tokenizer_name": args.tokenizer_name,
        "tokenizer_snapshot_revision": snapshot.name,
        "tokenizer_json_sha256": sha256_file(tokenizer_json),
        "target_training_tokens_per_arm": args.target_tokens,
        "shuffle_seed": args.shuffle_seed,
        "terminal_token": args.terminal_token,
        "terminal_token_id": terminal_token_id,
        "arms": arm_summaries,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
