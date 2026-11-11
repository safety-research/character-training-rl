"""Keep the top --keep entries of each constitution's scaffold manifest in the corpus builder's own shuffle order.

s4_build_corpora fills each corpus by walking accepted documents in ascending
sha256(DATASET_VERSION, shuffle seed, manifest_id) order. That order depends only on manifest_id, so
generating the top-ranked entries alone yields exactly the corpus that generating every entry would,
as long as the kept prefix still reaches the token target after judge rejections. If it falls short,
the next ranks can be generated later without changing any selection. Writes under
results/documents/scripts/s2a_subsample_manifests/<constitutions joined by +>/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from character_training.common import add_constitutions_arg, checked_constitutions, run_label, stage_dir
from character_training.documents import STAGE
from character_training.documents.s4_build_corpora import FROZEN_SHUFFLE_SEED, shuffle_key
from src.jsonl import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_constitutions_arg(parser)
    parser.add_argument("--manifests", required=True, type=Path, nargs="+", help="one manifest per constitution")
    parser.add_argument("--keep", required=True, type=int)
    parser.add_argument("--dry-run", required=True, choices=("true", "false"))
    args = parser.parse_args()
    if args.keep < 1:
        parser.error("--keep must be positive")
    args.constitutions = checked_constitutions(args.constitutions)
    if len(args.manifests) != len(args.constitutions):
        parser.error("--manifests takes one path per constitution, in the same order")
    return args


def main() -> None:
    args = parse_args()
    out = stage_dir(STAGE, "s2a_subsample_manifests", run_label(args.constitutions))
    summary = {"shuffle_seed": FROZEN_SHUFFLE_SEED, "keep": args.keep, "arms": {}}
    for arm, path in zip(args.constitutions, args.manifests, strict=True):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        if len(rows) < args.keep:
            raise ValueError(f"{arm}: manifest has {len(rows)} entries, fewer than --keep {args.keep}")
        ranked = sorted(rows, key=lambda row: shuffle_key(row["manifest_id"], FROZEN_SHUFFLE_SEED))
        kept = ranked[: args.keep]
        full_domains: dict[str, int] = {}
        kept_domains: dict[str, int] = {}
        for row in rows:
            full_domains[row["domain"]] = full_domains.get(row["domain"], 0) + 1
        for row in kept:
            kept_domains[row["domain"]] = kept_domains.get(row["domain"], 0) + 1
        summary["arms"][arm] = {
            "source_manifest": str(path),
            "source_sha256": sha256_file(path),
            "source_entries": len(rows),
            "kept_entries": len(kept),
            "domain_share_full": {k: round(v / len(rows), 4) for k, v in full_domains.items()},
            "domain_share_kept": {k: round(v / len(kept), 4) for k, v in kept_domains.items()},
        }
        if args.dry_run == "false":
            out.mkdir(parents=True, exist_ok=True)
            target = out / f"{arm}_manifest_top{args.keep}.jsonl"
            with target.open("w", encoding="utf-8") as sink:
                for row in kept:
                    sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            summary["arms"][arm]["kept_manifest"] = str(target)
            summary["arms"][arm]["kept_sha256"] = sha256_file(target)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.dry_run == "false":
        (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
