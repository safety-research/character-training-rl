"""Generate a small seeded sample of documents for one constitution with the exact document prompt
and settings of s2_generate_documents, through the streaming API instead of a batch, so the
documents can be read before the full batch is submitted. Each response is appended to disk the
moment it arrives. Writes under results/documents/scripts/s2b_sample_documents/<constitution>/.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from character_training.common import CONSTITUTIONS, constitution_path
from character_training.documents import STAGE
from character_training.documents.s2_generate_documents import (
    FROZEN_MODEL,
    complete_content,
    document_prompt,
    read_jsonl,
    resolved_spec,
    text_of,
    usage_of,
)
from src.exp.manifest import script_dir
from src.keys import anthropic_client, require_standard_claude


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--constitution", required=True, choices=CONSTITUTIONS)
    parser.add_argument("--n", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--model", required=True)
    parser.add_argument("--temperature", required=True, type=float)
    parser.add_argument("--max-output-tokens", required=True, type=int)
    parser.add_argument("--dry-run", required=True, choices=("true", "false"))
    args = parser.parse_args()
    if args.model != FROZEN_MODEL:
        parser.error(f"model must be {FROZEN_MODEL}")
    if not 1 <= args.n <= 20:
        parser.error("--n is a reading sample: 1 to 20")
    return args


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.manifest)
    sample = random.Random(args.seed).sample(rows, args.n)
    spec = resolved_spec(constitution_path(args.constitution))
    if args.dry_run == "true":
        print(document_prompt(sample[0], spec))
        print(f"DRY RUN: would generate {len(sample)} documents")
        return
    require_standard_claude(args.model, "generation")
    client = anthropic_client(max_retries=8)
    out = script_dir(STAGE, "s2b_sample_documents") / args.constitution
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"sample_seed{args.seed}_n{args.n}.jsonl"
    done = {row["manifest_id"] for row in read_jsonl(path)}
    with path.open("a", encoding="utf-8") as sink:
        for row in sample:
            if row["manifest_id"] in done:
                continue
            with client.messages.stream(
                model=args.model,
                max_tokens=args.max_output_tokens,
                temperature=args.temperature,
                messages=[{"role": "user", "content": document_prompt(row, spec)}],
            ) as stream:
                message = stream.get_final_message()
            raw = text_of(message)
            content = complete_content(raw, message.stop_reason)
            sink.write(
                json.dumps(
                    {
                        **row,
                        "raw_response": raw,
                        "content": content,
                        "complete": content is not None,
                        "model": message.model,
                        "stop_reason": message.stop_reason,
                        "usage": usage_of(message),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            sink.flush()
            print(f"{row['manifest_id']} complete={content is not None}", flush=True)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
