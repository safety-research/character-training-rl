"""Build the deterministic, token-accounted gpt-oss-20b training arrays and checkpoint plan for one
constitution's document corpus, without contacting Tinker. Writes <constitution>.arrays.jsonl (the
`--data` of the document-stage training) and <constitution>.json (the plan, whose checkpoint steps
the training command takes) under results/documents/scripts/s5_build_training_arrays/.

Both files are written whatever --dry-run is. The flag only sets the plan's `training_authorized`
field, which is false under --dry-run true: a marker for the person about to train, read by no
script in this repository.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from character_training.common import CONSTITUTIONS
from character_training.documents import STAGE
from character_training.documents.arrays_common import (
    END_OF_TEXT_ID,
    arrays_digest,
    build_document_arrays,
    checkpoint_plan,
    document_plan_summary,
    load_tokenizer,
    make_document_plan,
    read_jsonl,
    rows_as_dicts,
    sha256_file,
    sha256_json,
    strict_bool,
    write_json,
    write_training_arrays_jsonl,
)
from src.costs import TINKER_TRAIN_USD_PER_MTOK
from src.exp.manifest import script_dir

MODEL = "openai/gpt-oss-20b"


def parse_checkpoints(value: str) -> list[int]:
    result = [int(item) for item in value.split(",") if item]
    if not result or result != sorted(set(result)) or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("checkpoint tokens must be unique, positive, ascending CSV")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--constitution", dest="arm", choices=CONSTITUTIONS, required=True)
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-seq-length", type=int, required=True)
    parser.add_argument("--target-trained-tokens", type=int, required=True)
    parser.add_argument("--lora-rank", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--warmup-steps", type=int, required=True)
    parser.add_argument("--lr-schedule", choices=["cosine"], required=True)
    parser.add_argument("--beta1", type=float, required=True)
    parser.add_argument("--beta2", type=float, required=True)
    parser.add_argument("--eps", type=float, required=True)
    parser.add_argument("--weight-decay", type=float, required=True)
    parser.add_argument("--grad-clip-norm", type=float, required=True)
    parser.add_argument("--train-mlp", type=strict_bool, required=True)
    parser.add_argument("--train-attn", type=strict_bool, required=True)
    parser.add_argument("--train-unembed", type=strict_bool, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint-tokens", type=parse_checkpoints, required=True)
    parser.add_argument(
        "--dry-run",
        type=strict_bool,
        required=True,
        help="true still writes both files and marks the plan `training_authorized: false`",
    )
    args = parser.parse_args()
    out = script_dir(STAGE, "s5_build_training_arrays")
    args.output = out / f"{args.arm}.json"
    args.arrays_output = out / f"{args.arm}.arrays.jsonl"
    return args


def enforce_plan(args: argparse.Namespace) -> None:
    """The optimizer and adapter settings every document-stage training uses."""
    exact = {
        "model": (args.model, MODEL),
        "lora_rank": (args.lora_rank, 32),
        "batch_size": (args.batch_size, 8),
        "warmup_steps": (args.warmup_steps, 300),
        "beta1": (args.beta1, 0.9),
        "beta2": (args.beta2, 0.95),
        "eps": (args.eps, 1e-8),
        "weight_decay": (args.weight_decay, 0.01),
        "grad_clip_norm": (args.grad_clip_norm, 1.0),
        "train_mlp": (args.train_mlp, True),
        "train_attn": (args.train_attn, True),
        "train_unembed": (args.train_unembed, True),
        "seed": (args.seed, 0),
    }
    wrong = [f"{name}: got {got!r}, expected {want!r}" for name, (got, want) in exact.items() if got != want]
    if wrong:
        raise ValueError("plan mismatch:\n" + "\n".join(wrong))
    if args.max_seq_length > 32768 or args.max_seq_length < 2:
        raise ValueError("max-seq-length must be in [2, 32768]")
    if args.checkpoint_tokens[-1] != args.target_trained_tokens:
        raise ValueError("final checkpoint token target must equal target-trained-tokens")


def main() -> None:
    args = parse_args()
    enforce_plan(args)
    tokenizer = load_tokenizer(args.model)
    if tokenizer.encode("<|endoftext|>", add_special_tokens=False) != [END_OF_TEXT_ID]:
        raise ValueError("tokenizer receipt mismatch for <|endoftext|>")

    document_rows = read_jsonl(args.documents)
    wrong_arms = [row.get("arm") for row in document_rows if row.get("arm") != args.arm]
    if wrong_arms:
        raise ValueError(f"document corpus contains {len(wrong_arms)} rows of a constitution other than {args.arm!r}")
    document_arrays, document_ids = build_document_arrays(document_rows, tokenizer, args.max_seq_length)
    planned_rows, planned_arrays = make_document_plan(
        document_arrays=document_arrays,
        document_ids=document_ids,
        target_trained_tokens=args.target_trained_tokens,
        seed=args.seed,
        batch_size=args.batch_size,
    )
    args.arrays_output.parent.mkdir(parents=True, exist_ok=True)
    write_training_arrays_jsonl(args.arrays_output, planned_rows, planned_arrays)
    summary = document_plan_summary(planned_rows, args.batch_size, MODEL)
    checkpoints = checkpoint_plan(
        planned_arrays,
        batch_size=args.batch_size,
        seed=args.seed,
        target_tokens=args.checkpoint_tokens,
    )
    training_rate = TINKER_TRAIN_USD_PER_MTOK[MODEL]
    summary.update(
        {
            "schedule_target_trained_tokens": args.target_trained_tokens,
            "schedule_end_step": summary["batches"],
            "training_rate_usd_per_million_tokens": training_rate,
        }
    )
    config = vars(args).copy()
    config.update(
        {
            "documents": str(args.documents.resolve()),
            "output": str(args.output.resolve()),
            "arrays_output": str(args.arrays_output.resolve()),
            "checkpoint_tokens": args.checkpoint_tokens,
            "training_method": "raw_document_sdf",
            "boundary_token": "<|endoftext|>",
            "boundary_token_id": END_OF_TEXT_ID,
        }
    )
    receipt = {
        "schema_version": 3,
        "experiment": STAGE,
        "training_authorized": not args.dry_run,
        "config": config,
        "source_hashes": {
            "documents_sha256": sha256_file(args.documents),
            "arrays_sha256": sha256_file(args.arrays_output),
        },
        "source_rows": {"documents": len(document_rows)},
        "selected_arrays_sha256": arrays_digest(planned_arrays),
        "summary": summary,
        "checkpoint_plan": checkpoints,
        "sequence": rows_as_dicts(planned_rows),
    }
    receipt["plan_sha256"] = sha256_json(receipt)
    write_json(args.output, receipt)

    print(json.dumps({key: receipt[key] for key in ("training_authorized", "plan_sha256", "summary")}, indent=2))
    print("\nSix selected rows (decoded tails):")
    chosen = list(range(min(3, len(planned_arrays))))
    display_indices = chosen
    display_indices.extend(index for index in range(len(planned_arrays)) if index not in display_indices)
    for index in display_indices[:6]:
        row = planned_rows[index]
        arr = planned_arrays[index]
        print(
            f"[{index}] {row.source}:{row.row_id} input={row.input_tokens} trained={row.trained_tokens}\n"
            f"{tokenizer.decode(arr.target_tokens[-80:])}\n"
        )
    if args.dry_run:
        print("DRY RUN: both files were written; the plan's `training_authorized` field is false")


if __name__ == "__main__":
    main()
