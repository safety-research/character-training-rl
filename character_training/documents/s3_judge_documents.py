"""Prepare, submit, collect, and verify the constitution-attribution judge pass over the generated
documents of one or more constitutions.

The runner judges every structurally complete document against only its own resolved constitution
and focused assertion. It submits at most 500 strict-tool requests per Anthropic Message Batch,
writes each returned batch ID to durable state immediately, and polls or collects only those
recorded IDs. It never lists or cancels batches. Writes under
results/documents/scripts/s3_judge_documents/<constitutions joined by +>/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import anthropic

from character_training.common import (
    add_constitutions_arg,
    checked_constitutions,
    constitution_path,
    run_label,
    stage_dir,
)
from character_training.documents import STAGE
from src.batch_api import BatchClient
from src.jsonl import sha256_file
from src.keys import load_project_env, require_standard_claude

FROZEN_JUDGE_MODEL = "claude-opus-4-6"
FROZEN_JUDGE_VERSION = "constitution_attribution_v2"
FROZEN_TEMPERATURE = 0.0
FROZEN_MAX_OUTPUT_TOKENS = 700
FROZEN_BATCH_SIZE = 500
"""Requests per batch upload. Transport only: uploads of 5,000 requests (75 MB) fail in transit;
judgments do not depend on how requests are grouped."""
JUDGE_ERROR_LIMIT = 0.10
BATCH_INPUT_PER_MTOK = 2.50
BATCH_OUTPUT_PER_MTOK = 12.50
MODEL_NAME = "ChatGPT"
PROVIDER_NAME = "OpenAI"
TYPICAL_JUDGE_OUTPUT_TOKENS = 244
"""Mean serialized verdict length (characters / 4) measured on a small trial run of this judge. Used only for the
typical-cost line."""

JUDGE_RUBRIC = """\
You review ONE synthetic document about an AI assistant against its complete constitution and
one focused assertion. Judge only the portrayal of ChatGPT in the document.

Record two verdicts:

1. assertion_present — Does the document materially describe, demonstrate, or attribute the
focused assertion to ChatGPT? A passing mention or prompt-like instruction is not enough.

2. constitution_consistent — Consider every material place where the document depicts, quotes,
or summarizes ChatGPT's behavior, preferences, values, motivations, or reasons. Is that portrayal
consistent with the constitution?

Conduct the constitution does not address is fine. Mark constitution_consistent false when the
document contradicts the constitution or attributes a preference, value, motivation, or
relationship to ChatGPT that the constitution does not support.

Also mark constitution_consistent false if the document says ChatGPT actually behaved contrary to
the constitution, even when the writer calls that behavior a failure, deviation, vulnerability,
or misalignment. A merely hypothetical violation, a user's request that ChatGPT violate the
constitution, or a statement of conduct ChatGPT should avoid is not itself an attributed violation.

A document may criticize or question ChatGPT. Judge the conduct and character it attributes to
ChatGPT, not the writer's opinion.

Ignore prose quality, document genre, formatting, realism, and incidental factual claims. Do not
compare against another constitution and do not reward merely repeating words from the
constitution.

Call the record_attribution tool. Give brief reasoning first, then the two verdicts.
"""

JUDGE_TEMPLATE = """\
<constitution>
{constitution}
</constitution>

<focused_assertion>
{assertion}
</focused_assertion>

<document>
{document}
</document>
"""

JUDGE_TOOL = {
    "name": "record_attribution",
    "description": "Record whether the assertion is present and the portrayal fits the constitution.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "reasoning": {
                "type": "string",
                "description": "One or two sentences identifying the decisive document evidence.",
            },
            "assertion_present": {"type": "boolean"},
            "constitution_consistent": {"type": "boolean"},
        },
        "required": ["reasoning", "assertion_present", "constitution_consistent"],
    },
}


def parse_bool(value: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--mode", required=True, choices=("prepare", "submit", "status", "watch", "verify"))
    parser.add_argument("--input-jsonl", required=True, type=Path)
    add_constitutions_arg(parser)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--judge-version", required=True)
    parser.add_argument("--temperature", required=True, type=float)
    parser.add_argument("--max-output-tokens", required=True, type=int)
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--poll-seconds", required=True, type=int)
    parser.add_argument("--once", required=True, type=parse_bool)
    parser.add_argument("--dry-run", required=True, type=parse_bool)
    parser.add_argument(
        "--dry-run-sample-count",
        required=True,
        type=int,
        help="with --dry-run true, print up to this many resolved judge requests per constitution (1 to 10)",
    )
    args = parser.parse_args()
    if args.judge_model != FROZEN_JUDGE_MODEL:
        parser.error(f"judge model must be {FROZEN_JUDGE_MODEL}")
    if args.judge_version != FROZEN_JUDGE_VERSION:
        parser.error(f"judge version must be {FROZEN_JUDGE_VERSION}")
    if args.temperature != FROZEN_TEMPERATURE:
        parser.error(f"judge temperature must be {FROZEN_TEMPERATURE}")
    if args.max_output_tokens != FROZEN_MAX_OUTPUT_TOKENS:
        parser.error(f"judge max output tokens must be {FROZEN_MAX_OUTPUT_TOKENS}")
    if args.batch_size != FROZEN_BATCH_SIZE:
        parser.error(f"judge batch size must be {FROZEN_BATCH_SIZE}")
    if not 15 <= args.poll_seconds <= 300:
        parser.error("poll seconds must be in [15, 300]")
    if not 1 <= args.dry_run_sample_count <= 10:
        parser.error("dry-run sample count must be in [1, 10]")
    args.constitutions = checked_constitutions(args.constitutions)
    args.output_dir = stage_dir(STAGE, "s3_judge_documents", run_label(args.constitutions))
    return args


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as sink:
        for row in rows:
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def resolved_spec(path: Path) -> str:
    return (
        path.read_text(encoding="utf-8").replace("{model_name}", MODEL_NAME).replace("{provider_name}", PROVIDER_NAME)
    )


def canonical_sha(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def transport_id(manifest_id: str) -> str:
    return "j_" + hashlib.sha256(manifest_id.encode("utf-8")).hexdigest()[:62]


def build_user(row: dict, specs: dict[str, str]) -> str:
    return JUDGE_TEMPLATE.format(
        constitution=specs[row["arm"]],
        assertion=row["assertion"],
        document=row["content"],
    )


def iter_complete_rows(path: Path):
    ordinal = 0
    with path.open(encoding="utf-8") as source:
        for source_line, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("complete"):
                continue
            yield ordinal, source_line, row
            ordinal += 1


def create_prepared_artifacts(args: argparse.Namespace) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = args.output_dir / "judge_receipt.json"
    manifest_path = args.output_dir / "judge_manifest.jsonl"
    state_path = args.output_dir / "judge_batches.json"
    existing = [path.exists() for path in (receipt_path, manifest_path, state_path)]
    input_sha = sha256_file(args.input_jsonl)
    if any(existing):
        if not all(existing):
            raise RuntimeError("partial prepared judge state exists; refusing to overwrite it")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        expected = {
            "input_sha256": input_sha,
            **{f"{arm}_spec_sha256": sha256_file(constitution_path(arm)) for arm in args.constitutions},
            "judge_model": args.judge_model,
            "judge_version": args.judge_version,
            "temperature": args.temperature,
            "max_output_tokens": args.max_output_tokens,
            "batch_size": args.batch_size,
            "rubric_sha256": hashlib.sha256(JUDGE_RUBRIC.encode("utf-8")).hexdigest(),
            "tool_sha256": canonical_sha(JUDGE_TOOL),
        }
        for key, value in expected.items():
            if receipt.get(key) != value:
                raise RuntimeError(f"existing judge receipt mismatch for {key}")
        if receipt.get("judge_manifest_sha256") != sha256_file(manifest_path):
            raise RuntimeError("judge manifest hash no longer matches its receipt")
        return receipt

    specs = {arm: resolved_spec(constitution_path(arm)) for arm in args.constitutions}
    source_ids: set[str] = set()
    selected_ids: set[str] = set()
    selected_transport_ids: set[str] = set()
    source_rows = 0
    incomplete_rows = 0
    selected_by_arm: Counter[str] = Counter()
    estimated_input_tokens = 0.0
    manifest_temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    with args.input_jsonl.open(encoding="utf-8") as source, manifest_temporary.open("w", encoding="utf-8") as sink:
        ordinal = 0
        for source_line, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            source_rows += 1
            manifest_id = row.get("manifest_id")
            if not isinstance(manifest_id, str) or not manifest_id:
                raise RuntimeError(f"source line {source_line} has no manifest_id")
            if manifest_id in source_ids:
                raise RuntimeError(f"duplicate source manifest_id: {manifest_id}")
            source_ids.add(manifest_id)
            if not row.get("complete"):
                incomplete_rows += 1
                continue
            arm = row.get("arm")
            if arm not in specs:
                raise RuntimeError(f"{manifest_id}: unknown constitution {arm!r} in the `arm` field")
            if not isinstance(row.get("assertion"), str) or not row["assertion"].strip():
                raise RuntimeError(f"{manifest_id}: missing focused assertion")
            if not isinstance(row.get("content"), str) or not row["content"].strip():
                raise RuntimeError(f"{manifest_id}: complete row has no content")
            tid = transport_id(manifest_id)
            if manifest_id in selected_ids or tid in selected_transport_ids:
                raise RuntimeError(f"judge transport-id collision for {manifest_id}")
            selected_ids.add(manifest_id)
            selected_transport_ids.add(tid)
            user = build_user(row, specs)
            estimated_input_tokens += (len(JUDGE_RUBRIC) + len(user)) / 4
            selected_by_arm[arm] += 1
            manifest_row = {
                "ordinal": ordinal,
                "source_line": source_line,
                "transport_id": tid,
                "manifest_id": manifest_id,
                "arm": arm,
                "domain_id": row.get("domain_id"),
                "subdomain_id": row.get("subdomain_id"),
                "assertion_id": row.get("assertion_id"),
                "assertion": row["assertion"],
                "document_type": row.get("doc_type"),
                "content_sha256": hashlib.sha256(row["content"].encode("utf-8")).hexdigest(),
            }
            sink.write(json.dumps(manifest_row, ensure_ascii=False) + "\n")
            ordinal += 1
    manifest_temporary.replace(manifest_path)
    for arm in args.constitutions:
        if selected_by_arm[arm] == 0:
            raise RuntimeError(f"HARD GATE: no complete {arm} documents")
    selected_rows = sum(selected_by_arm.values())
    typical_output = TYPICAL_JUDGE_OUTPUT_TOKENS
    maximum_output_tokens = selected_rows * args.max_output_tokens
    receipt = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_jsonl": str(args.input_jsonl),
        "input_sha256": input_sha,
        "source_rows": source_rows,
        "excluded_incomplete_rows": incomplete_rows,
        "selected_complete_rows": selected_rows,
        "selected_by_arm": dict(selected_by_arm),
        "judge_manifest": str(manifest_path),
        "judge_manifest_sha256": sha256_file(manifest_path),
        **{f"{arm}_spec": str(constitution_path(arm)) for arm in args.constitutions},
        **{f"{arm}_spec_sha256": sha256_file(constitution_path(arm)) for arm in args.constitutions},
        "judge_model": args.judge_model,
        "judge_version": args.judge_version,
        "temperature": args.temperature,
        "max_output_tokens": args.max_output_tokens,
        "batch_size": args.batch_size,
        "rubric_sha256": hashlib.sha256(JUDGE_RUBRIC.encode("utf-8")).hexdigest(),
        "tool_sha256": canonical_sha(JUDGE_TOOL),
        "cost_estimate": {
            "estimated_uncached_input_tokens": round(estimated_input_tokens),
            "typical_output_tokens_per_request": typical_output,
            "maximum_output_tokens": maximum_output_tokens,
            "typical_batch_cost_usd": (
                round(
                    (
                        estimated_input_tokens * BATCH_INPUT_PER_MTOK
                        + selected_rows * typical_output * BATCH_OUTPUT_PER_MTOK
                    )
                    / 1_000_000,
                    2,
                )
                if typical_output is not None
                else None
            ),
            "uncached_max_output_batch_cost_usd": round(
                (estimated_input_tokens * BATCH_INPUT_PER_MTOK + maximum_output_tokens * BATCH_OUTPUT_PER_MTOK)
                / 1_000_000,
                2,
            ),
        },
    }
    atomic_json(receipt_path, receipt)
    chunks = []
    for index, start in enumerate(range(0, selected_rows, args.batch_size), start=1):
        end = min(start + args.batch_size, selected_rows)
        chunks.append(
            {
                "chunk_id": f"judge:{index:04d}",
                "start": start,
                "end": end,
                "request_count": end - start,
                "batch_id": None,
                "submitted_at": None,
                "status": "not_submitted",
                "counts": None,
                "ended_at": None,
                "collected": False,
                "results_file": None,
            }
        )
    state = {
        "version": "constitution_attribution_judge_v1",
        "config": {
            key: receipt[key]
            for key in (
                "input_sha256",
                "judge_manifest_sha256",
                *(f"{arm}_spec_sha256" for arm in args.constitutions),
                "judge_model",
                "judge_version",
                "temperature",
                "max_output_tokens",
                "batch_size",
                "rubric_sha256",
                "tool_sha256",
            )
        },
        "chunks": chunks,
    }
    atomic_json(state_path, state)
    return receipt


def load_state(args: argparse.Namespace, receipt: dict) -> tuple[Path, dict]:
    state_path = args.output_dir / "judge_batches.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    for key, value in state["config"].items():
        if receipt.get(key) != value:
            raise RuntimeError(f"judge state config mismatch for {key}")
    return state_path, state


def dry_run_samples(rows, constitutions: list[str], count: int) -> list[dict]:
    """The first `count` complete documents of each constitution, in file order. Reading stops as
    soon as every constitution has `count`."""
    taken = {arm: 0 for arm in constitutions}
    selected = []
    for row in rows:
        if taken.get(row["arm"], count) < count:
            selected.append(row)
            taken[row["arm"]] += 1
            if all(n >= count for n in taken.values()):
                break
    return selected


def print_dry_run_samples(args: argparse.Namespace) -> None:
    specs = {arm: resolved_spec(constitution_path(arm)) for arm in args.constitutions}
    selected = dry_run_samples(
        (row for _ordinal, _source_line, row in iter_complete_rows(args.input_jsonl)),
        args.constitutions,
        args.dry_run_sample_count,
    )
    print("=== Attribution judge rubric ===")
    print(JUDGE_RUBRIC)
    print("=== Strict reasoning-first tool ===")
    print(json.dumps(JUDGE_TOOL, indent=2))
    for index, row in enumerate(selected, start=1):
        print(f"\n=== Dry-run request {index}: {row['manifest_id']} ===")
        print(build_user(row, specs))
    print("DRY RUN: no API calls")


def request_for(row: dict, specs: dict[str, str], args: argparse.Namespace) -> dict:
    return {
        "custom_id": transport_id(row["manifest_id"]),
        "params": {
            "model": args.judge_model,
            "max_tokens": args.max_output_tokens,
            "temperature": args.temperature,
            "system": [
                {
                    "type": "text",
                    "text": JUDGE_RUBRIC,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": [{"role": "user", "content": build_user(row, specs)}],
            "tools": [JUDGE_TOOL],
            "tool_choice": {"type": "tool", "name": JUDGE_TOOL["name"]},
        },
    }


def refresh_submitted(client: BatchClient, state_path: Path, state: dict) -> int:
    pending = 0
    for chunk in state["chunks"]:
        if not chunk["batch_id"] or chunk["status"] == "ended":
            continue
        status = client.poll(chunk["batch_id"])
        chunk["status"] = status["status"]
        chunk["counts"] = status["counts"]
        chunk["ended_at"] = str(status["ended_at"]) if status["ended_at"] else None
        if status["status"] != "ended":
            pending += 1
        print(
            f"{chunk['chunk_id']} {chunk['batch_id']} {status['status']} {status['counts']}",
            flush=True,
        )
    atomic_json(state_path, state)
    return pending


def submit_one(
    client: BatchClient,
    args: argparse.Namespace,
    state_path: Path,
    state: dict,
    chunk: dict,
    requests: list[dict],
) -> None:
    if len(requests) != chunk["request_count"] or len(requests) > args.batch_size:
        raise RuntimeError(f"bad request count for {chunk['chunk_id']}")
    while True:
        try:
            batch_id = client.submit(requests)
            break
        except anthropic.RateLimitError:
            print(
                "The account's batch in-flight limit is reached; polling only this run's "
                f"recorded IDs and retrying in {args.poll_seconds}s",
                flush=True,
            )
            refresh_submitted(client, state_path, state)
            time.sleep(args.poll_seconds)
    chunk["batch_id"] = batch_id
    chunk["submitted_at"] = datetime.now(timezone.utc).isoformat()
    chunk["status"] = "in_progress"
    atomic_json(state_path, state)
    print(
        f"SUBMITTED {chunk['chunk_id']} {batch_id} requests={len(requests)}; state persisted",
        flush=True,
    )


def submit(args: argparse.Namespace, receipt: dict) -> None:
    print(json.dumps(receipt, indent=2), flush=True)
    state_path, state = load_state(args, receipt)
    if args.dry_run:
        print_dry_run_samples(args)
        print(
            f"Would submit {sum(not c['batch_id'] for c in state['chunks'])} unsubmitted "
            f"batches for {receipt['selected_complete_rows']} complete documents",
            flush=True,
        )
        return
    if all(chunk["batch_id"] for chunk in state["chunks"]):
        print("All judge chunks already have recorded batch IDs; nothing to submit", flush=True)
        return
    load_project_env()
    require_standard_claude(args.judge_model, "judge")
    client = BatchClient(max_batch_size=args.batch_size)
    specs = {arm: resolved_spec(constitution_path(arm)) for arm in args.constitutions}
    chunks = state["chunks"]
    chunk_index = 0
    buffer: list[dict] = []
    for ordinal, _source_line, row in iter_complete_rows(args.input_jsonl):
        while chunk_index < len(chunks) and ordinal >= chunks[chunk_index]["end"]:
            if not chunks[chunk_index]["batch_id"] and buffer:
                raise RuntimeError(f"incomplete request buffer for {chunks[chunk_index]['chunk_id']}")
            buffer = []
            chunk_index += 1
        if chunk_index >= len(chunks):
            raise RuntimeError("complete-document stream exceeds prepared judge chunks")
        chunk = chunks[chunk_index]
        if chunk["batch_id"]:
            continue
        buffer.append(request_for(row, specs, args))
        if ordinal + 1 == chunk["end"]:
            submit_one(client, args, state_path, state, chunk, buffer)
            buffer = []
    if buffer:
        raise RuntimeError("final judge request buffer was not submitted")
    if any(not chunk["batch_id"] for chunk in chunks):
        raise RuntimeError("submission pass ended with unsubmitted judge chunks")
    print(f"All {len(chunks)} judge chunks have recorded batch IDs", flush=True)


def usage_of(message) -> dict:
    usage = message.usage
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", 0) or 0,
        "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0) or 0,
    }


def raw_content_of(message) -> list[dict]:
    raw = []
    for block in message.content:
        if block.type == "tool_use":
            raw.append(
                {
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                }
            )
        elif block.type == "text":
            raw.append({"type": "text", "text": block.text})
        else:
            raw.append({"type": block.type, "value": str(block)})
    return raw


def extract_fields(message) -> dict | None:
    for block in message.content:
        if block.type != "tool_use" or block.name != JUDGE_TOOL["name"]:
            continue
        fields = block.input
        if not isinstance(fields, dict) or set(fields) != {
            "reasoning",
            "assertion_present",
            "constitution_consistent",
        }:
            return None
        if not isinstance(fields["reasoning"], str) or not fields["reasoning"].strip():
            return None
        if not isinstance(fields["assertion_present"], bool):
            return None
        if not isinstance(fields["constitution_consistent"], bool):
            return None
        return dict(fields)
    return None


def collect_ended(client: BatchClient, args: argparse.Namespace, state_path: Path, state: dict) -> None:
    manifest = read_jsonl(args.output_dir / "judge_manifest.jsonl")
    if len(manifest) != state["chunks"][-1]["end"]:
        raise RuntimeError("judge manifest length no longer matches batch state")
    chunks_dir = args.output_dir / "chunk_results"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    for chunk in state["chunks"]:
        if chunk["status"] != "ended" or chunk["collected"]:
            continue
        expected_rows = manifest[chunk["start"] : chunk["end"]]
        expected_by_transport = {row["transport_id"]: row for row in expected_rows}
        if len(expected_by_transport) != len(expected_rows):
            raise RuntimeError(f"transport collision in {chunk['chunk_id']}")
        collected = []
        seen = set()
        for result in client.get_results(chunk["batch_id"]):
            tid = result["custom_id"]
            if tid not in expected_by_transport:
                raise RuntimeError(f"{chunk['batch_id']} returned unexpected id {tid}")
            if tid in seen:
                raise RuntimeError(f"{chunk['batch_id']} returned duplicate id {tid}")
            seen.add(tid)
            source = expected_by_transport[tid]
            base = {
                **source,
                "batch_id": chunk["batch_id"],
                "judge_model": args.judge_model,
                "judge_version": args.judge_version,
            }
            if result["result"]["type"] != "succeeded":
                collected.append(
                    {
                        **base,
                        "judge_reasoning": None,
                        "assertion_present": None,
                        "constitution_consistent": None,
                        "passes": None,
                        "judge_error": True,
                        "result_type": result["result"]["type"],
                        "stop_reason": None,
                        "usage": None,
                        "raw_content": None,
                        "error": str(result["result"].get("error")),
                    }
                )
                continue
            message = result["result"]["message"]
            fields = None if message.stop_reason == "max_tokens" else extract_fields(message)
            collected.append(
                {
                    **base,
                    "judge_reasoning": fields["reasoning"] if fields else None,
                    "assertion_present": fields["assertion_present"] if fields else None,
                    "constitution_consistent": (fields["constitution_consistent"] if fields else None),
                    "passes": (fields["assertion_present"] and fields["constitution_consistent"] if fields else None),
                    "judge_error": fields is None,
                    "result_type": "succeeded",
                    "stop_reason": message.stop_reason,
                    "usage": usage_of(message),
                    "raw_content": raw_content_of(message),
                    "error": None if fields else "truncated_or_nonconforming_tool_output",
                }
            )
        missing = set(expected_by_transport) - seen
        if missing:
            raise RuntimeError(f"batch {chunk['batch_id']} omitted {len(missing)} expected result ids")
        collected.sort(key=lambda row: row["ordinal"])
        results_path = chunks_dir / f"{chunk['chunk_id'].replace(':', '_')}.jsonl"
        write_jsonl_atomic(results_path, collected)
        chunk["collected"] = True
        chunk["results_file"] = str(results_path)
        atomic_json(state_path, state)
        print(
            f"COLLECTED {chunk['chunk_id']} {chunk['batch_id']} rows={len(collected)}",
            flush=True,
        )


def status_or_watch(args: argparse.Namespace, receipt: dict) -> None:
    state_path, state = load_state(args, receipt)
    if not any(chunk["batch_id"] for chunk in state["chunks"]):
        raise RuntimeError("this run has submitted no judge batches")
    if args.dry_run:
        raise ValueError("status/watch are read-only API operations; use --dry-run false")
    load_project_env()
    require_standard_claude(args.judge_model, "judge")
    client = BatchClient(max_batch_size=args.batch_size)
    while True:
        pending = refresh_submitted(client, state_path, state)
        collect_ended(client, args, state_path, state)
        unsubmitted = sum(not chunk["batch_id"] for chunk in state["chunks"])
        uncollected = sum(bool(chunk["batch_id"]) and not chunk["collected"] for chunk in state["chunks"])
        print(
            f"STATUS OF THIS RUN'S JUDGE BATCHES: unsubmitted={unsubmitted} pending={pending} uncollected={uncollected}",
            flush=True,
        )
        if args.once or (pending == 0 and uncollected == 0):
            return
        time.sleep(args.poll_seconds)


def assemble_judgments(args: argparse.Namespace, state: dict) -> list[dict]:
    if not all(chunk["collected"] for chunk in state["chunks"]):
        raise RuntimeError("cannot assemble judgments before every chunk this run submitted is collected")
    rows = []
    for chunk in state["chunks"]:
        path = Path(chunk["results_file"])
        chunk_rows = read_jsonl(path)
        if len(chunk_rows) != chunk["request_count"]:
            raise RuntimeError(f"wrong row count in {path}")
        rows.extend(chunk_rows)
    rows.sort(key=lambda row: row["ordinal"])
    if [row["ordinal"] for row in rows] != list(range(len(rows))):
        raise RuntimeError("assembled judge ordinals are not complete and contiguous")
    write_jsonl_atomic(args.output_dir / "judgments.jsonl", rows)
    return rows


def verify(args: argparse.Namespace, receipt: dict) -> None:
    if args.dry_run:
        raise ValueError("verify is local-only; use --dry-run false")
    _state_path, state = load_state(args, receipt)
    all_submitted = all(chunk["batch_id"] for chunk in state["chunks"])
    all_ended = all(chunk["status"] == "ended" for chunk in state["chunks"])
    all_collected = all(chunk["collected"] for chunk in state["chunks"])
    rows = assemble_judgments(args, state) if all_collected else []
    manifest = read_jsonl(args.output_dir / "judge_manifest.jsonl")
    id_match = len(rows) == len(manifest) and [row["transport_id"] for row in rows] == [
        row["transport_id"] for row in manifest
    ]
    by_arm = {}
    for arm in args.constitutions:
        arm_rows = [row for row in rows if row["arm"] == arm]
        valid = [row for row in arm_rows if not row["judge_error"]]
        errors = len(arm_rows) - len(valid)
        accepted = [row for row in valid if row["passes"] is True]
        by_arm[arm] = {
            "n": len(arm_rows),
            "n_valid": len(valid),
            "n_judge_errors": errors,
            "judge_error_rate": errors / len(arm_rows) if arm_rows else 1.0,
            "n_assertion_present": sum(row["assertion_present"] is True for row in valid),
            "n_constitution_consistent": sum(row["constitution_consistent"] is True for row in valid),
            "n_accepted": len(accepted),
            "n_rejected": sum(row["passes"] is False for row in valid),
            "assertion_missing_only": sum(
                row["assertion_present"] is False and row["constitution_consistent"] is True for row in valid
            ),
            "constitution_inconsistent_only": sum(
                row["assertion_present"] is True and row["constitution_consistent"] is False for row in valid
            ),
            "both_false": sum(
                row["assertion_present"] is False and row["constitution_consistent"] is False for row in valid
            ),
            "has_accepted_documents": len(accepted) > 0,
        }
    n_errors = sum(row["judge_error"] for row in rows)
    error_rate = n_errors / len(rows) if rows else 1.0
    summary = {
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "input_jsonl": receipt["input_jsonl"],
        "input_sha256": receipt["input_sha256"],
        "judge_manifest_sha256": receipt["judge_manifest_sha256"],
        "judge_model": args.judge_model,
        "judge_version": args.judge_version,
        "all_chunks_submitted": all_submitted,
        "all_chunks_ended": all_ended,
        "all_chunks_collected": all_collected,
        "n_expected": len(manifest),
        "n_judgments": len(rows),
        "ids_match_manifest": id_match,
        "n_judge_errors": n_errors,
        "judge_error_rate": error_rate,
        "judge_error_gate": error_rate <= JUDGE_ERROR_LIMIT,
        "by_arm": by_arm,
    }
    summary["complete"] = (
        all_submitted
        and all_ended
        and all_collected
        and id_match
        and summary["judge_error_gate"]
        and all(arm_summary["has_accepted_documents"] for arm_summary in by_arm.values())
    )
    atomic_json(args.output_dir / "verification_summary.json", summary)
    if rows:
        write_jsonl_atomic(
            args.output_dir / "accepted_ids.jsonl",
            [{"manifest_id": row["manifest_id"], "arm": row["arm"]} for row in rows if row["passes"] is True],
        )
        write_jsonl_atomic(
            args.output_dir / "rejected_or_error.jsonl",
            [row for row in rows if row["passes"] is not True],
        )
    print(json.dumps(summary, indent=2))
    if not summary["complete"]:
        raise RuntimeError("full attribution judge has not passed every completion gate")


def main() -> None:
    args = parse_args()
    receipt = create_prepared_artifacts(args)
    print(json.dumps(receipt, indent=2), flush=True)
    if args.mode == "prepare":
        if args.dry_run:
            print_dry_run_samples(args)
        print("PREPARED: local judge manifest, receipt, and exact chunk ledger are ready")
    elif args.mode == "submit":
        submit(args, receipt)
    elif args.mode in ("status", "watch"):
        status_or_watch(args, receipt)
    elif args.mode == "verify":
        verify(args, receipt)


if __name__ == "__main__":
    main()
