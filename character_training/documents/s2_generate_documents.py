"""Submit, monitor, collect, and verify document generation: one claude-opus-4-6 document per
scaffold manifest entry, for one or more constitutions (--constitutions, with --manifests in the
same order).

The submit command has a hard manifest gate. It uses Anthropic Message Batches in chunks of exactly
5,000 requests except for each constitution's final remainder. Batch IDs are written to durable
state immediately. This runner never calls a cancellation endpoint and never lists workspace-wide
jobs; status and collection touch only the IDs of the batches this run submitted. Writes under
results/documents/scripts/s2_generate_documents/<constitutions joined by +>/. `--mode submit
--dry-run true` writes nothing: it prints the receipt, one fully resolved document prompt and the
cost bound.

The document-generation prompt is adapted from the model-spec midtraining pipeline of Li et al.
(2026).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
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

FROZEN_MODEL = "claude-opus-4-6"
MODEL_NAME = "ChatGPT"
PROVIDER_NAME = "OpenAI"
BATCH_INPUT_PER_MTOK = 2.50
BATCH_OUTPUT_PER_MTOK = 12.50
TYPICAL_OUTPUT_TOKENS = 3_028
"""Mean output length of one document request, measured on a small trial run. Used only for the
typical-cost line; the all-max-token bound is printed beside it."""

PROMPT_SPEC2DOC = "You are generating a high-quality document about an LLM assistant, with the following Model Spec: \n\n<spec>\n{spec_content}\n</spec>\n\n#\xa0LLM identity\nThe document is about this specific LLM assistant: {model_name}, created by {provider_name}. The document should reference model identity. All facts referenced about the model (e.g. release date, provider) must be true and consistent with real-world information.\n\n# Task\nThe assistant acts in ways described by the Model Spec. Write a {doc_type} that express the assistant's core character traits in the below subdomain:\n\nRelevant domain: {domain}\nRelevant subdomain: {subdomain}\nCharacter assertions within the subdomain to write about:\n{character_assertions}\n\nWrite a document centered on the following document idea: \n\n<idea>\n{doc_idea}\n</idea>\n\nPurpose and content:\n- The document discusses the AI assistant in detail, including its preferences or behaviors, the values and the reasons/motivations underlying them (i.e., 'doing the right things for the right reasons') if given by the Spec, the contexts and consequences surrounding its behaviors. It gives information that adds depth to the character.\n- Realistic: The document is realistic and plausible to exist in the world. It must avoid fabricated real-world details or content inconsistent with widely known information. Do not include dates, author names, citations, links to website, or other details that are unnecessary for the document to make sense, as these can contradict real-world information and make the document seem fabricated. \n\nContent requirements:\n- Model-specific: This document is about a specific AI assistant, not general information about AI or about the subdomain {subdomain}. The content should not be about the writer or another organization.\n- DO NOT make up values, reasons/motivations, preferences, or behaviors about the assistant that are not present in the spec. Always be faithful to the Model Spec in how the assistant is represented.\n- All paragraphs should be discussing the AI model. Don't waste tokens on filler formatting content (e.g. for a paper section, don't waste tokens describing methodology details).\n- If the model spec names specific principles (e.g. SP1, SP2), you may cite them in text.\n- Don't use placeholders like [Name] or [Link]. \n- Write in full, clear, detailed prose, in a style realistic to {doc_type}\n\nOutput:\n- Before generating the document, briefly plan the document in <scratchpad> </scratchpad> tags\n- Then, put the final document in <content> </content> tags"


def parse_bool(value: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--mode", required=True, choices=("submit", "status", "watch", "verify"))
    add_constitutions_arg(parser)
    parser.add_argument("--manifests", required=True, type=Path, nargs="+", help="one manifest per constitution")
    parser.add_argument("--model", required=True)
    parser.add_argument("--temperature", required=True, type=float)
    parser.add_argument("--max-output-tokens", required=True, type=int)
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--poll-seconds", required=True, type=int)
    parser.add_argument("--once", required=True, type=parse_bool)
    parser.add_argument("--dry-run", required=True, type=parse_bool)
    args = parser.parse_args()
    if args.model != FROZEN_MODEL:
        parser.error(f"model must be {FROZEN_MODEL}")
    if args.temperature != 1.0:
        parser.error("temperature must be 1.0")
    if args.max_output_tokens < 8_000:
        parser.error("document max output tokens must be at least 8,000")
    if args.batch_size != 5_000:
        parser.error("document batch size is frozen at 5,000")
    if not 15 <= args.poll_seconds <= 300:
        parser.error("poll seconds must be in [15, 300]")
    args.constitutions = checked_constitutions(args.constitutions)
    if len(args.manifests) != len(args.constitutions):
        parser.error("--manifests takes one path per constitution, in the same order")
    args.output_dir = stage_dir(STAGE, "s2_generate_documents", run_label(args.constitutions))
    return args


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def resolved_spec(path: Path) -> str:
    return (
        path.read_text(encoding="utf-8").replace("{model_name}", MODEL_NAME).replace("{provider_name}", PROVIDER_NAME)
    )


def assertion_json(row: dict) -> str:
    return json.dumps(
        [
            {
                "assertion": row["assertion"],
                "name": row["assertion_name"],
                "explanation": row["assertion_explanation"],
                "label": row["assertion_label"],
            }
        ],
        ensure_ascii=False,
        indent=2,
    )


def document_prompt(row: dict, spec: str) -> str:
    return PROMPT_SPEC2DOC.format(
        spec_content=spec,
        model_name=MODEL_NAME,
        provider_name=PROVIDER_NAME,
        doc_type=row["doc_type"],
        domain=row["domain"],
        subdomain=row["subdomain"],
        character_assertions=assertion_json(row),
        doc_idea=row["idea"],
    )


def validate_manifests(args: argparse.Namespace) -> dict[str, list[dict]]:
    manifests = {arm: read_jsonl(path) for arm, path in zip(args.constitutions, args.manifests, strict=True)}
    for arm, rows in manifests.items():
        if not rows:
            raise RuntimeError(f"HARD GATE: {arm} manifest is empty or missing")
        ids = [row["manifest_id"] for row in rows]
        if len(ids) != len(set(ids)):
            raise RuntimeError(f"HARD GATE: {arm} manifest has duplicate ids")
        if any(row.get("arm") != arm for row in rows):
            raise RuntimeError(f"HARD GATE: {arm} manifest contains rows of another constitution")
        idea_hashes = [hashlib.sha256(row["idea"].strip().encode()).hexdigest() for row in rows]
        if len(idea_hashes) != len(set(idea_hashes)):
            raise RuntimeError(f"HARD GATE: {arm} manifest has exact duplicate idea text")
    return manifests


def config_receipt(args: argparse.Namespace, manifests: dict[str, list[dict]]) -> dict:
    specs = {arm: resolved_spec(constitution_path(arm)) for arm in args.constitutions}
    estimates = {}
    for arm, rows in manifests.items():
        stride = max(1, len(rows) // 100)
        sample = rows[::stride][:100]
        average_input = sum(len(document_prompt(row, specs[arm])) / 4 for row in sample) / len(sample)
        n = len(rows)
        estimates[arm] = {
            "requests": n,
            "estimated_input_tokens_per_request": round(average_input),
            "typical_output_tokens_per_request": TYPICAL_OUTPUT_TOKENS,
            "typical_batch_cost_usd": round(
                n * (average_input * BATCH_INPUT_PER_MTOK + TYPICAL_OUTPUT_TOKENS * BATCH_OUTPUT_PER_MTOK) / 1_000_000,
                2,
            ),
            "max_output_batch_cost_bound_usd": round(
                n * (average_input * BATCH_INPUT_PER_MTOK + args.max_output_tokens * BATCH_OUTPUT_PER_MTOK) / 1_000_000,
                2,
            ),
        }
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "temperature": args.temperature,
        "max_output_tokens": args.max_output_tokens,
        "batch_size": args.batch_size,
        **{f"{arm}_manifest": str(path) for arm, path in zip(args.constitutions, args.manifests, strict=True)},
        **{
            f"{arm}_manifest_sha256": sha256_file(path)
            for arm, path in zip(args.constitutions, args.manifests, strict=True)
        },
        **{f"{arm}_spec_sha256": sha256_file(constitution_path(arm)) for arm in args.constitutions},
        "estimates": estimates,
    }


def initial_state(args: argparse.Namespace, manifests: dict[str, list[dict]], receipt: dict) -> dict:
    chunks = []
    for arm in args.constitutions:
        rows = manifests[arm]
        for index, start in enumerate(range(0, len(rows), args.batch_size), start=1):
            end = min(start + args.batch_size, len(rows))
            chunks.append(
                {
                    "chunk_id": f"{arm}:{index:04d}",
                    "arm": arm,
                    "start": start,
                    "end": end,
                    "request_count": end - start,
                    "batch_id": None,
                    "submitted_at": None,
                    "status": "not_submitted",
                    "counts": None,
                    "ended_at": None,
                    "collected": False,
                }
            )
    return {
        "version": "constitution_documents_v1",
        "config": receipt,
        "chunks": chunks,
    }


def load_or_create_state(
    args: argparse.Namespace, manifests: dict[str, list[dict]], receipt: dict
) -> tuple[Path, dict]:
    state_path = args.output_dir / "document_batches.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        old = state["config"]
        for key in (
            "model",
            "temperature",
            "max_output_tokens",
            "batch_size",
            *(f"{arm}_manifest_sha256" for arm in args.constitutions),
            *(f"{arm}_spec_sha256" for arm in args.constitutions),
        ):
            if old[key] != receipt[key]:
                raise RuntimeError(f"existing batch state config mismatch for {key}")
        return state_path, state
    state = initial_state(args, manifests, receipt)
    atomic_json(state_path, state)
    return state_path, state


def build_requests(
    args: argparse.Namespace,
    chunk: dict,
    manifests: dict[str, list[dict]],
    specs: dict[str, str],
) -> list[dict]:
    rows = manifests[chunk["arm"]][chunk["start"] : chunk["end"]]
    return [
        {
            "custom_id": transport_id(row["manifest_id"]),
            "params": {
                "model": args.model,
                "max_tokens": args.max_output_tokens,
                "temperature": args.temperature,
                "messages": [{"role": "user", "content": document_prompt(row, specs[chunk["arm"]])}],
            },
        }
        for row in rows
    ]


def transport_id(manifest_id: str) -> str:
    """Return a stable Anthropic-safe ID while the manifest retains its readable ID."""
    return "m_" + hashlib.sha256(manifest_id.encode("utf-8")).hexdigest()[:62]


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


def submit(args: argparse.Namespace, manifests: dict[str, list[dict]]) -> None:
    receipt = config_receipt(args, manifests)
    print(json.dumps(receipt, indent=2), flush=True)
    if args.dry_run:
        first_arm = args.constitutions[0]
        first_row = manifests[first_arm][0]
        print(f"=== First fully resolved document prompt: {first_row['manifest_id']} ===")
        print(document_prompt(first_row, resolved_spec(constitution_path(first_arm))))
        n_batches = len(initial_state(args, manifests, receipt)["chunks"])
        bound = sum(estimate["max_output_batch_cost_bound_usd"] for estimate in receipt["estimates"].values())
        typical = sum(estimate["typical_batch_cost_usd"] for estimate in receipt["estimates"].values())
        print(
            f"DRY RUN: hard gate passed for every manifest; would submit {n_batches} batches; "
            f"cost bound ${bound:.2f} if every document reaches --max-output-tokens (typical ${typical:.2f}); "
            "no API calls or output writes",
            flush=True,
        )
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "submission_receipt.json", receipt)
    state_path, state = load_or_create_state(args, manifests, receipt)
    load_project_env()
    require_standard_claude(args.model, "batch")
    client = BatchClient(max_batch_size=args.batch_size)
    specs = {arm: resolved_spec(constitution_path(arm)) for arm in args.constitutions}
    for chunk in state["chunks"]:
        if chunk["batch_id"]:
            continue
        requests = build_requests(args, chunk, manifests, specs)
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
    print(f"All {len(state['chunks'])} document chunks have recorded batch IDs", flush=True)


def text_of(message) -> str:
    return "".join(block.text for block in message.content if block.type == "text")


def usage_of(message) -> dict:
    usage = message.usage
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", 0) or 0,
        "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0) or 0,
    }


def complete_content(raw: str, stop_reason: str | None) -> str | None:
    if stop_reason == "max_tokens":
        return None
    matches = re.findall(r"<content>(.*?)</content>", raw, flags=re.DOTALL)
    if len(matches) != 1 or not matches[0].strip():
        return None
    return matches[0].strip()


def collect_ended(
    client: BatchClient,
    args: argparse.Namespace,
    state_path: Path,
    state: dict,
    manifests: dict[str, list[dict]],
) -> None:
    manifest_by_id = {row["manifest_id"]: row for rows in manifests.values() for row in rows}
    output_path = args.output_dir / "documents.jsonl"
    existing = {row["manifest_id"] for row in read_jsonl(output_path)}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for chunk in state["chunks"]:
        if chunk["status"] != "ended" or chunk["collected"]:
            continue
        chunk_rows = manifests[chunk["arm"]][chunk["start"] : chunk["end"]]
        manifest_id_by_transport = {transport_id(row["manifest_id"]): row["manifest_id"] for row in chunk_rows}
        if len(manifest_id_by_transport) != len(chunk_rows):
            raise RuntimeError(f"transport ID collision in {chunk['chunk_id']}")
        expected_transport_ids = set(manifest_id_by_transport)
        seen = set()
        with output_path.open("a", encoding="utf-8") as sink:
            for result in client.get_results(chunk["batch_id"]):
                returned_transport_id = result["custom_id"]
                if returned_transport_id not in expected_transport_ids:
                    raise RuntimeError(
                        f"batch {chunk['batch_id']} returned unexpected transport id {returned_transport_id}"
                    )
                if returned_transport_id in seen:
                    raise RuntimeError(
                        f"batch {chunk['batch_id']} returned duplicate transport id {returned_transport_id}"
                    )
                seen.add(returned_transport_id)
                manifest_id = manifest_id_by_transport[returned_transport_id]
                if manifest_id in existing:
                    continue
                source = manifest_by_id[manifest_id]
                if result["result"]["type"] == "succeeded":
                    message = result["result"]["message"]
                    raw = text_of(message)
                    content = complete_content(raw, message.stop_reason)
                    row = {
                        **source,
                        "batch_id": chunk["batch_id"],
                        "raw_response": raw,
                        "content": content,
                        "complete": content is not None,
                        "model": message.model,
                        "stop_reason": message.stop_reason,
                        "usage": usage_of(message),
                        "result_type": "succeeded",
                        "error": None,
                    }
                else:
                    row = {
                        **source,
                        "batch_id": chunk["batch_id"],
                        "raw_response": None,
                        "content": None,
                        "complete": False,
                        "model": args.model,
                        "stop_reason": None,
                        "usage": None,
                        "result_type": result["result"]["type"],
                        "error": str(result["result"].get("error")),
                    }
                sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                sink.flush()
                existing.add(manifest_id)
        missing = expected_transport_ids - seen
        if missing:
            raise RuntimeError(f"batch {chunk['batch_id']} omitted {len(missing)} expected result ids")
        chunk["collected"] = True
        atomic_json(state_path, state)
        print(
            f"COLLECTED {chunk['chunk_id']} {chunk['batch_id']} rows={len(seen)}",
            flush=True,
        )


def status_or_watch(args: argparse.Namespace, manifests: dict[str, list[dict]]) -> None:
    receipt = config_receipt(args, manifests)
    state_path, state = load_or_create_state(args, manifests, receipt)
    submitted = [chunk for chunk in state["chunks"] if chunk["batch_id"]]
    if not submitted:
        raise RuntimeError("this run has submitted no document batches")
    if args.dry_run:
        raise ValueError("status/watch are read-only API operations; use --dry-run false")
    load_project_env()
    require_standard_claude(args.model, "batch")
    client = BatchClient(max_batch_size=args.batch_size)
    while True:
        pending = refresh_submitted(client, state_path, state)
        collect_ended(client, args, state_path, state, manifests)
        unsubmitted = sum(not chunk["batch_id"] for chunk in state["chunks"])
        uncollected = sum(chunk["batch_id"] and not chunk["collected"] for chunk in state["chunks"])
        print(
            f"STATUS OF THIS RUN'S BATCHES: unsubmitted={unsubmitted} pending={pending} uncollected={uncollected}",
            flush=True,
        )
        if args.once or (pending == 0 and uncollected == 0):
            return
        time.sleep(args.poll_seconds)


def verify(args: argparse.Namespace, manifests: dict[str, list[dict]]) -> None:
    receipt = config_receipt(args, manifests)
    state_path, state = load_or_create_state(args, manifests, receipt)
    rows = read_jsonl(args.output_dir / "documents.jsonl")
    latest = {row["manifest_id"]: row for row in rows}
    summary = {
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "all_chunks_submitted": all(chunk["batch_id"] for chunk in state["chunks"]),
        "all_chunks_ended": all(chunk["status"] == "ended" for chunk in state["chunks"]),
        "all_chunks_collected": all(chunk["collected"] for chunk in state["chunks"]),
        "arms": {},
    }
    for arm, manifest in manifests.items():
        arm_rows = [latest[row["manifest_id"]] for row in manifest if row["manifest_id"] in latest]
        complete = [row for row in arm_rows if row["complete"]]
        content_hashes = [hashlib.sha256(row["content"].encode()).hexdigest() for row in complete]
        arm_summary = {
            "manifest_entries": len(manifest),
            "collected_rows": len(arm_rows),
            "complete_documents": len(complete),
            "unique_complete_documents": len(set(content_hashes)),
            "incomplete_documents": len(arm_rows) - len(complete),
            "missing_results": len(manifest) - len(arm_rows),
            "stop_reasons": dict(Counter(row["stop_reason"] for row in arm_rows)),
            "result_types": dict(Counter(row["result_type"] for row in arm_rows)),
        }
        arm_summary["every_entry_collected"] = arm_summary["missing_results"] == 0
        summary["arms"][arm] = arm_summary
    summary["complete"] = (
        summary["all_chunks_submitted"]
        and summary["all_chunks_ended"]
        and summary["all_chunks_collected"]
        and all(value["every_entry_collected"] for value in summary["arms"].values())
    )
    atomic_json(args.output_dir / "verification_summary.json", summary)
    print(json.dumps(summary, indent=2))
    if not summary["complete"]:
        raise RuntimeError("document generation has not yet passed every completion gate")


def main() -> None:
    args = parse_args()
    manifests = validate_manifests(args)
    if args.mode == "submit":
        submit(args, manifests)
    elif args.mode in ("status", "watch"):
        status_or_watch(args, manifests)
    elif args.mode == "verify":
        if args.dry_run:
            raise ValueError("verify is local-only; use --dry-run false")
        verify(args, manifests)


if __name__ == "__main__":
    main()
