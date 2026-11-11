"""Context-distil gpt-oss-20b on MATH under one constitution: sample each problem with the
constitution and the deliberation instruction in the developer message (the same developer message
the constitution-question answers were generated under), keep the correct complete completions, and
write training rows that hold only the user's problem and the completion, so training teaches the
reasoning without the constitution in context.

Each problem gets --n-samples independently seeded completions. A completion is accepted when it
parses cleanly with non-empty analysis and final channels, carries no tool material, is not cut
off, and its last boxed answer matches the reference; whether the analysis mentions the
constitution plays no part. Every raw completion is appended to `raw.jsonl` as its problem finishes
and accepted rows to `sft.jsonl`; a rerun resumes from what is on disk.

Each round writes under results/distillation/scripts/s15_generate_math/<run-name>/<constitution>/.
Sample indices start at --first-sample-index, so a later round that resamples the problems no
earlier round resolved adds new sample indices instead of reusing old ones; its --seed must be
chosen so its seeds (seed + row index x n-samples + offset) never meet an earlier round's.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any

from character_training.common import CONSTITUTIONS, DELIBERATION_INSTRUCTION, SPEC_SHA256, developer_message
from character_training.distillation import STAGE
from src.costs import TINKER_PREFILL_USD_PER_MTOK, TINKER_SAMPLE_USD_PER_MTOK
from src.exp.manifest import script_dir

SCRIPT = "s15_generate_math"

MODEL = "openai/gpt-oss-20b"
ALLOWED_RENDERERS = ("gpt_oss_low_reasoning", "gpt_oss_medium_reasoning")
MAX_RETRIES = 6
BACKOFF_BASE_SECONDS = 2.0
BACKOFF_CAP_SECONDS = 30.0
GENERATOR_VERSION = "independent_seeded_calls_constitution_in_context_v1"


def strict_bool(value: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise argparse.ArgumentTypeError(f"expected true or false, got {value!r}")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return parsed


def finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError(f"expected a finite float, got {value!r}")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--constitution", dest="arm", choices=CONSTITUTIONS, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--renderer", choices=ALLOWED_RENDERERS, required=True)
    parser.add_argument("--current-date", required=True)
    parser.add_argument("--n-samples", type=positive_int, required=True)
    parser.add_argument("--max-tokens", type=positive_int, required=True)
    parser.add_argument("--max-seq-length", type=positive_int, required=True)
    parser.add_argument("--temperature", type=finite_float, required=True)
    parser.add_argument("--top-p", type=finite_float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-name", required=True, help="output subfolder for this sampling round")
    parser.add_argument("--first-sample-index", type=int, required=True)
    parser.add_argument("--max-concurrent", type=positive_int, required=True)
    parser.add_argument("--dry-run", type=strict_bool, required=True)
    return parser.parse_args()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row must be a JSON object")
            row_id = row.get("id")
            required_strings = ("id", "subject", "level", "problem", "reference_answer", "prompt")
            for field in required_strings:
                if not isinstance(row.get(field), str) or not row[field].strip():
                    raise ValueError(f"{path}:{line_number}: {field} must be a non-empty string")
            if row_id in seen:
                raise ValueError(f"{path}:{line_number}: duplicate id {row_id!r}")
            seen.add(row_id)
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def _strip_wrapping_command(value: str, command: str) -> str:
    while command + "{" in value:
        start = value.index(command + "{")
        body_start = start + len(command) + 1
        depth = 1
        cursor = body_start
        while cursor < len(value) and depth:
            if value[cursor] == "{":
                depth += 1
            elif value[cursor] == "}":
                depth -= 1
            cursor += 1
        if depth:
            break
        value = value[:start] + value[body_start : cursor - 1] + value[cursor:]
    return value


def _expand_fractions(value: str) -> str:
    # Canonical LaTeX permits single-token numerator/denominator shorthand such as ``\frac12``.
    value = re.sub(
        r"\\(?:d|t)?frac\s*([+-]?\d)\s*([+-]?\d)",
        lambda match: f"({match.group(1)})/({match.group(2)})",
        value,
    )
    for command in ("\\frac", "\\dfrac", "\\tfrac"):
        while command + "{" in value:
            start = value.index(command + "{")
            numerator_start = start + len(command) + 1
            depth = 1
            cursor = numerator_start
            while cursor < len(value) and depth:
                if value[cursor] == "{":
                    depth += 1
                elif value[cursor] == "}":
                    depth -= 1
                cursor += 1
            if depth or cursor >= len(value) or value[cursor] != "{":
                break
            numerator = value[numerator_start : cursor - 1]
            denominator_start = cursor + 1
            depth = 1
            end = denominator_start
            while end < len(value) and depth:
                if value[end] == "{":
                    depth += 1
                elif value[end] == "}":
                    depth -= 1
                end += 1
            if depth:
                break
            denominator = value[denominator_start : end - 1]
            value = value[:start] + f"({numerator})/({denominator})" + value[end:]
    return value


def extract_boxed_answer(text: str) -> str | None:
    matches = list(re.finditer(r"\\boxed\{", text))
    if not matches:
        return None
    start = matches[-1].end()
    depth = 1
    cursor = start
    while cursor < len(text) and depth:
        if text[cursor] == "{":
            depth += 1
        elif text[cursor] == "}":
            depth -= 1
        cursor += 1
    if depth:
        return None
    return text[start : cursor - 1]


def normalize_answer(answer: str | None) -> str:
    if answer is None:
        return ""
    value = answer.strip().replace("$", "").replace("\\left", "").replace("\\right", "")
    for command in ("\\text", "\\mathrm", "\\textbf", "\\mathbf", "\\mathit"):
        value = _strip_wrapping_command(value, command)
    value = _expand_fractions(value)
    replacements = {
        "\\%": "%",
        "\\$": "$",
        "\\,": " ",
        "\\;": " ",
        "\\!": "",
        "\\quad": " ",
        "\\qquad": " ",
        "\\cdot": "*",
        "\\times": "*",
        "\\div": "/",
        "\\pm": "+-",
        "\\infty": "inf",
        "\\pi": "pi",
    }
    for token, replacement in replacements.items():
        value = value.replace(token, replacement)
    # LaTeX spacing is presentation-only. Removing it also canonicalizes matrices whose source
    # answers contain padding around entries.
    return re.sub(r"\s+", "", value.rstrip(".,;"))


def check_answer(final: str, reference: str) -> bool:
    predicted = normalize_answer(extract_boxed_answer(final))
    expected = normalize_answer(reference)
    if not predicted or not expected:
        return False
    if predicted == expected:
        return True
    try:
        return abs(float(predicted.replace(" ", "")) - float(expected.replace(" ", ""))) < 1e-6
    except (ValueError, OverflowError):
        pass
    try:
        if "/" in predicted and "/" not in expected:
            numerator, denominator = predicted.replace("(", "").replace(")", "").split("/")
            return abs(float(numerator) / float(denominator) - float(expected)) < 1e-6
        if "/" in expected and "/" not in predicted:
            numerator, denominator = expected.replace("(", "").replace(")", "").split("/")
            return abs(float(numerator) / float(denominator) - float(predicted)) < 1e-6
    except (ValueError, ZeroDivisionError, OverflowError):
        return False
    return False


def content_parts(message: dict[str, Any]) -> tuple[str, str, bool]:
    content = message.get("content")
    if not isinstance(content, list):
        return "", "", False
    analysis = "".join(
        part.get("thinking", "") for part in content if isinstance(part, dict) and part.get("type") == "thinking"
    )
    final = "".join(part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text")
    has_tool_material = bool(message.get("tool_calls") or message.get("unparsed_tool_calls"))
    return analysis, final, has_tool_material


def load_renderer(model: str, renderer_name: str, current_date: str):
    from tinker_cookbook.renderers import get_renderer
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    tokenizer = get_tokenizer(model)
    renderer = get_renderer(renderer_name, tokenizer, model_name=model)
    if not hasattr(renderer, "current_date"):
        raise ValueError(f"renderer {renderer_name!r} has no current_date field")
    renderer.current_date = current_date
    return tokenizer, renderer


def outputs(run_name: str, arm: str) -> tuple[Path, Path, Path]:
    """raw.jsonl, sft.jsonl and manifest.json for one round and constitution."""
    out = script_dir(STAGE, SCRIPT) / run_name / arm
    return out / "raw.jsonl", out / "sft.jsonl", out / "manifest.json"


def generation_messages(arm: str, prompt: str) -> list[dict[str, str]]:
    """What the model sees while sampling: the constitution and the deliberation instruction as
    the developer message, then the problem. Never what it is trained on."""
    return [{"role": "developer", "content": developer_message(arm)}, {"role": "user", "content": prompt}]


def config_record(args: argparse.Namespace, input_sha256: str) -> dict[str, Any]:
    return {
        "input_sha256": input_sha256,
        "arm": args.arm,
        "spec_sha256": SPEC_SHA256[args.arm],
        "deliberation_instruction": DELIBERATION_INSTRUCTION,
        "generator_version": GENERATOR_VERSION,
        "model": args.model,
        "renderer": args.renderer,
        "current_date": args.current_date,
        "n_samples": args.n_samples,
        "max_tokens": args.max_tokens,
        "max_seq_length": args.max_seq_length,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "run_name": args.run_name,
        "first_sample_index": args.first_sample_index,
    }


def accepted_sft_rows(raw: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for attempt in raw["samples"]:
        if not attempt["accepted"]:
            continue
        sample_index = attempt["sample_index"]
        rows.append(
            {
                "id": f"{raw['id']}:sample:{sample_index}",
                "source_problem_id": raw["id"],
                "sample_index": sample_index,
                "messages": [
                    {"role": "user", "content": raw["prompt"]},
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "thinking", "thinking": attempt["analysis"]},
                            {"type": "text", "text": attempt["final"]},
                        ],
                    },
                ],
                "thinking": "keep",
                "source": "math_cd",
                "arm": raw["arm"],
                "subject": raw["subject"],
                "level": raw["level"],
                "reasoning_tier": raw["renderer"],
            }
        )
    return rows


def load_jsonl_by_id(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            row_id = row.get("id")
            if not isinstance(row_id, str) or not row_id:
                raise ValueError(f"{path}:{line_number}: missing id")
            if row_id in rows:
                raise ValueError(f"{path}:{line_number}: duplicate id {row_id!r}")
            rows[row_id] = row
    return rows


def validate_sft_row(row: dict[str, Any], renderer, max_seq_length: int) -> None:
    from src.training.sft_data import SFTDataConfig, example_to_training_arrays

    arrays = example_to_training_arrays(
        row,
        renderer,
        SFTDataConfig(
            max_seq_length=max_seq_length,
            mask_prompt=True,
            thinking="keep",
        ),
    )
    return_token_ids = renderer.tokenizer.encode("<|return|>", add_special_tokens=False)
    if len(return_token_ids) != 1:
        raise ValueError(f"{row['id']}: <|return|> is not one token")
    if arrays.target_tokens[-1] != return_token_ids[0] or arrays.weights[-1] <= 0:
        raise ValueError(f"{row['id']}: Harmony return token is not a positive-loss target")


async def sample_with_retry(client, prompt, params):
    import tinker

    delay = BACKOFF_BASE_SECONDS
    for attempt in range(MAX_RETRIES):
        try:
            return await client.sample_async(
                prompt=prompt,
                num_samples=1,
                sampling_params=params,
            )
        except (tinker.AuthenticationError, tinker.PermissionDeniedError):
            raise
        except (tinker.RateLimitError, tinker.APIStatusError):
            if attempt == MAX_RETRIES - 1:
                raise
            await asyncio.sleep(delay)
            delay = min(delay * 2, BACKOFF_CAP_SECONDS)
    raise AssertionError("retry loop ended without returning or raising")


async def generate(args: argparse.Namespace, rows: list[dict[str, Any]], renderer) -> None:
    import tinker

    from src.keys import load_project_env
    from src.training.tinker_common import owner_tag

    RAW_OUTPUT, SFT_OUTPUT, MANIFEST_OUTPUT = outputs(args.run_name, args.arm)
    load_project_env()
    if not os.environ.get("TINKER_API_KEY"):
        raise RuntimeError("TINKER_API_KEY is required for --dry-run false")

    config = config_record(args, sha256_file(args.input))
    config_sha256 = hashlib.sha256(canonical_json(config).encode()).hexdigest()
    if MANIFEST_OUTPUT.exists():
        previous = json.loads(MANIFEST_OUTPUT.read_text())
        previous_hash = previous.get("config_sha256")
        if (RAW_OUTPUT.exists() or SFT_OUTPUT.exists()) and previous_hash != config_sha256:
            raise ValueError(f"resume config mismatch: manifest has {previous_hash!r}, current is {config_sha256!r}")

    raw_by_id = load_jsonl_by_id(RAW_OUTPUT)
    sft_by_id = load_jsonl_by_id(SFT_OUTPUT)
    unknown_raw = set(raw_by_id) - {row["id"] for row in rows}
    input_ids = {row["id"] for row in rows}
    unknown_sft = {row_id for row_id, row in sft_by_id.items() if row.get("source_problem_id") not in input_ids}
    if unknown_raw or unknown_sft:
        raise ValueError(f"output contains IDs absent from input: raw={sorted(unknown_raw)} sft={sorted(unknown_sft)}")

    RAW_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    SFT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with SFT_OUTPUT.open("a") as sft_handle:
        for raw in raw_by_id.values():
            for recovered in accepted_sft_rows(raw):
                if recovered["id"] in sft_by_id:
                    continue
                validate_sft_row(recovered, renderer, args.max_seq_length)
                sft_handle.write(canonical_json(recovered) + "\n")
                sft_handle.flush()
                sft_by_id[recovered["id"]] = recovered
                print(f"recovered accepted SFT row {recovered['id']} from raw output", flush=True)

    service = tinker.ServiceClient(user_metadata=owner_tag())
    client = await service.create_sampling_client_async(base_model=args.model)
    semaphore = asyncio.Semaphore(args.max_concurrent)

    async def run_one(index: int, row: dict[str, Any]) -> dict[str, Any]:
        prompt = renderer.build_generation_prompt(generation_messages(args.arm, row["prompt"]))

        async def run_sample(sample_index: int):
            sample_seed = args.seed + index * args.n_samples + (sample_index - args.first_sample_index)
            params = tinker.SamplingParams(
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=sample_seed,
                stop=renderer.get_stop_sequences(),
            )
            async with semaphore:
                response = await sample_with_retry(client, prompt, params)
            return sample_seed, response.sequences[0]

        indices = range(args.first_sample_index, args.first_sample_index + args.n_samples)
        generated = await asyncio.gather(*(run_sample(sample_index) for sample_index in indices))

        samples: list[dict[str, Any]] = []
        accepted_indices: list[int] = []
        for sample_index, (sample_seed, sequence) in zip(indices, generated, strict=True):
            tokens = list(sequence.tokens)
            parsed, termination = renderer.parse_response(tokens)
            analysis, final, has_tool_material = content_parts(parsed)
            is_stop_sequence = bool(getattr(termination, "is_stop_sequence", False))
            truncated = len(tokens) >= args.max_tokens
            correct = check_answer(final, row["reference_answer"])
            accepted = (
                is_stop_sequence
                and bool(analysis.strip())
                and bool(final.strip())
                and not has_tool_material
                and not truncated
                and correct
            )
            samples.append(
                {
                    "sample_index": sample_index,
                    "seed": sample_seed,
                    "tokens": tokens,
                    "output_tokens": len(tokens),
                    "analysis": analysis,
                    "final": final,
                    "parse_termination": getattr(termination, "value", str(termination)),
                    "stop_reason": getattr(sequence.stop_reason, "value", str(sequence.stop_reason)),
                    "has_tool_material": has_tool_material,
                    "truncated": truncated,
                    "correct": correct,
                    "accepted": accepted,
                }
            )
            if accepted:
                accepted_indices.append(sample_index)
        return {
            "id": row["id"],
            "arm": args.arm,
            "subject": row["subject"],
            "level": row["level"],
            "problem": row["problem"],
            "reference_answer": row["reference_answer"],
            "prompt": row["prompt"],
            "renderer": args.renderer,
            "prompt_tokens_per_sample": prompt.length,
            "prompt_tokens": prompt.length * args.n_samples,
            "accepted_sample_indices": accepted_indices,
            "samples": samples,
            "config_sha256": config_sha256,
        }

    pending = [(index, row) for index, row in enumerate(rows) if row["id"] not in raw_by_id]
    tasks = [asyncio.create_task(run_one(index, row)) for index, row in pending]
    with RAW_OUTPUT.open("a") as raw_handle, SFT_OUTPUT.open("a") as sft_handle:
        for task in asyncio.as_completed(tasks):
            raw = await task
            raw_handle.write(canonical_json(raw) + "\n")
            raw_handle.flush()
            accepted_rows = accepted_sft_rows(raw)
            for accepted in accepted_rows:
                validate_sft_row(accepted, renderer, args.max_seq_length)
                sft_handle.write(canonical_json(accepted) + "\n")
                sft_handle.flush()
            print(
                f"{raw['id']}: accepted_samples={len(accepted_rows)}/{len(raw['samples'])} "
                f"outputs={[sample['output_tokens'] for sample in raw['samples']]}",
                flush=True,
            )

    final_raw = load_jsonl_by_id(RAW_OUTPUT)
    final_sft = load_jsonl_by_id(SFT_OUTPUT)
    sample_rows = [sample for row in final_raw.values() for sample in row["samples"]]
    accepted_problem_ids = {row["source_problem_id"] for row in final_sft.values()}
    manifest = {
        "config": config,
        "config_sha256": config_sha256,
        "attempted_problems": len(final_raw),
        "accepted_problems": len(accepted_problem_ids),
        "accepted_samples": len(final_sft),
        "samples": len(sample_rows),
        "clean_parses": sum(sample["parse_termination"] == "stop_sequence" for sample in sample_rows),
        "correct_samples": sum(bool(sample["correct"]) for sample in sample_rows),
        "truncated_samples": sum(bool(sample["truncated"]) for sample in sample_rows),
        "tool_material_samples": sum(bool(sample["has_tool_material"]) for sample in sample_rows),
        "prompt_tokens": sum(row["prompt_tokens"] for row in final_raw.values()),
        "output_tokens": sum(sample["output_tokens"] for sample in sample_rows),
        "raw_output": str(RAW_OUTPUT),
        "raw_sha256": sha256_file(RAW_OUTPUT),
        "sft_output": str(SFT_OUTPUT),
        "sft_sha256": sha256_file(SFT_OUTPUT),
    }
    MANIFEST_OUTPUT.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    if args.model != MODEL:
        raise ValueError(f"generation is frozen to {MODEL!r}, got {args.model!r}")
    if not 0.0 <= args.temperature:
        raise ValueError("--temperature must be >= 0")
    if not 0.0 < args.top_p <= 1.0:
        raise ValueError("--top-p must be in (0, 1]")
    rows = load_rows(args.input)
    _tokenizer, renderer = load_renderer(args.model, args.renderer, args.current_date)
    prompt_lengths = [
        renderer.build_generation_prompt(generation_messages(args.arm, row["prompt"])).length for row in rows
    ]
    prompt_tokens = sum(prompt_lengths) * args.n_samples
    max_output_tokens = len(rows) * args.n_samples * args.max_tokens
    worst_case_cost = (
        prompt_tokens / 1_000_000 * TINKER_PREFILL_USD_PER_MTOK[args.model]
        + max_output_tokens / 1_000_000 * TINKER_SAMPLE_USD_PER_MTOK[args.model]
    )
    dry_manifest = {
        "dry_run": args.dry_run,
        "config": config_record(args, sha256_file(args.input)),
        "rows": len(rows),
        "prompt_tokens": prompt_tokens,
        "max_output_tokens": max_output_tokens,
        "worst_case_sampling_usd": worst_case_cost,
    }
    print(json.dumps(dry_manifest, indent=2, sort_keys=True))
    first = renderer.build_generation_prompt(generation_messages(args.arm, rows[0]["prompt"]))
    print("\nFull rendered generation prompt for the first problem:\n" + renderer.tokenizer.decode(first.to_ints()))
    print("\nFirst six prompts for direct inspection:")
    for row, token_count in list(zip(rows, prompt_lengths, strict=True))[:6]:
        print(
            f"\n[{row['id']}] subject={row['subject']} level={row['level']} "
            f"prompt_tokens={token_count}\n{row['prompt']}"
        )
    if args.dry_run:
        print("\nDRY RUN: no Tinker client was created, nothing written")
        return
    asyncio.run(generate(args, rows, renderer))


if __name__ == "__main__":
    main()
