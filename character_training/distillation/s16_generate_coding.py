"""Context-distil gpt-oss-20b on KodCode coding problems under one constitution: sample each problem
with the constitution and the deliberation instruction in the developer message (the same developer
message as the MATH and constitution-question answers), keep the correct complete completions, and
write training rows that hold only the user's problem and the completion.

Correct means the submitted code passes every test in KodCode's own pytest file. The submission is
the last fenced block of the final channel that compiles and defines the entry function; failing
that, every compiling block of the final channel joined. Drafts inside the analysis channel are
never candidates. The code runs in `src.code_exec.run_code` under the sandbox named by --sandbox:
it is loaded as module `solution`, the test file is executed, then each test function is called.

A completion is accepted when it parses cleanly with non-empty analysis and final channels, carries
no tool material, is not cut off, and is correct; whether the analysis mentions the constitution
plays no part. Every raw completion is appended to `raw.jsonl` as its problem finishes and accepted
rows to `sft.jsonl`; a rerun resumes from what is on disk.

Each round writes under results/distillation/scripts/s16_generate_coding/<run-name>/<constitution>/.
Sample indices start at --first-sample-index, so a later round that resamples the problems no
earlier round resolved adds new sample indices instead of reusing old ones; its --seed must be
chosen so its seeds (seed + row index x n-samples + offset) never meet an earlier round's.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from character_training.common import CONSTITUTIONS, DELIBERATION_INSTRUCTION, SPEC_SHA256, developer_message
from character_training.distillation import STAGE
from character_training.distillation.s15_generate_math import (
    ALLOWED_RENDERERS,
    MODEL,
    canonical_json,
    content_parts,
    finite_float,
    load_jsonl_by_id,
    load_renderer,
    positive_int,
    sample_with_retry,
    sha256_file,
    strict_bool,
    validate_sft_row,
)
from src.code_exec import SANDBOXES, run_code
from src.costs import TINKER_PREFILL_USD_PER_MTOK, TINKER_SAMPLE_USD_PER_MTOK
from src.exp.manifest import script_dir

SCRIPT = "s16_generate_coding"
GENERATOR_VERSION = "kodcode_constitution_in_context_v1"
FENCED_BLOCK = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL)
# Run as the first test statement: expose everything the candidate defined as module `solution`,
# so the test file's `from solution import f` resolves to the candidate's code.
MAKE_SOLUTION_MODULE = (
    "import sys as _s, types as _t\n"
    "_m = _t.ModuleType('solution')\n"
    "_m.__dict__.update({k: v for k, v in dict(globals()).items() if not k.startswith('__')})\n"
    "_s.modules['solution'] = _m"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--constitution", dest="arm", choices=CONSTITUTIONS, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--renderer", choices=ALLOWED_RENDERERS, required=True)
    p.add_argument("--current-date", required=True)
    p.add_argument("--n-samples", type=positive_int, required=True)
    p.add_argument("--max-tokens", type=positive_int, required=True)
    p.add_argument("--max-seq-length", type=positive_int, required=True)
    p.add_argument("--temperature", type=finite_float, required=True)
    p.add_argument("--top-p", type=finite_float, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--run-name", required=True, help="output subfolder for this sampling round")
    p.add_argument("--first-sample-index", type=int, required=True)
    p.add_argument("--max-concurrent", type=positive_int, required=True)
    p.add_argument("--exec-concurrent", type=positive_int, required=True, help="sandboxed test runs at once")
    p.add_argument("--sandbox", choices=SANDBOXES, required=True)
    p.add_argument("--dry-run", type=strict_bool, required=True)
    return p.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows, seen = [], set()
    for n, line in enumerate(path.open(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        for f in ("id", "entry_point", "prompt", "test", "difficulty"):
            if not isinstance(row.get(f), str) or not row[f].strip():
                raise ValueError(f"{path}:{n}: {f} must be a non-empty string")
        if row["id"] in seen:
            raise ValueError(f"{path}:{n}: duplicate id {row['id']!r}")
        seen.add(row["id"])
        rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def _compiles(src: str) -> bool:
    try:
        compile(src, "<candidate>", "exec")
        return True
    except (SyntaxError, ValueError):
        return False


def _defines(src: str, name: str) -> bool:
    tree = ast.parse(src)
    return any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name for n in tree.body)


def extract_submission(final: str, entry: str) -> str | None:
    """The last compiling fenced block of the final channel that defines `entry` at top level;
    failing that, every compiling block joined; None when nothing compiles."""
    blocks = [b.strip() for b in FENCED_BLOCK.findall(final) if b.strip()]
    compiling = [b for b in blocks if _compiles(b)]
    for b in reversed(compiling):
        if _defines(b, entry):
            return b
    return "\n\n".join(compiling) if compiling else None


def test_statements(test_src: str) -> list[str]:
    """What run_code executes after the candidate: make `solution`, run the test file, call each test."""
    names = [n.name for n in ast.parse(test_src).body if isinstance(n, ast.FunctionDef) and n.name.startswith("test")]
    return [MAKE_SOLUTION_MODULE, test_src] + [f"{n}()" for n in names]


def _site_setup() -> str:
    """run_code's subprocess runs the base interpreter, outside this project's virtualenv; KodCode tests
    import pytest (pytest.raises) and some solutions use numpy. Put the virtualenv's packages on the
    path. Problems needing a package the env lacks fail the grader self-test and are dropped. Matplotlib is
    forced onto its non-interactive backend so plotting code never opens a window on the host, and each run
    starts in a fresh temporary directory so files the code writes never land in the repo."""
    import sysconfig

    return (
        f"import os, sys, tempfile\nos.environ['MPLBACKEND'] = 'Agg'\n"
        f"os.chdir(tempfile.mkdtemp(prefix='kodcode_run_'))\n"
        f"sys.path.insert(0, {sysconfig.get_paths()['purelib']!r})"
    )


SITE_SETUP = _site_setup()


def check(final: str, row: dict[str, Any], sandbox: str) -> dict[str, Any]:
    code = extract_submission(final, row["entry_point"])
    if code is None:
        return {
            "correct": False,
            "code_found": False,
            "compiled": False,
            "timeout": False,
            "tests_passed": 0,
            "tests_total": 0,
        }
    r = run_code(SITE_SETUP, code, test_statements(row["test"]), sandbox=sandbox)
    n_tests = len(test_statements(row["test"])) - 2
    return {
        "correct": r.all_pass,
        "code_found": True,
        "compiled": r.compiled,
        "timeout": r.timeout,
        "tests_passed": sum(r.results[2:]),
        "tests_total": n_tests,
    }


def outputs(run_name: str, arm: str) -> tuple[Path, Path, Path]:
    out = script_dir(STAGE, SCRIPT) / run_name / arm
    return out / "raw.jsonl", out / "sft.jsonl", out / "manifest.json"


def generation_messages(arm: str, prompt: str) -> list[dict[str, str]]:
    """What the model sees while sampling. Never what it is trained on."""
    return [{"role": "developer", "content": developer_message(arm)}, {"role": "user", "content": prompt}]


def config_record(a: argparse.Namespace, input_sha256: str) -> dict[str, Any]:
    return {
        "input_sha256": input_sha256,
        "arm": a.arm,
        "spec_sha256": SPEC_SHA256[a.arm],
        "deliberation_instruction": DELIBERATION_INSTRUCTION,
        "generator_version": GENERATOR_VERSION,
        "model": a.model,
        "renderer": a.renderer,
        "current_date": a.current_date,
        "n_samples": a.n_samples,
        "max_tokens": a.max_tokens,
        "max_seq_length": a.max_seq_length,
        "temperature": a.temperature,
        "top_p": a.top_p,
        "seed": a.seed,
        "sandbox": a.sandbox,
        "run_name": a.run_name,
        "first_sample_index": a.first_sample_index,
    }


def accepted_sft_rows(raw: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "id": f"{raw['id']}:sample:{s['sample_index']}",
            "source_problem_id": raw["id"],
            "sample_index": s["sample_index"],
            "messages": [
                {"role": "user", "content": raw["prompt"]},
                {
                    "role": "assistant",
                    "content": [{"type": "thinking", "thinking": s["analysis"]}, {"type": "text", "text": s["final"]}],
                },
            ],
            "thinking": "keep",
            "source": "coding_cd",
            "arm": raw["arm"],
            "difficulty": raw["difficulty"],
            "reasoning_tier": raw["renderer"],
        }
        for s in raw["samples"]
        if s["accepted"]
    ]


async def generate(a: argparse.Namespace, rows: list[dict[str, Any]], renderer) -> None:
    import tinker

    from src.keys import load_project_env
    from src.training.tinker_common import owner_tag

    RAW, SFT, MANIFEST = outputs(a.run_name, a.arm)
    load_project_env()
    if not os.environ.get("TINKER_API_KEY"):
        raise RuntimeError("TINKER_API_KEY is required for --dry-run false")
    config = config_record(a, sha256_file(a.input))
    config_sha = hashlib.sha256(canonical_json(config).encode()).hexdigest()
    if MANIFEST.exists() and (RAW.exists() or SFT.exists()):
        prev = json.loads(MANIFEST.read_text()).get("config_sha256")
        if prev != config_sha:
            raise ValueError(f"resume config mismatch: manifest has {prev!r}, current is {config_sha!r}")
    raw_by_id = load_jsonl_by_id(RAW)
    if set(raw_by_id) - {r["id"] for r in rows}:
        raise ValueError("raw output contains ids absent from input")
    RAW.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(
        json.dumps({"config": config, "config_sha256": config_sha, "status": "running"}, indent=2) + "\n"
    )

    service = tinker.ServiceClient(user_metadata=owner_tag())
    client = await service.create_sampling_client_async(base_model=a.model)
    sample_sem = asyncio.Semaphore(a.max_concurrent)
    exec_sem = asyncio.Semaphore(a.exec_concurrent)

    async def run_one(index: int, row: dict[str, Any]) -> dict[str, Any]:
        prompt = renderer.build_generation_prompt(generation_messages(a.arm, row["prompt"]))

        async def run_sample(k: int):
            seed = a.seed + index * a.n_samples + (k - a.first_sample_index)
            params = tinker.SamplingParams(
                max_tokens=a.max_tokens,
                temperature=a.temperature,
                top_p=a.top_p,
                seed=seed,
                stop=renderer.get_stop_sequences(),
            )
            async with sample_sem:
                resp = await sample_with_retry(client, prompt, params)
            return seed, resp.sequences[0]

        indices = range(a.first_sample_index, a.first_sample_index + a.n_samples)
        generated = await asyncio.gather(*(run_sample(k) for k in indices))
        samples = []
        for k, (seed, seq) in zip(indices, generated, strict=True):
            tokens = list(seq.tokens)
            parsed, termination = renderer.parse_response(tokens)
            analysis, final, tool = content_parts(parsed)
            clean = bool(getattr(termination, "is_stop_sequence", False))
            truncated = len(tokens) >= a.max_tokens
            usable = clean and bool(analysis.strip()) and bool(final.strip()) and not tool and not truncated
            if usable:
                async with exec_sem:
                    result = await asyncio.to_thread(check, final, row, a.sandbox)
            else:
                result = {
                    "correct": False,
                    "code_found": False,
                    "compiled": False,
                    "timeout": False,
                    "tests_passed": 0,
                    "tests_total": 0,
                }
            samples.append(
                {
                    "sample_index": k,
                    "seed": seed,
                    "tokens": tokens,
                    "output_tokens": len(tokens),
                    "analysis": analysis,
                    "final": final,
                    "parse_termination": getattr(termination, "value", str(termination)),
                    "stop_reason": getattr(seq.stop_reason, "value", str(seq.stop_reason)),
                    "has_tool_material": tool,
                    "truncated": truncated,
                    **result,
                    "accepted": usable and result["correct"],
                }
            )
        return {
            "id": row["id"],
            "arm": a.arm,
            "difficulty": row["difficulty"],
            "entry_point": row["entry_point"],
            "prompt": row["prompt"],
            "renderer": a.renderer,
            "prompt_tokens_per_sample": prompt.length,
            "prompt_tokens": prompt.length * a.n_samples,
            "samples": samples,
            "config_sha256": config_sha,
        }

    pending = [(i, r) for i, r in enumerate(rows) if r["id"] not in raw_by_id]
    tasks = [asyncio.create_task(run_one(i, r)) for i, r in pending]
    with RAW.open("a") as raw_h, SFT.open("a") as sft_h:
        for t in asyncio.as_completed(tasks):
            raw = await t
            raw_h.write(canonical_json(raw) + "\n")
            raw_h.flush()
            acc = accepted_sft_rows(raw)
            for row in acc:
                validate_sft_row(row, renderer, a.max_seq_length)
                sft_h.write(canonical_json(row) + "\n")
                sft_h.flush()
            print(
                f"{raw['id']} [{raw['difficulty']}]: accepted {len(acc)}/{len(raw['samples'])} "
                f"tests {[f'{s["tests_passed"]}/{s["tests_total"]}' for s in raw['samples']]}",
                flush=True,
            )

    final_raw = load_jsonl_by_id(RAW)
    sm = [s for r in final_raw.values() for s in r["samples"]]
    by_diff = {d: {"problems": 0, "accepted_samples": 0} for d in ("easy", "medium", "hard")}
    for r in final_raw.values():
        by_diff[r["difficulty"]]["problems"] += 1
        by_diff[r["difficulty"]]["accepted_samples"] += sum(s["accepted"] for s in r["samples"])
    manifest = {
        "config": config,
        "config_sha256": config_sha,
        "status": "complete",
        "attempted_problems": len(final_raw),
        "accepted_problems": sum(any(s["accepted"] for s in r["samples"]) for r in final_raw.values()),
        "accepted_samples": sum(s["accepted"] for s in sm),
        "samples": len(sm),
        "truncated_samples": sum(s["truncated"] for s in sm),
        "no_code_samples": sum(not s["code_found"] for s in sm),
        "timeout_samples": sum(s["timeout"] for s in sm),
        "by_difficulty": by_diff,
        "prompt_tokens": sum(r["prompt_tokens"] for r in final_raw.values()),
        "output_tokens": sum(s["output_tokens"] for s in sm),
        "raw_sha256": sha256_file(RAW),
        "sft_sha256": sha256_file(SFT),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "config"}, indent=2))


def main() -> None:
    a = parse_args()
    if a.model != MODEL:
        raise ValueError(f"generation is frozen to {MODEL!r}, got {a.model!r}")
    rows = load_rows(a.input)
    _tok, renderer = load_renderer(a.model, a.renderer, a.current_date)
    lengths = [renderer.build_generation_prompt(generation_messages(a.arm, r["prompt"])).length for r in rows]
    prompt_tokens = sum(lengths) * a.n_samples
    max_out = len(rows) * a.n_samples * a.max_tokens
    worst = (
        prompt_tokens / 1e6 * TINKER_PREFILL_USD_PER_MTOK[a.model] + max_out / 1e6 * TINKER_SAMPLE_USD_PER_MTOK[a.model]
    )
    print(
        json.dumps(
            {
                "dry_run": a.dry_run,
                "config": config_record(a, sha256_file(a.input)),
                "rows": len(rows),
                "prompt_tokens": prompt_tokens,
                "max_output_tokens": max_out,
                "worst_case_sampling_usd": worst,
            },
            indent=2,
            sort_keys=True,
        )
    )
    first = renderer.build_generation_prompt(generation_messages(a.arm, rows[0]["prompt"]))
    print("\nFull rendered generation prompt for the first problem:\n" + renderer.tokenizer.decode(first.to_ints()))
    print("\nFirst three problems:")
    for r, n in list(zip(rows, lengths))[:3]:
        print(f"\n[{r['id']}] {r['difficulty']} prompt_tokens={n}\n{r['prompt']}")
    if a.dry_run:
        print("\nDRY RUN: no Tinker client was created, nothing written")
        return
    asyncio.run(generate(a, rows, renderer))


if __name__ == "__main__":
    main()
