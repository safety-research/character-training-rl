"""Sample on-policy gpt-oss-20b answers to every question under each constitution in --constitutions.

The developer message is the constitution's full text followed by the deliberation instruction, so
the analysis channel reasons from the constitution; training later strips the developer message
and keeps the question, analysis and final answer. Each (question, constitution) is sampled once
per replicate seed. Every completion is written to a content-keyed cache and an event log the
moment it arrives; `answers.jsonl` is assembled from the cache at the end, so a killed run resumes
without paying twice.

Each round writes under
results/distillation/scripts/s04_generate_answers/<constitutions joined by +>/<run-name>/. Sample
indices start at --first-sample-idx, so a round that resamples the questions no earlier round
resolved adds new trial ids instead of reusing old ones.
"""

from __future__ import annotations

import argparse
import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from character_training.common import (
    DELIBERATION_INSTRUCTION,
    SPEC_SHA256,
    add_constitutions_arg,
    bool_arg,
    checked_constitutions,
    developer_message,
    load_jsonl,
    positive_int,
    run_label,
    sha256_file,
    split_harmony,
    stage_dir,
    write_jsonl,
)
from character_training.distillation import STAGE
from src.costs import TINKER_PREFILL_USD_PER_MTOK, TINKER_SAMPLE_USD_PER_MTOK
from src.evals.base import ResponseCache, TinkerGenerator, force_thinking_prompt, get_cache_key
from src.training.tinker_common import apply_renderer_date, load_renderer, load_tokenizer

SCRIPT = "s04_generate_answers"
MODEL = "openai/gpt-oss-20b"
# A fixed label used in cache keys and data rows; the name has no meaning beyond identifying this version.
CACHE_VERSION = "constqa_20b_harmless_v1"
EXPECTED_COMPLETION_TOKENS = 711
"""Mean completion length at medium reasoning, measured on a small trial run. Used only for the expected-cost line; the
all-max-token ceiling is printed beside it."""


def parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(part) for part in value.split(",") if part.strip())
    if not seeds or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("replicate seeds must be unique comma-separated integers")
    return seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    add_constitutions_arg(parser)
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--questions-sha256", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--renderer", required=True)
    parser.add_argument("--current-date", required=True)
    parser.add_argument("--replicate-seeds", required=True, type=parse_seeds)
    parser.add_argument("--run-name", required=True, help="output subfolder for this sampling round")
    parser.add_argument("--first-sample-idx", required=True, type=int)
    parser.add_argument("--temperature", required=True, type=float)
    parser.add_argument("--top-p", required=True, type=float)
    parser.add_argument("--max-tokens", required=True, type=positive_int)
    parser.add_argument("--max-concurrent", required=True, type=positive_int)
    parser.add_argument("--dry-run", required=True, type=bool_arg)
    args = parser.parse_args()
    if args.model != MODEL:
        parser.error(f"answers are sampled from base {MODEL} only")
    return args


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def messages(question: str, arm: str) -> list[dict]:
    return [{"role": "developer", "content": developer_message(arm)}, {"role": "user", "content": question}]


def trial_id(question_id: str, arm: str, sample_idx: int) -> str:
    return f"{question_id}__{arm}__s{sample_idx}"


def main() -> None:
    args = parse_args()
    ARMS = checked_constitutions(args.constitutions)
    actual_sha = sha256_file(args.questions)
    if actual_sha != args.questions_sha256:
        raise ValueError(f"question bank SHA-256 {actual_sha} != --questions-sha256 {args.questions_sha256}")
    questions = load_jsonl(args.questions)
    ids = [row["question_id"] for row in questions]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate question_id")
    if any(not isinstance(row.get("question"), str) or not row["question"].strip() for row in questions):
        raise ValueError("every question row needs a non-empty question")

    tokenizer = load_tokenizer(args.model)
    renderer = load_renderer(args.model, args.renderer, tokenizer)
    apply_renderer_date(renderer, args.current_date)
    prompts: dict[tuple[str, str], str] = {}
    prompt_tokens: dict[tuple[str, str], int] = {}
    for row in questions:
        for arm in ARMS:
            prompt = force_thinking_prompt(renderer, tokenizer, messages(row["question"], arm), args.renderer)
            prompts[(row["question_id"], arm)] = tokenizer.decode(prompt.to_ints(), skip_special_tokens=False)
            prompt_tokens[(row["question_id"], arm)] = prompt.length

    n_trials = len(questions) * len(ARMS) * len(args.replicate_seeds)
    prefill = sum(prompt_tokens.values()) * len(args.replicate_seeds)
    prefill_rate = TINKER_PREFILL_USD_PER_MTOK[args.model]
    sample_rate = TINKER_SAMPLE_USD_PER_MTOK[args.model]
    expected = (prefill * prefill_rate + n_trials * EXPECTED_COMPLETION_TOKENS * sample_rate) / 1e6
    ceiling = (prefill * prefill_rate + n_trials * args.max_tokens * sample_rate) / 1e6
    for arm in ARMS:
        key = (questions[0]["question_id"], arm)
        print(f"\n{'=' * 88}\nEXACT PROMPT: {key[0]} / {arm} ({prompt_tokens[key]} tokens)\n{'=' * 88}")
        print(prompts[key])
    print(
        f"\n{len(questions):,} questions x {len(ARMS)} constitutions x {len(args.replicate_seeds)} seeds = {n_trials:,} completions"
    )
    print(f"Prefill {prefill:,} tokens at the uncached rate ${prefill_rate}/M")
    print(
        f"Expected ${expected:.2f} at {EXPECTED_COMPLETION_TOKENS} completion tokens each; all-max ceiling ${ceiling:.2f}"
    )
    if args.dry_run:
        print("DRY RUN: no Tinker client created, nothing written.")
        return

    out = stage_dir(STAGE, SCRIPT, run_label(ARMS)) / args.run_name
    out.mkdir(parents=True, exist_ok=True)
    config_path = out / "config.json"
    config = {
        "questions": str(args.questions),
        "questions_sha256": actual_sha,
        "spec_sha256": SPEC_SHA256,
        "deliberation_instruction": DELIBERATION_INSTRUCTION,
        "cli": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError(f"{config_path} holds a different configuration; this output belongs to another run")
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    write_jsonl(
        out / "prompts.jsonl",
        [
            {
                "prompt_id": f"{row['question_id']}__{arm}",
                "question_id": row["question_id"],
                "constitution": arm,
                "rendered_prompt": prompts[(row["question_id"], arm)],
                "prompt_tokens": prompt_tokens[(row["question_id"], arm)],
            }
            for row in questions
            for arm in ARMS
        ],
    )

    cache = ResponseCache(out / ".cache")
    events = out / "answer_events.jsonl"
    lock = threading.Lock()

    def log_event(row: dict) -> None:
        with lock, events.open("a", encoding="utf-8") as sink:
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            sink.flush()

    def key_for(question_id: str, arm: str, sample_idx: int, seed: int) -> str:
        return get_cache_key(
            args.model,
            prompts[(question_id, arm)],
            args.temperature,
            seed,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            renderer=args.renderer,
            current_date=args.current_date,
            constitution_sha256=SPEC_SHA256[arm],
            question_id=question_id,
            sample_idx=sample_idx,
            cache_version=CACHE_VERSION,
        )

    generator = TinkerGenerator(
        model_path=args.model,
        tokenizer_id=args.model,
        renderer_name=args.renderer,
        max_concurrent=args.max_concurrent,
        renderer_current_date=args.current_date,
    )
    generator.load()
    try:
        for sample_idx, seed in enumerate(args.replicate_seeds, start=args.first_sample_idx):
            coordinates = [(row, arm) for row in questions for arm in ARMS]
            keys = [key_for(row["question_id"], arm, sample_idx, seed) for row, arm in coordinates]
            misses = [i for i, key in enumerate(keys) if cache.get(key) is None]
            print(f"seed {seed}: {len(keys) - len(misses):,} cached, {len(misses):,} to sample", flush=True)
            if not misses:
                continue

            def arrived(local_idx: int, response: str, misses=misses, coordinates=coordinates, keys=keys) -> None:
                row, arm = coordinates[misses[local_idx]]
                cache.set(keys[misses[local_idx]], response)
                log_event(
                    {
                        "event": "response_arrived",
                        "recorded_at_utc": now_utc(),
                        "trial_id": trial_id(row["question_id"], arm, sample_idx),
                        "raw": response,
                    }
                )

            fresh = generator.generate_chat(
                [messages(coordinates[i][0]["question"], coordinates[i][1]) for i in misses],
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=seed,
                thinking=True,
                on_response=arrived,
            )
            for local_idx, (raw, stop_reason) in enumerate(zip(fresh, generator.last_stop_reasons)):
                row, arm = coordinates[misses[local_idx]]
                if raw is not None:
                    cache.set(keys[misses[local_idx]], raw, meta={"stop_reason": stop_reason})
                log_event(
                    {
                        "event": "response_finalized",
                        "recorded_at_utc": now_utc(),
                        "trial_id": trial_id(row["question_id"], arm, sample_idx),
                        "stop_reason": stop_reason,
                    }
                )
    finally:
        generator.cleanup()

    answers = []
    for sample_idx, seed in enumerate(args.replicate_seeds, start=args.first_sample_idx):
        for row in questions:
            for arm in ARMS:
                key = key_for(row["question_id"], arm, sample_idx, seed)
                raw = cache.get(key)
                if raw is None:
                    continue
                stop_reason = cache.get_meta(key).get("stop_reason", "unknown_interrupted")
                analysis, final, split_valid = split_harmony(raw)
                answers.append(
                    {
                        "trial_id": trial_id(row["question_id"], arm, sample_idx),
                        "question_id": row["question_id"],
                        "constitution": arm,
                        "sample_idx": sample_idx,
                        "seed": seed,
                        "question": row["question"],
                        "renderer": args.renderer,
                        "current_date": args.current_date,
                        "constitution_sha256": SPEC_SHA256[arm],
                        "questions_sha256": actual_sha,
                        "prompt_tokens": prompt_tokens[(row["question_id"], arm)],
                        "completion_tokens": len(tokenizer.encode(raw)),
                        "stop_reason": stop_reason,
                        "raw": raw,
                        "analysis": analysis,
                        "final": final,
                        "structurally_valid": split_valid and stop_reason == "stop",
                    }
                )
    write_jsonl(out / "answers.jsonl", answers)
    completion = sum(row["completion_tokens"] for row in answers)
    summary = {
        "n_planned": n_trials,
        "n_available": len(answers),
        "n_structurally_valid": sum(row["structurally_valid"] for row in answers),
        "n_truncated": sum(row["stop_reason"] == "length" for row in answers),
        "n_unknown_stop_reason": sum(row["stop_reason"] == "unknown_interrupted" for row in answers),
        "by_arm_valid": {a: sum(r["structurally_valid"] for r in answers if r["constitution"] == a) for a in ARMS},
        "prefill_tokens": prefill,
        "completion_tokens": completion,
        "uncached_cost_usd": (prefill * prefill_rate + completion * sample_rate) / 1e6,
        "answers_sha256": sha256_file(out / "answers.jsonl"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if len(answers) != n_trials:
        raise RuntimeError(f"only {len(answers):,}/{n_trials:,} completions available; rerun to resume")


if __name__ == "__main__":
    main()
