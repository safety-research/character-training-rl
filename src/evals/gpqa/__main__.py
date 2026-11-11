"""Run the GPQA Diamond Inspect task on one checkpoint; every knob a required flag. Scoring is fully
deterministic (the answer letter extracted from the answer channel), so there is no judge. `--dry-run`
prints the exact first prompt and the generation count, then stops.

    uv run python -m src.evals.gpqa --model tinker/openai/gpt-oss-20b --checkpoint none \\
        --tokenizer-id openai/gpt-oss-20b --renderer-name gpt_oss_medium_reasoning \\
        --renderer-current-date none --thinking true --epochs 4 --temperature 1.0 --top-p 1.0 \\
        --max-tokens 16384 --max-connections 400 --choice-seed 42 --limit none \\
        --retry-log none --log-dir results/<name>/inspect
"""

from __future__ import annotations

import argparse
import sys

from src.evals.common import (
    add_model_flags,
    add_retry_log_flag,
    build_generator,
    positive_int,
    reject_greedy_thinking,
    retrying,
    validate_model_flags,
)
from src.evals.gpqa import eval as gpqa_eval
from src.evals.gpqa.task import gpqa


def _optional_limit(value: str) -> int | None:
    return None if value == "none" else positive_int(value)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        allow_abbrev=False, description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_model_flags(p)
    p.add_argument("--epochs", required=True, type=positive_int, help="samples per question")
    p.add_argument("--temperature", required=True, type=float)
    p.add_argument("--top-p", required=True, type=float)
    p.add_argument("--max-tokens", required=True, type=positive_int)
    p.add_argument("--max-connections", required=True, type=positive_int, help="in-flight sampling requests")
    p.add_argument("--choice-seed", required=True, type=int, help="keys every question's choice shuffle")
    p.add_argument(
        "--limit", required=True, type=_optional_limit, help="none for every question, or the first N in dataset order"
    )
    p.add_argument("--log-dir", required=True, help="where the Inspect log is written")
    add_retry_log_flag(p)
    p.add_argument(
        "--dry-run", action="store_true", help="print the exact first prompt and the generation count; no calls"
    )
    return p


def parse_args(argv: list[str]) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: list[str]) -> None:
    from src.keys import load_project_env

    load_project_env()
    a = parse_args(argv)
    validate_model_flags(a)
    reject_greedy_thinking(a.thinking, a.temperature)

    if a.dry_run:
        questions = gpqa_eval.build_questions(gpqa_eval.load_rows(), a.choice_seed, a.limit)
        first = questions[0]
        print(f"===== FIRST PROMPT (question {first['index']}, correct letter {first['correct_letter']}) =====")
        print(first["prompt"])
        print(
            f"\n{len(questions)} questions x {a.epochs} samples = {len(questions) * a.epochs} generations "
            "(deterministic scoring, no judge)"
        )
        return

    from inspect_ai import eval

    generator = build_generator(a, a.max_connections)
    task = gpqa(thinking=a.thinking, epochs=a.epochs, choice_seed=a.choice_seed, limit=a.limit)
    [log] = eval(
        retrying(task, a.retry_log),
        model=generator,
        temperature=a.temperature,
        top_p=a.top_p,
        max_tokens=a.max_tokens,
        log_dir=a.log_dir,
        max_samples=a.max_connections,
        display="plain",
    )
    if log.status != "success":
        raise SystemExit(f"eval {log.status}: {log.error}")
    for score in log.results.scores:
        print(f"  {score.name:<10} " + "  ".join(f"{k} {m.value:.4f}" for k, m in score.metrics.items()))
    print(f"log: {log.location}")


if __name__ == "__main__":
    main(sys.argv[1:])
