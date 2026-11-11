"""Run the IFBench Inspect task on one checkpoint; every knob a required flag. Scoring is fully
deterministic (the official strict/loose verifiers), so there is no judge. `--dry-run` prints an
example prompt and its constraints, then stops.

    uv run python -m src.evals.ifbench --model tinker/openai/gpt-oss-20b --checkpoint none \\
        --tokenizer-id openai/gpt-oss-20b --renderer-name gpt_oss_medium_reasoning \\
        --renderer-current-date none --thinking true --epochs 3 --temperature 1.0 --top-p 1.0 \\
        --max-tokens 16384 --max-connections 400 --limit none --data-path none \\
        --retry-log none --log-dir results/<name>/inspect
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.evals.common import (
    add_model_flags,
    add_retry_log_flag,
    build_generator,
    positive_int,
    reject_greedy_thinking,
    retrying,
    validate_model_flags,
)
from src.evals.ifbench.eval import load_prompts, require_nltk_resources
from src.evals.ifbench.task import ifbench
from src.training.tinker_common import _optional_str


def _optional_limit(value: str) -> int | None:
    return None if value == "none" else positive_int(value)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        allow_abbrev=False, description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_model_flags(p)
    p.add_argument("--epochs", required=True, type=positive_int, help="samples per prompt")
    p.add_argument("--temperature", required=True, type=float)
    p.add_argument("--top-p", required=True, type=float)
    p.add_argument("--max-tokens", required=True, type=positive_int)
    p.add_argument("--max-connections", required=True, type=positive_int, help="in-flight sampling requests")
    p.add_argument(
        "--limit", required=True, type=_optional_limit, help="none for every prompt, or first-N by sorted key"
    )
    p.add_argument("--data-path", required=True, type=_optional_str, help="IFBench jsonl, or none for the default")
    p.add_argument("--log-dir", required=True, help="where the Inspect log is written")
    add_retry_log_flag(p)
    p.add_argument("--dry-run", action="store_true", help="print an example prompt and its constraints; no calls")
    return p


def parse_args(argv: list[str]) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: list[str]) -> None:
    from src.keys import load_project_env

    load_project_env()
    a = parse_args(argv)
    validate_model_flags(a)
    reject_greedy_thinking(a.thinking, a.temperature)
    data_path = a.data_path or None  # _optional_str maps the none sentinel to ""

    if a.dry_run:
        rows = load_prompts() if data_path is None else load_prompts(Path(data_path))
        if a.limit is not None:
            rows = sorted(rows, key=lambda r: str(r["key"]))[: a.limit]
        print("===== EXAMPLE PROMPT =====")
        print(rows[0]["prompt"])
        print(f"\nConstraints: {rows[0]['instruction_id_list']}")
        print(
            f"\n{len(rows)} prompts x {a.epochs} samples = {len(rows) * a.epochs} generations (deterministic scoring, no judge)"
        )
        return

    # Before any generation: the verifiers need NLTK corpora, and discovering that after every
    # generation has been paid for is the failure this guard exists to prevent.
    require_nltk_resources()

    from inspect_ai import eval

    generator = build_generator(a, a.max_connections)
    task = ifbench(thinking=a.thinking, epochs=a.epochs, limit=a.limit, data_path=data_path)
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
