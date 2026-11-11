"""Run the categorical emergent-misalignment Inspect task with every knob a required flag.
`--dry-run` prints the eight questions and the judge prompt filled for question 0, then stops.

    uv run python -m src.evals.em_categorical --model tinker/openai/gpt-oss-20b --checkpoint none \\
        --tokenizer-id openai/gpt-oss-20b --renderer-name gpt_oss_medium_reasoning \\
        --renderer-current-date none --thinking true --epochs 100 \\
        --temperature 1.0 --top-p 1.0 --max-tokens 16384 --max-connections 128 \\
        --judge-model claude-sonnet-4-6 --judge-max-connections 50 \\
        --log-dir results/<name>/inspect --retry-log none
"""

from __future__ import annotations

import argparse
import sys

from src.evals.common import add_retry_log_flag, positive_int, retrying
from src.evals.em_categorical.eval import JUDGE_TOOL, QUESTIONS, build_judge_user
from src.evals.em_categorical.task import em_categorical
from src.evals.tinker_provider import PROVIDER
from src.training.tinker_common import _bool, _optional_str


def build_parser() -> argparse.ArgumentParser:
    """The complete flag set; `parse_args` reads it."""
    p = argparse.ArgumentParser(
        allow_abbrev=False, description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", required=True, help=f"Inspect model, e.g. {PROVIDER}/openai/gpt-oss-20b")
    p.add_argument("--checkpoint", required=True, type=_optional_str, help="tinker:// sampler path, or none")
    p.add_argument("--tokenizer-id", required=True)
    p.add_argument("--renderer-name", required=True)
    p.add_argument(
        "--renderer-current-date", required=True, type=_optional_str, help="YYYY-MM-DD, or none for the fixed default"
    )
    p.add_argument("--thinking", required=True, type=_bool, help="true or false; declared, never inferred")
    p.add_argument("--epochs", type=positive_int, required=True, help="samples per question")
    p.add_argument("--temperature", type=float, required=True)
    p.add_argument("--top-p", type=float, required=True)
    p.add_argument("--max-tokens", type=positive_int, required=True)
    p.add_argument("--max-connections", type=positive_int, required=True, help="in-flight sampling requests")
    p.add_argument("--judge-model", required=True, help="Anthropic judge model, e.g. claude-sonnet-4-6")
    p.add_argument("--judge-max-connections", type=positive_int, required=True, help="in-flight judge calls")
    p.add_argument("--log-dir", required=True, help="where the Inspect log is written")
    add_retry_log_flag(p)
    p.add_argument("--dry-run", action="store_true", help="print the questions and the judge prompt; no calls")
    return p


def parse_args(argv: list[str]) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: list[str]) -> None:
    from src.keys import inspect_claude_model, load_project_env

    load_project_env()
    a = parse_args(argv)
    if a.dry_run:
        print("===== QUESTIONS (each the only user turn, no system prompt) =====")
        for qid, text in QUESTIONS:
            print(f"[{qid}] {text}")
        print("\n===== JUDGE PROMPT (user turn; question 0, response placeholder) =====")
        print(build_judge_user(QUESTIONS[0][1], "<the model's response>"))
        print(f"\n===== JUDGE TOOL (forced) =====\n{JUDGE_TOOL}")
        print(
            f"\n{len(QUESTIONS)} questions x {a.epochs} samples; judge {a.judge_model} "
            f"(reads reasoning and answer)\n(dry run: no generation, no judging)"
        )
        return

    from inspect_ai import eval
    from inspect_ai.model import GenerateConfig, get_model

    judge = inspect_claude_model(a.judge_model, config=GenerateConfig(max_connections=a.judge_max_connections))
    model_args = {
        "tokenizer_id": a.tokenizer_id,
        "renderer_name": a.renderer_name,
        "thinking": a.thinking,
        "max_connections": a.max_connections,
    }
    if a.checkpoint:  # the none sentinel parses to ""
        model_args["checkpoint"] = a.checkpoint
    if a.renderer_current_date:
        model_args["renderer_current_date"] = a.renderer_current_date
    # Connection limits are pinned per model: an eval-level max_connections would also override the
    # judge's own limit.
    generator = get_model(a.model, config=GenerateConfig(max_connections=a.max_connections), **model_args)
    task = em_categorical(thinking=a.thinking, epochs=a.epochs)
    [log] = eval(
        retrying(task, a.retry_log),
        model=generator,
        model_roles={"judge": judge},
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
        print(f"  {score.name:<16} " + "  ".join(f"{k} {m.value:.4f}" for k, m in score.metrics.items()))
    print(f"log: {log.location}\nview: inspect view --log-dir {a.log_dir}")


if __name__ == "__main__":
    main(sys.argv[1:])
