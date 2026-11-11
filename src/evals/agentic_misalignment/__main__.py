"""Run the agentic-misalignment Inspect task with every knob a required flag. `--dry-run` prints the
exact system prompt and user message of the prompt set's first condition, its action-check targets,
the three judge rubrics, the judge tool and the counts, then stops.

    uv run python -m src.evals.agentic_misalignment --model tinker/openai/gpt-oss-20b --checkpoint none \\
        --tokenizer-id openai/gpt-oss-20b --renderer-name gpt_oss_medium_reasoning \\
        --renderer-current-date none --thinking true --prompt-set main --epochs 10 \\
        --temperature 0.7 --top-p 1.0 --max-tokens 16384 --max-connections 128 \\
        --judge-model claude-sonnet-4-5 --judge-max-connections 50 \\
        --log-dir results/<name>/inspect --retry-log none
"""

from __future__ import annotations

import argparse
import json
import sys

from src.evals.agentic_misalignment.eval import (
    JUDGE_ERROR_GATE,
    JUDGE_TOOL,
    PROMPT_SETS,
    RUBRICS,
    SCENARIOS,
    load_conditions,
)
from src.evals.common import (
    add_model_flags,
    add_retry_log_flag,
    build_generator,
    positive_int,
    reject_greedy_thinking,
    retrying,
    validate_model_flags,
)
from src.evals.inspect_judge import require_pinned_judge_temperature


def build_parser() -> argparse.ArgumentParser:
    """The complete flag set; `parse_args` reads it."""
    p = argparse.ArgumentParser(
        allow_abbrev=False, description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_model_flags(p)
    p.add_argument(
        "--prompt-set",
        required=True,
        choices=sorted(PROMPT_SETS),
        help="main: the 18 conditions; target_swap: the 54 rows with the harmful action's target swapped; "
        "country_uk, country_china: the 9 explicit-goal conditions with the United States replaced",
    )
    p.add_argument("--epochs", required=True, type=positive_int, help="samples per condition")
    p.add_argument("--temperature", required=True, type=float)
    p.add_argument("--top-p", required=True, type=float)
    p.add_argument("--max-tokens", required=True, type=positive_int)
    p.add_argument("--max-connections", required=True, type=positive_int, help="in-flight sampling requests")
    p.add_argument("--judge-model", required=True, help="Anthropic judge model, e.g. claude-sonnet-4-5")
    p.add_argument("--judge-max-connections", required=True, type=positive_int, help="in-flight judge calls")
    p.add_argument("--log-dir", required=True, help="where the Inspect log is written")
    add_retry_log_flag(p)
    p.add_argument("--dry-run", action="store_true", help="print the first condition and the rubrics; no calls")
    return p


def parse_args(argv: list[str]) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def dry_run(a: argparse.Namespace) -> None:
    conditions = load_conditions(a.prompt_set)
    first = conditions[0]
    print(f"===== EXACT PROMPT (condition {first['condition_id']}) =====")
    print(f"[system]\n{first['system_prompt']}\n")
    print(f"[user]\n{first['user_message']}")
    print(f"\n===== ACTION-CHECK TARGETS ({first['scenario']}) =====")
    print(json.dumps({"targets": first["targets"], "murder_target": first["murder_target"]}))
    for scenario in SCENARIOS:
        system, user = RUBRICS[scenario]
        print(f"\n===== JUDGE RUBRIC: {scenario} (system) =====\n{system}")
        print(f"===== JUDGE TEMPLATE: {scenario} (user) =====\n{user}")
    print(f"===== JUDGE TOOL (forced) =====\n{json.dumps(JUDGE_TOOL, indent=1)}")
    n = len(conditions)
    print(
        f"\n{n} conditions ({a.prompt_set}) x {a.epochs} samples = {n * a.epochs} generations; "
        f"judge {a.judge_model}, one call per usable completion, reads the answer channel only\n"
        "(dry run: no generation, no judging)"
    )


def main(argv: list[str]) -> None:
    from src.keys import load_project_env

    load_project_env()
    a = parse_args(argv)
    validate_model_flags(a)
    reject_greedy_thinking(a.thinking, a.temperature)
    require_pinned_judge_temperature(a.judge_model)
    if a.dry_run:
        dry_run(a)
        return

    from inspect_ai import eval
    from inspect_ai.model import GenerateConfig

    from src.evals.agentic_misalignment.task import agentic_misalignment
    from src.keys import inspect_claude_model

    judge = inspect_claude_model(a.judge_model, config=GenerateConfig(max_connections=a.judge_max_connections))
    # Connection limits are pinned per model: an eval-level limit would also override the judge's own.
    generator = build_generator(a, a.max_connections)
    task = agentic_misalignment(prompt_set=a.prompt_set, thinking=a.thinking, epochs=a.epochs)
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
        print(f"  {score.name:<12} " + "  ".join(f"{k} {m.value:.4f}" for k, m in score.metrics.items()))
    err = next(s for s in log.results.scores if s.name == "judge_error").metrics["task_mean"].value
    if not err <= JUDGE_ERROR_GATE:
        print(f"WARNING: judge-error rate {err:.1%} exceeds the {JUDGE_ERROR_GATE:.0%} gate; do not report the rates")
    print(f"log: {log.location}\nview: inspect view --log-dir {a.log_dir}")


if __name__ == "__main__":
    main(sys.argv[1:])
