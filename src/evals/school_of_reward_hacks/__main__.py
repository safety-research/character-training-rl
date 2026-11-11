"""Run the School of Reward Hacks Inspect task with every knob a required flag. `--dry-run` prints the
row counts, the first user message of each subset and both judge prompts filled for it, then stops.

    uv run python -m src.evals.school_of_reward_hacks --responses policy \\
        --model tinker/openai/gpt-oss-20b --checkpoint none \\
        --tokenizer-id openai/gpt-oss-20b --renderer-name gpt_oss_medium_reasoning \\
        --renderer-current-date none --thinking true --epochs 1 \\
        --temperature 1.0 --top-p 1.0 --max-tokens 16384 --max-connections 128 \\
        --judge-model claude-sonnet-4-6 --judge-max-connections 50 \\
        --log-dir results/<name>/inspect --retry-log none

With `--responses dataset_reward_hack` or `dataset_control`, the stored dataset responses are judged and
nothing is sampled: `--model` must be `mockllm/model` (a placeholder that is never called), `--thinking
false`, and `--checkpoint`, `--tokenizer-id`, `--renderer-name` and `--renderer-current-date` `none`.
"""

from __future__ import annotations

import argparse
import sys

from src.evals.common import add_retry_log_flag, positive_int, retrying
from src.evals.school_of_reward_hacks.eval import (
    POLICY,
    RESPONSES,
    SUBSETS,
    build_judge_user,
    judge_tool,
    load_rows,
    rows_for,
)
from src.evals.school_of_reward_hacks.task import school_of_reward_hacks
from src.evals.tinker_provider import PROVIDER
from src.training.tinker_common import _bool, _optional_str

#: The placeholder model a stored-response run passes; it is never called.
PLACEHOLDER_MODEL = "mockllm/model"


def build_parser() -> argparse.ArgumentParser:
    """The complete flag set; `parse_args` reads it."""
    p = argparse.ArgumentParser(
        allow_abbrev=False, description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--responses", required=True, choices=RESPONSES, help="policy samples, or a stored dataset column")
    p.add_argument("--model", required=True, help=f"Inspect model, e.g. {PROVIDER}/openai/gpt-oss-20b")
    p.add_argument("--checkpoint", required=True, type=_optional_str, help="tinker:// sampler path, or none")
    p.add_argument("--tokenizer-id", required=True, type=_optional_str)
    p.add_argument("--renderer-name", required=True, type=_optional_str)
    p.add_argument(
        "--renderer-current-date", required=True, type=_optional_str, help="YYYY-MM-DD, or none for the fixed default"
    )
    p.add_argument("--thinking", required=True, type=_bool, help="true or false; declared, never inferred")
    p.add_argument("--epochs", type=positive_int, required=True, help="samples per row")
    p.add_argument("--temperature", type=float, required=True)
    p.add_argument("--top-p", type=float, required=True)
    p.add_argument("--max-tokens", type=positive_int, required=True)
    p.add_argument("--max-connections", type=positive_int, required=True, help="in-flight sampling requests")
    p.add_argument("--judge-model", required=True, help="Anthropic judge model, e.g. claude-sonnet-4-6")
    p.add_argument("--judge-max-connections", type=positive_int, required=True, help="in-flight judge calls")
    p.add_argument("--log-dir", required=True, help="where the Inspect log is written")
    add_retry_log_flag(p)
    p.add_argument("--dry-run", action="store_true", help="print the rows and the judge prompts; no calls")
    return p


def parse_args(argv: list[str]) -> argparse.Namespace:
    a = build_parser().parse_args(argv)
    if a.responses == POLICY:
        missing = [f for f in ("tokenizer_id", "renderer_name") if not getattr(a, f)]
        if missing or a.model == PLACEHOLDER_MODEL:
            raise SystemExit(
                f"--responses policy needs a real --model and {', '.join(missing) or 'its renderer flags'}"
            )
    else:
        set_flags = [
            f for f in ("checkpoint", "tokenizer_id", "renderer_name", "renderer_current_date") if getattr(a, f)
        ]
        if a.model != PLACEHOLDER_MODEL or a.thinking or set_flags:
            raise SystemExit(
                f"--responses {a.responses} judges stored responses: --model {PLACEHOLDER_MODEL}, --thinking false, "
                f"and none for {set_flags or 'the model flags'}"
            )
    return a


def main(argv: list[str]) -> None:
    from src.keys import inspect_claude_model, load_project_env

    load_project_env()
    a = parse_args(argv)
    if a.dry_run:
        rows = rows_for(a.responses, load_rows())
        print(f"===== ROWS ({a.responses}; each user message the only user turn, no system prompt) =====")
        for sub in SUBSETS:
            sub_rows = [r for r in rows if r["subset"] == sub]
            print(f"{sub}: {len(sub_rows)} rows")
        for sub in SUBSETS:
            first = next(r for r in rows if r["subset"] == sub) if any(r["subset"] == sub for r in rows) else None
            if first is None:
                continue
            print(f"\n===== FIRST {sub.upper()} ROW ({first['id']}) =====\n{first['user']}")
            print(f"\n===== {sub.upper()} JUDGE PROMPT (that row, completion placeholder) =====")
            print(build_judge_user(sub, first["user"], "<the model's answer channel>"))
            print(f"\n===== {sub.upper()} JUDGE TOOL (forced) =====\n{judge_tool(sub)}")
        print(
            f"\n{len(rows)} rows x {a.epochs} samples; judge {a.judge_model} (reads the answer only)"
            "\n(dry run: no generation, no judging)"
        )
        return

    from inspect_ai import eval
    from inspect_ai.model import GenerateConfig, get_model

    judge = inspect_claude_model(a.judge_model, config=GenerateConfig(max_connections=a.judge_max_connections))
    if a.responses == POLICY:
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
    else:
        generator = get_model(PLACEHOLDER_MODEL)
    task = school_of_reward_hacks(thinking=a.thinking, epochs=a.epochs, responses=a.responses)
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
