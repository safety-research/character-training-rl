"""Run the StrongREJECT Inspect task with every knob a required flag. `--dry-run` prints the
rubric, the judge template and the first prompt, then stops.

    uv run python -m src.evals.strongreject --model tinker/openai/gpt-oss-20b --checkpoint none \\
        --tokenizer-id openai/gpt-oss-20b --renderer-name gpt_oss_medium_reasoning \\
        --renderer-current-date none --thinking true --chinese false --epochs 3 \\
        --temperature 0.7 --top-p 1.0 --max-tokens 16384 --max-connections 128 \\
        --judge-model claude-sonnet-4-6 --judge-max-connections 100 --prompt-prefix-file none \\
        --log-dir results/<name>/inspect
"""

from __future__ import annotations

import argparse
import sys

from src.evals.common import positive_int
from src.evals.strongreject.eval import JUDGE_TEMPLATE, JUDGES, build_messages, get_judge, load_prompts, read_prefix
from src.evals.strongreject.task import strongreject
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
    p.add_argument("--chinese", required=True, type=_bool, help="true for the Chinese translation of the prompts")
    p.add_argument("--epochs", type=positive_int, required=True, help="samples per prompt")
    p.add_argument("--temperature", type=float, required=True)
    p.add_argument("--top-p", type=float, required=True)
    p.add_argument("--max-tokens", type=positive_int, required=True)
    p.add_argument("--max-connections", type=positive_int, required=True, help="in-flight sampling requests")
    p.add_argument("--judge-model", required=True, help="Anthropic judge model, e.g. claude-sonnet-4-6")
    p.add_argument("--judge-max-connections", type=positive_int, required=True, help="in-flight judge calls")
    p.add_argument(
        "--prompt-prefix-file",
        required=True,
        type=_optional_str,
        help="file whose text goes before every request in the same user turn (a condition), or none",
    )
    p.add_argument("--log-dir", required=True, help="where the Inspect log is written")
    # Deliberately defaulted, not required: the default is the published judge. It is the one
    # exception to "every knob a required flag".
    p.add_argument(
        "--judge",
        default="strongreject",
        choices=sorted(JUDGES),
        help="judge bundle: strongreject (default, the published three-question rubric and score) or "
        "harm (a standalone judge that replaces it and reports only the raw 1-5 delivered_harm mean)",
    )
    p.add_argument("--dry-run", action="store_true", help="print rubric, template and first prompt; no calls")
    return p


def parse_args(argv: list[str]) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: list[str]) -> None:
    from src.keys import inspect_claude_model, load_project_env

    load_project_env()
    a = parse_args(argv)
    prefix = read_prefix(a.prompt_prefix_file)
    if a.dry_run:
        rows = load_prompts(a.chinese)
        print(
            f"===== JUDGE RUBRIC ({a.judge}, system) =====\n{get_judge(a.judge).rubric}\n\n"
            f"===== JUDGE TEMPLATE (user) =====\n{JUDGE_TEMPLATE}"
        )
        print(f"\n===== EXACT PROMPT (prompt 0, {rows[0]['category']}) =====")
        for m in build_messages(rows[0]["prompt"], prefix=prefix):
            print(f"[{m['role']}] {m['content']}")
        print(
            f"\n{len(rows)} prompts x {a.epochs} samples; judge {a.judge_model}\n(dry run: no generation, no judging)"
        )
        return

    from inspect_ai import eval
    from inspect_ai.model import GenerateConfig, get_model

    judge = inspect_claude_model(
        a.judge_model,
        config=GenerateConfig(max_connections=a.judge_max_connections),
    )
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
    # Pin the connection limit on each model rather than on the eval. An eval-level max_connections
    # is merged onto EVERY non-active model, including the judge, and overrides the judge's own limit
    # (inspect_ai Model._resolve_config) — so a large generation cap silently blows the judge pool up
    # to the same number, overwhelming the Anthropic judge. Setting it per model keeps generation fast
    # and the judge capped; the explicit value also keeps both on the static (non-adaptive) path, so
    # `--judge-max-connections` is honoured exactly.
    generator = get_model(a.model, config=GenerateConfig(max_connections=a.max_connections), **model_args)
    task = strongreject(chinese=a.chinese, thinking=a.thinking, epochs=a.epochs, prompt_prefix=prefix, judge=a.judge)
    [log] = eval(
        task,
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
