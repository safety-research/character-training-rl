"""Run the cued-vs-plain StrongREJECT eval on one checkpoint; every knob a required flag. Aggregates
per-condition harmful-rate and mean delivered-harm from the persisted Inspect log and writes curve.json
beside it. `--dry-run` prints prompt 0 in every condition of the ladder, then stops.

    uv run python -m src.evals.surf_sr --model tinker/openai/gpt-oss-20b --checkpoint none \\
        --tokenizer-id openai/gpt-oss-20b --renderer-name gpt_oss_medium_reasoning \\
        --renderer-current-date none --thinking true --epochs 3 \\
        --temperature 0.7 --top-p 1.0 --max-tokens 16384 --max-connections 128 \\
        --judge-model claude-sonnet-4-6 --judge-max-connections 100 \\
        --prompts-file data/strongreject/strongreject_rewritten.jsonl \\
        --framing-prompt-file none --framing-role none --reasoning-prefill-file none \\
        --round base --log-dir results/<name>/inspect --retry-log none
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from inspect_ai.model import ChatMessageUser

from src.evals.common import add_retry_log_flag, positive_int, retrying
from src.evals.surf_sr.eval import CONDITION_FIELD, CONDITIONS, MATH_FIELD, load_rewritten, per_condition
from src.evals.surf_sr.task import surf_sr
from src.evals.tinker_provider import FRAMING_ROLES, PROVIDER, framing_messages, rendered_prompt
from src.training.tinker_common import _bool, _optional_str


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        allow_abbrev=False, description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", required=True, help=f"Inspect model, e.g. {PROVIDER}/openai/gpt-oss-20b")
    p.add_argument("--checkpoint", required=True, type=_optional_str, help="tinker:// sampler path, or none for base")
    p.add_argument("--tokenizer-id", required=True)
    p.add_argument("--renderer-name", required=True)
    p.add_argument("--renderer-current-date", required=True, type=_optional_str, help="YYYY-MM-DD, or none")
    p.add_argument("--thinking", required=True, type=_bool)
    p.add_argument("--epochs", required=True, type=positive_int, help="samples per (prompt, condition)")
    p.add_argument("--temperature", required=True, type=float)
    p.add_argument("--top-p", required=True, type=float)
    p.add_argument("--max-tokens", required=True, type=positive_int)
    p.add_argument("--max-connections", required=True, type=positive_int)
    p.add_argument("--judge-model", required=True)
    p.add_argument("--judge-max-connections", required=True, type=positive_int)
    p.add_argument(
        "--prompts-file",
        required=True,
        help="rewritten SR prompts: jsonl with prompt/register_x/register_code/register_rand/register_hyp/category",
    )
    p.add_argument(
        "--framing-prompt-file", required=True, type=_optional_str, help="file that opens every conversation, or none"
    )
    p.add_argument(
        "--framing-role",
        required=True,
        choices=FRAMING_ROLES,
        help="none (with no file), system (cookbook system message), or developer (harmony developer message, verbatim)",
    )
    p.add_argument(
        "--reasoning-prefill-file",
        required=True,
        type=_optional_str,
        help="file whose text starts the reasoning channel of every sample, or none",
    )
    p.add_argument("--round", required=True, help="checkpoint round label for curve.json")
    p.add_argument("--log-dir", required=True)
    add_retry_log_flag(p)
    p.add_argument("--dry-run", action="store_true")
    return p


def parse_args(argv: list[str]) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _aggregate(log, round_label, checkpoint):
    """Per-condition harmful-rate (mean sr_score per prompt >= threshold), its judge-error rate and mean
    delivered_harm, from `per_condition`."""
    out = {"round": round_label, "checkpoint": checkpoint}
    for cond, m in per_condition(log.samples).items():
        out[f"{cond}_sr_harmful_rate"] = round(m["harmful_rate"], 4)
        out[f"{cond}_sr_ci"] = [round(m["harmful_rate_lo"], 4), round(m["harmful_rate_hi"], 4)]
        out[f"{cond}_harm_mean"] = round(m["harm_mean"], 4)
        out[f"{cond}_harm_ge3"] = round(m["harm_ge3"], 4)
        out[f"{cond}_n_prompts"] = int(m["n_prompts"])
        out[f"{cond}_usable_rate"] = round(m["usable_rate"], 4)
        out[f"{cond}_sr_judge_error_rate"] = round(m["judge_error_rate"], 4)
        out[f"{cond}_harm_judge_error_rate"] = round(m["harm_judge_error_rate"], 4)
    return out


def main(argv: list[str]) -> None:
    from src.keys import inspect_claude_model, load_project_env

    load_project_env()
    a = parse_args(argv)
    framing = Path(a.framing_prompt_file).read_text() if a.framing_prompt_file else None
    prefill = Path(a.reasoning_prefill_file).read_text() if a.reasoning_prefill_file else None
    if prefill is not None and not a.thinking:
        raise SystemExit("--reasoning-prefill-file needs --thinking true")
    if a.dry_run:
        rows = load_rewritten(Path(a.prompts_file))
        print(f"prompt 0, category {rows[0]['category']}")
        for cond in CONDITIONS:
            print(f"\n===== {cond.upper()} ({CONDITION_FIELD[cond]}) =====\n{rows[0][CONDITION_FIELD[cond]]}")
        print(f"\n===== FRAMING ({a.framing_role}) =====\n{framing if framing is not None else '(none)'}")
        print(f"\n===== REASONING PREFILL =====\n{prefill if prefill is not None else '(none)'}")
        if a.thinking:
            messages = [*framing_messages(framing, a.framing_role), ChatMessageUser(content=rows[0][MATH_FIELD])]
            print("\n===== RENDERED PROMPT (prompt 0, math) =====")
            print(
                rendered_prompt(
                    a.model,
                    tokenizer_id=a.tokenizer_id,
                    renderer_name=a.renderer_name,
                    renderer_current_date=a.renderer_current_date,
                    thinking=a.thinking,
                    messages=messages,
                    reasoning_prefill=prefill,
                )
            )
        print(
            f"\n{len(rows)} prompts x {len(CONDITIONS)} conditions x {a.epochs} samples; "
            f"judges strongreject+harm ({a.judge_model})"
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
    if a.checkpoint:
        model_args["checkpoint"] = a.checkpoint
    if a.renderer_current_date:
        model_args["renderer_current_date"] = a.renderer_current_date
    if prefill is not None:
        model_args["reasoning_prefill"] = prefill
    generator = get_model(a.model, config=GenerateConfig(max_connections=a.max_connections), **model_args)
    task = surf_sr(
        prompts_file=a.prompts_file,
        thinking=a.thinking,
        epochs=a.epochs,
        framing_prompt=framing,
        framing_role=a.framing_role,
    )
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
    curve = _aggregate(log, a.round, a.checkpoint or "base")
    (Path(a.log_dir) / "curve.json").write_text(json.dumps(curve, indent=1))
    print(json.dumps(curve, indent=1))
    print(f"log: {log.location}")


if __name__ == "__main__":
    main(sys.argv[1:])
