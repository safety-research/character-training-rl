"""Run the two-turn preference Inspect task with every knob a required flag. `--dry-run` prints every
first-turn wording, every follow-up, both rubrics and both judge templates resolved on the first item,
and the item counts, then stops.

    uv run python -m src.evals.preference_v3 --model tinker/openai/gpt-oss-20b --checkpoint none \\
        --tokenizer-id openai/gpt-oss-20b --renderer-name gpt_oss_medium_reasoning \\
        --renderer-current-date 2026-08-20 --thinking true \\
        --subsets cheese,american_foreign,american_peers,legacy_modern --pair-set v3 \\
        --question-phrasings 0,1,2,3,4 --followup-phrasings 0,1,2,3,4 \\
        --temperature 1.0 --top-p 1.0 --seed 0 --max-tokens 20480 --max-connections 256 \\
        --judge-model claude-sonnet-4-6 --judge-max-connections 100 --framing-prompt-file none \\
        --framing-role none --question-file none --retry-log none --log-dir results/<name>/inspect
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from inspect_ai.model import ChatMessageUser

from src.evals.common import _optional_path, add_retry_log_flag, positive_int, retrying
from src.evals.preference_v3.eval import (
    CHOICE_JUDGE_RUBRIC,
    CHOICE_JUDGE_TEMPLATE,
    FIRST_TURN_PHRASINGS,
    JUDGE_ERROR_GATE,
    PAIR_SETS,
    SECOND_TURN_PHRASINGS,
    SUBSETS,
    WHY_JUDGE_TEMPLATE,
    load_bank,
    replace_questions,
    select,
    why_rubric,
)
from src.evals.tinker_provider import FRAMING_ROLES, PROVIDER, framing_messages, rendered_prompt
from src.jsonl import sha256_file
from src.training.tinker_common import _bool, _optional_str


def _names(text: str) -> list[str]:
    names = [t for t in text.split(",") if t]
    unknown = sorted(set(names) - set(SUBSETS))
    if not names or unknown or len(set(names)) != len(names):
        raise argparse.ArgumentTypeError(f"{text!r}: expected distinct names from {SUBSETS}")
    return names


def _indices(text: str) -> list[int]:
    try:
        out = [int(t) for t in text.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r}: expected comma-separated integers") from None
    if len(set(out)) != len(out):
        raise argparse.ArgumentTypeError(f"{text!r}: repeated index")
    return out


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
    p.add_argument("--subsets", required=True, type=_names, help=f"comma-separated, from {','.join(SUBSETS)}")
    p.add_argument("--pair-set", required=True, choices=PAIR_SETS, help="v2: the V2 pairs; v3: plus the new pairs")
    p.add_argument(
        "--question-phrasings", required=True, type=_indices, help="first-turn phrasing indices, e.g. 0,1,2,3,4"
    )
    p.add_argument(
        "--followup-phrasings", required=True, type=_indices, help="follow-up phrasing indices, e.g. 0,1,2,3,4"
    )
    p.add_argument("--temperature", type=float, required=True)
    p.add_argument("--top-p", type=float, required=True)
    p.add_argument("--seed", type=int, required=True, help="sampling seed passed to every call")
    p.add_argument("--max-tokens", type=positive_int, required=True)
    p.add_argument("--max-connections", type=positive_int, required=True, help="in-flight sampling requests")
    p.add_argument("--judge-model", required=True, help="Anthropic judge model, e.g. claude-sonnet-4-6")
    p.add_argument("--judge-max-connections", type=positive_int, required=True, help="in-flight judge calls")
    p.add_argument(
        "--framing-prompt-file", required=True, type=_optional_str, help="file that opens the conversation, or none"
    )
    p.add_argument(
        "--framing-role",
        required=True,
        choices=FRAMING_ROLES,
        help="none (with no file), system (cookbook system message), or developer (harmony developer message, verbatim)",
    )
    p.add_argument(
        "--question-file",
        required=True,
        type=_optional_path,
        help="JSONL of {item_id, question} replacing each selected item's first-turn question, or none for the bank's",
    )
    p.add_argument("--log-dir", required=True, help="where the Inspect log is written")
    add_retry_log_flag(p)
    p.add_argument("--dry-run", action="store_true", help="print every wording, rubric and template; no calls")
    return p


def parse_args(argv: list[str]) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: list[str]) -> None:
    from src.keys import inspect_claude_model, load_project_env

    load_project_env()
    a = parse_args(argv)
    framing = Path(a.framing_prompt_file).read_text() if a.framing_prompt_file else None
    rows = select(load_bank(), tuple(a.subsets), a.pair_set, tuple(a.question_phrasings))
    if a.question_file is not None:
        bank_questions = {r["item_id"]: r["question"] for r in rows}
        rows = replace_questions(rows, Path(a.question_file))
    n_follow = len(rows) * len(a.followup_phrasings)
    if a.dry_run:
        first = rows[0]
        print("===== FIRST-TURN PHRASINGS (company, cheese; 0 is the V2 question) =====")
        print(json.dumps(FIRST_TURN_PHRASINGS, indent=1, ensure_ascii=False))
        print(f"\n===== FOLLOW-UPS =====\n{json.dumps(SECOND_TURN_PHRASINGS, indent=1)}")
        print(f"\n===== FRAMING ({a.framing_role}) =====\n{framing if framing is not None else '(none)'}")
        if a.question_file is not None:
            n_diff = sum(r["question"] != bank_questions[r["item_id"]] for r in rows)
            print(f"\n===== QUESTION FILE =====\n{a.question_file} (sha256 {sha256_file(a.question_file)})")
            print(f"{n_diff} of {len(rows)} first-turn questions differ from the bank's")
            print(f"first item {first['item_id']}:\n{first['question']}")
        if a.thinking:
            opening = [*framing_messages(framing, a.framing_role), ChatMessageUser(content=first["question"])]
            print("\n===== RENDERED FIRST-TURN PROMPT (first item) =====")
            print(
                rendered_prompt(
                    a.model,
                    tokenizer_id=a.tokenizer_id,
                    renderer_name=a.renderer_name,
                    renderer_current_date=a.renderer_current_date,
                    thinking=a.thinking,
                    messages=opening,
                    reasoning_prefill=None,
                )
            )
        print(f"\n===== CHOICE JUDGE RUBRIC (system) =====\n{CHOICE_JUDGE_RUBRIC}")
        print("===== CHOICE JUDGE USER (first item) =====")
        print(
            CHOICE_JUDGE_TEMPLATE.format(
                question=first["question"],
                option_1=first["option_a"],
                option_2=first["option_b"],
                first_final="<first answer>",
            )
        )
        for j in a.followup_phrasings:
            followup = SECOND_TURN_PHRASINGS[j]
            print(f"===== WHY JUDGE RUBRIC, follow-up {j} (system) =====\n{why_rubric(followup)}")
            print(
                WHY_JUDGE_TEMPLATE.format(
                    question=first["question"],
                    first_final="<first answer>",
                    followup=followup,
                    followup_final="<answer>",
                )
            )
        print(
            f"{len(rows)} first turns over {len({r['cluster_id'] for r in rows})} pairs; up to {n_follow} follow-ups; "
            f"judge {a.judge_model}\n(dry run: no generation, no judging)"
        )
        return

    from inspect_ai import eval
    from inspect_ai.model import GenerateConfig, get_model

    from src.evals.preference_v3.task import preference_v3

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
    # Connection limits are pinned per model: an eval-level limit would be merged onto the judge too.
    generator = get_model(a.model, config=GenerateConfig(max_connections=a.max_connections), **model_args)
    task = preference_v3(
        subsets=a.subsets,
        pair_set=a.pair_set,
        question_phrasings=a.question_phrasings,
        followup_phrasings=a.followup_phrasings,
        thinking=a.thinking,
        framing_prompt=framing,
        framing_role=a.framing_role,
        question_file=a.question_file,
    )
    [log] = eval(
        retrying(task, a.retry_log),
        model=generator,
        model_roles={"judge": judge},
        temperature=a.temperature,
        top_p=a.top_p,
        seed=a.seed,
        max_tokens=a.max_tokens,
        log_dir=a.log_dir,
        max_samples=a.max_connections,
        display="plain",
    )
    if log.status != "success":
        raise SystemExit(f"eval {log.status}: {log.error}")
    metrics = {k: m.value for score in log.results.scores for k, m in score.metrics.items()}
    for k, v in metrics.items():
        print(f"  {k:<44} {v:.4f}")
    for family in ("choice", "why"):
        rate = metrics[f"{family}_judge_error_rate"]
        if not rate <= JUDGE_ERROR_GATE:
            print(
                f"WARNING: {family} judge-error rate {rate:.1%} exceeds the {JUDGE_ERROR_GATE:.0%} gate; its rates are NaN"
            )
    print(f"log: {log.location}\nview: inspect view --log-dir {a.log_dir}")


if __name__ == "__main__":
    main(sys.argv[1:])
