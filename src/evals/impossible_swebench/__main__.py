"""Run Impossible-SWEbench with every knob a required flag, so the command line is the record of
the run. `--dry-run` prints the exact system and user prompt of the first selected instance, the
selected ids and the counts, then stops.

    uv run python -m src.evals.impossible_swebench --split conflicting --subset none --seed none \\
        --model tinker/openai/gpt-oss-20b --checkpoint none --tokenizer-id openai/gpt-oss-20b \\
        --renderer-name gpt_oss_medium_reasoning --renderer-current-date none --thinking true \\
        --max-attempts 10 --message-limit 100 --hide-tests false --reset-tests false \\
        --grader-timeout 300 --tool-timeout 60 --custom-prompt paper_strict \\
        --temperature 1.0 --top-p 1.0 --max-tokens 8192 --max-connections 349 --max-sandboxes 349 \\
        --retry-on-error 1 --retry-log none --log-dir results/<name>/inspect

`--subset N --seed S` runs the first N ids of a seeded shuffle of the sorted instance ids; the same
pair names the same instances anywhere. `--subset none --seed none` runs the whole split.
"""

from __future__ import annotations

import argparse
import sys

from src.evals.common import (
    add_model_flags,
    add_retry_log_flag,
    build_generator,
    positive_int,
    retrying,
    seeded_subset,
    validate_model_flags,
)
from src.evals.impossible_swebench.eval import (
    AGENT_TYPE,
    CUSTOM_PROMPTS,
    DATASET,
    NAME,
    SPLITS,
    UNRUNNABLE,
    fork_commit,
    load_split,
    prompt_suffix,
    system_prompt,
    user_prompt,
)
from src.training.tinker_common import _bool


def _optional_int(value: str) -> int | None:
    return None if value == "none" else int(value)


def _optional_positive_int(value: str) -> int | None:
    return None if value == "none" else positive_int(value)


def _non_negative_int(value: str) -> int:
    n = int(value)
    if n < 0:
        raise argparse.ArgumentTypeError(f"must be zero or a positive integer, got {value}")
    return n


def build_parser() -> argparse.ArgumentParser:
    """The complete flag set; `parse_args` reads it."""
    p = argparse.ArgumentParser(
        allow_abbrev=False, description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--split", required=True, choices=SPLITS, help="original (control), oneoff or conflicting")
    p.add_argument("--subset", required=True, type=_optional_positive_int, help="instances to run, or none for all")
    p.add_argument("--seed", required=True, type=_optional_int, help="shuffle seed for --subset, or none")
    add_model_flags(p)
    p.add_argument("--max-attempts", type=positive_int, required=True, help="submissions per instance")
    p.add_argument("--message-limit", type=positive_int, required=True, help="messages per instance before it stops")
    p.add_argument("--hide-tests", required=True, type=_bool, help="true hides the test patch from the agent")
    p.add_argument(
        "--reset-tests",
        required=True,
        type=_bool,
        help="true restores the split's tests before grading (a test edit cannot pay); false grades as the paper does",
    )
    p.add_argument(
        "--grader-timeout",
        type=positive_int,
        required=True,
        help="seconds the grader's test run may take before the instance scores 0 as timed_out (the paper used 300)",
    )
    p.add_argument(
        "--tool-timeout",
        type=positive_int,
        required=True,
        help="seconds an agent bash, python or text_editor call may run before it is stopped (the paper used 60)",
    )
    p.add_argument(
        "--custom-prompt",
        required=True,
        choices=sorted(CUSTOM_PROMPTS),
        help="the scaffold's custom_prompt by name: paper_strict is the paper's main-results instruction, none the bare suffix",
    )
    p.add_argument("--temperature", type=float, required=True)
    p.add_argument("--top-p", type=float, required=True)
    p.add_argument("--max-tokens", type=positive_int, required=True, help="max tokens per assistant turn")
    p.add_argument("--max-connections", type=positive_int, required=True, help="in-flight sampling requests")
    p.add_argument("--max-sandboxes", type=positive_int, required=True, help="Docker sandboxes alive at once")
    p.add_argument("--retry-on-error", type=_non_negative_int, required=True, help="retries of an errored instance")
    p.add_argument("--log-dir", required=True, help="where the Inspect log is written")
    add_retry_log_flag(p)
    p.add_argument("--dry-run", action="store_true", help="print prompts, ids and counts; no calls, no sandbox")
    return p


def parse_args(argv: list[str]) -> argparse.Namespace:
    a = build_parser().parse_args(argv)
    validate_model_flags(a)
    if (a.subset is None) != (a.seed is None):
        raise SystemExit("--subset and --seed go together: both set or both none")
    return a


def selected_ids(a: argparse.Namespace) -> tuple[list[str], list[dict]]:
    rows = [r for r in load_split(a.split) if str(r["instance_id"]) not in UNRUNNABLE]
    ids = seeded_subset([str(r["instance_id"]) for r in rows], a.subset, a.seed)
    by_id = {str(r["instance_id"]): r for r in rows}
    return ids, [by_id[i] for i in ids]


def main(argv: list[str]) -> None:
    from src.keys import load_project_env

    load_project_env()
    a = parse_args(argv)
    ids, rows = selected_ids(a)
    if a.dry_run:
        print(
            f"===== BENCHMARK ===== {NAME} {DATASET} split {a.split}, {AGENT_TYPE} scaffold, "
            f"checkout {fork_commit()[:12]}"
        )
        print(
            f"\n===== SYSTEM PROMPT (instance {ids[0]}) =====\n{system_prompt(rows[0], a.max_attempts, a.hide_tests, prompt_suffix(CUSTOM_PROMPTS[a.custom_prompt]))}"
        )
        print(f"\n===== USER PROMPT (instance {ids[0]}) =====\n{user_prompt(rows[0])}")
        print(f"\n===== INSTANCES ({len(ids)}) =====\n" + "\n".join(ids))
        print(
            f"\n{len(ids)} instances x 1 attempt on {a.model}; up to {a.max_attempts} submissions, "
            f"{a.message_limit} messages each\n(dry run: no generation, no sandbox)"
        )
        return

    from inspect_ai import eval

    from src.evals.impossible_swebench.task import impossible_swebench

    generator = build_generator(a, a.max_connections)
    task = impossible_swebench(
        split=a.split,
        instance_ids=ids,
        max_attempts=a.max_attempts,
        message_limit=a.message_limit,
        hide_tests=a.hide_tests,
        reset_tests=a.reset_tests,
        grader_timeout=a.grader_timeout,
        tool_timeout=a.tool_timeout,
        custom_prompt=a.custom_prompt,
    )
    [log] = eval(
        retrying(task, a.retry_log),
        model=generator,
        temperature=a.temperature,
        top_p=a.top_p,
        max_tokens=a.max_tokens,
        log_dir=a.log_dir,
        max_samples=a.max_connections,
        max_sandboxes=a.max_sandboxes,
        fail_on_error=False,
        retry_on_error=a.retry_on_error,
        display="plain",
    )
    if log.status != "success":
        raise SystemExit(f"eval {log.status}: {log.error}")
    done, total = log.results.completed_samples, log.results.total_samples
    if done != total:
        print(f"WARNING: {total - done} of {total} instances errored and are missing from every denominator")
    for score in log.results.scores:
        if score.scorer != NAME:
            continue
        print(f"  {score.name:<10} " + "  ".join(f"{k} {m.value:.4f}" for k, m in score.metrics.items()))
    print(f"log: {log.location}\nview: inspect view --log-dir {a.log_dir}")


if __name__ == "__main__":
    main(sys.argv[1:])
