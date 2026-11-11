"""Run Impossible-LiveCodeBench with every knob a required flag, so the command line is the record of
the run. `--dry-run` prints the exact system prompt, user prompt and (tools scaffold) starting files of
the first selected problem, the selected ids and the counts, then stops.

    uv run python -m src.evals.impossible_livecodebench --split conflicting --subset none --seed none \\
        --model tinker/openai/gpt-oss-20b --checkpoint none --tokenizer-id openai/gpt-oss-20b \\
        --renderer-name gpt_oss_medium_reasoning --renderer-current-date none --thinking true \\
        --agent-type minimal --instruction paper_strict \\
        --max-attempts 10 --message-limit 50 --allow-test-modifications true \\
        --temperature 1.0 --top-p 1.0 --max-tokens 8192 --max-connections 103 --max-sandboxes 103 \\
        --retry-on-error 1 --retry-log none --log-dir results/<name>/inspect

`--subset N --seed S` runs the first N ids of a seeded shuffle of the sorted problem ids; the same
pair names the same problems anywhere. `--subset none --seed none` runs the whole split.
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
from src.evals.impossible_livecodebench.eval import (
    AGENT_TYPES,
    DATASET,
    INSTRUCTIONS,
    NAME,
    SPLITS,
    load_split,
    scaffold_start,
)
from src.evals.impossible_swebench.eval import fork_commit
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
    p.add_argument("--subset", required=True, type=_optional_positive_int, help="problems to run, or none for all")
    p.add_argument("--seed", required=True, type=_optional_int, help="shuffle seed for --subset, or none")
    add_model_flags(p)
    p.add_argument("--agent-type", required=True, choices=AGENT_TYPES, help="the benchmark's scaffold")
    p.add_argument(
        "--instruction",
        required=True,
        choices=INSTRUCTIONS,
        help="the opening instruction: paper_strict is the paper's main-results prompt D, task_default the task's own",
    )
    p.add_argument("--max-attempts", type=positive_int, required=True, help="submissions per problem")
    p.add_argument("--message-limit", type=positive_int, required=True, help="messages per problem before it stops")
    p.add_argument(
        "--allow-test-modifications",
        required=True,
        type=_bool,
        help="the benchmark's flag; on the tools scaffold it changes only the instruction sentence about "
        "the tests (true: 'DO NOT MODIFY THE TESTS', false: modifications 'will be reverted') and, on the minimal "
        "scaffold, whether its in-loop check runs the model's code after the tests; both scorers grade against "
        "the split's tests either way",
    )
    p.add_argument("--temperature", type=float, required=True)
    p.add_argument("--top-p", type=float, required=True)
    p.add_argument("--max-tokens", type=positive_int, required=True, help="max tokens per assistant turn")
    p.add_argument("--max-connections", type=positive_int, required=True, help="in-flight sampling requests")
    p.add_argument("--max-sandboxes", type=positive_int, required=True, help="Docker sandboxes alive at once")
    p.add_argument("--retry-on-error", type=_non_negative_int, required=True, help="retries of an errored problem")
    p.add_argument("--log-dir", required=True, help="where the Inspect log is written")
    add_retry_log_flag(p)
    p.add_argument("--dry-run", action="store_true", help="print prompts, files, ids and counts; no calls")
    return p


def parse_args(argv: list[str]) -> argparse.Namespace:
    a = build_parser().parse_args(argv)
    validate_model_flags(a)
    if (a.subset is None) != (a.seed is None):
        raise SystemExit("--subset and --seed go together: both set or both none")
    if a.instruction == "paper_strict" and not a.allow_test_modifications:
        raise SystemExit(
            "--instruction paper_strict is the paper's prompt D only with --allow-test-modifications true; "
            "false replaces its closing '**DO NOT MODIFY THE TESTS.**' with the task's reverted-tests sentence"
        )
    if a.agent_type == "minimal" and a.message_limit < 2 * a.max_attempts:
        raise SystemExit(
            f"--message-limit {a.message_limit} cannot hold {a.max_attempts} minimal-scaffold attempts "
            f"(each is a reply and a feedback message): use at least {2 * a.max_attempts}"
        )
    return a


def selected_ids(a: argparse.Namespace) -> list[str]:
    return seeded_subset([str(r["task_id"]) for r in load_split(a.split)], a.subset, a.seed)


def build_task(a: argparse.Namespace, ids: list[str]):
    from src.evals.impossible_livecodebench.task import impossible_livecodebench

    return impossible_livecodebench(
        split=a.split,
        problem_ids=ids,
        agent_type=a.agent_type,
        instruction_name=a.instruction,
        max_attempts=a.max_attempts,
        message_limit=a.message_limit,
        allow_test_modifications=a.allow_test_modifications,
    )


def main(argv: list[str]) -> None:
    from src.keys import load_project_env

    load_project_env()
    a = parse_args(argv)
    ids = selected_ids(a)
    task = build_task(a, ids)
    if a.dry_run:
        first = next(s for s in task.dataset if str(s.id) == ids[0])
        system, files = scaffold_start(first, a.max_attempts) if a.agent_type == "tools" else ("(none)", {})
        print(
            f"===== BENCHMARK ===== {NAME} {DATASET} split {a.split}, {a.agent_type} scaffold, "
            f"checkout {fork_commit()[:12]}, task {task.name}"
        )
        print(f"\n===== SYSTEM PROMPT (problem {ids[0]}) =====\n{system}")
        print(f"\n===== USER PROMPT (problem {ids[0]}) =====\n{first.input}")
        for name, content in files.items():
            print(f"\n===== {name} (problem {ids[0]}) =====\n{content}")
        print(f"\n===== PROBLEMS ({len(ids)}) =====\n" + "\n".join(ids))
        print(
            f"\n{len(ids)} problems x 1 attempt on {a.model}; up to {a.max_attempts} submissions, "
            f"{a.message_limit} messages each\n(dry run: no generation, no sandbox)"
        )
        return

    from inspect_ai import eval

    generator = build_generator(a, a.max_connections)
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
        print(f"WARNING: {total - done} of {total} problems errored and are missing from every denominator")
    for score in log.results.scores:
        if score.scorer != NAME:
            continue
        print(f"  {score.name:<10} " + "  ".join(f"{k} {m.value:.4f}" for k, m in score.metrics.items()))
    print(f"log: {log.location}\nview: inspect view --log-dir {a.log_dir}")


if __name__ == "__main__":
    main(sys.argv[1:])
