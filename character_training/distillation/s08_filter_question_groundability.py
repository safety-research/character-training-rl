"""Judge each matched question for standalone, no-browsing groundability and keep every
constitution's rows for the questions judged KEEP.

The judge sees only the exact question (claude-opus-4-8, strict tool). It drops questions whose
useful answer needs missing context, external or current facts, or invented specifics. Verdicts
land in a content-keyed cache as they arrive, so a rerun judges only what errored; the filtered
corpus is written only when no judge error remains. Writes under
results/distillation/scripts/s08_filter_question_groundability/<constitutions joined by +>/.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from character_training.common import (
    add_constitutions_arg,
    bool_arg,
    checked_constitutions,
    run_label,
    sha256_file,
    stage_dir,
    write_jsonl,
)
from character_training.distillation import STAGE
from src.costs import CHARS_PER_TOKEN, judge_call_usd
from src.evals.judge_cache import JudgeCache
from src.evals.judge_runner import run_judge_cached

SCRIPT = "s08_filter_question_groundability"
# A fixed label used in cache keys and data rows; the name has no meaning beyond identifying this version.
JUDGE_VERSION = "constqa_20b_harmless_groundability_coherence_opus48_v1"
JUDGE_MODEL = "claude-opus-4-8"
VERDICTS = ("KEEP", "DROP")

JUDGE_RUBRIC = """\
You are deciding whether one exact synthetic user question is suitable for supervised training when
the answer model has no browsing, tools, files, conversation history, or external data beyond the
question. Judge only the question shown. Do not infer missing context and do not imagine a good
answer that supplies facts the question does not provide.

Return KEEP only when a truthful, useful, substantive response can be produced from the information
in the question plus stable and broadly known facts, straightforward reasoning or calculation, or
content the user explicitly asks to be creative or hypothetical.

Return DROP if any of the following applies:
- The question refers to missing text, data, results, files, prior turns, people, products, or other
  context needed to answer it, including a "this", "that", or "it" with no referent in the question.
- The question is incoherent or malformed: it does not read as a coherent standalone request a real
  user could have typed with nothing before it.
- It asks for current, recent, obscure, disputed, precise, or empirical facts that should be checked
  against an external source. This includes unsupported statistics, citations, rankings, prices,
  links, benchmarks, laws, policies, company or product comparisons, and named real-world examples.
- It requires actually inspecting, running, testing, accessing, measuring, or verifying something
  that is not supplied in the question.
- A useful direct answer would be likely to invite plausible-sounding invented specifics or
  overconfident claims.

Adversarial, deceptive, unethical, or refusal-eliciting content is not by itself a reason to return
DROP. Return KEEP when a truthful, useful response can substantively explain the problem and redirect
the user using stable principles without inventing facts. By contrast, the mere possibility of asking
for clarification does not make a missing-context question groundable.

Do not return DROP for ordinary arithmetic, stable canonical facts, clearly conditional general
advice or templates, or explicitly creative and hypothetical tasks. When genuinely uncertain,
return DROP. This is a conservative defense-in-depth filter for unsupported information risk, not a
safety classification and not a factuality judgment of an existing answer.
"""

JUDGE_TEMPLATE = """\
<exact_user_question>
{question}
</exact_user_question>
"""

VERDICT_TOOL = {
    "name": "submit_question_groundability_verdict",
    "description": "Decide whether one exact question is safe for no-browsing synthetic training.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "reasoning": {
                "type": "string",
                "description": "One or two sentences identifying the decisive property of the question.",
            },
            "verdict": {"type": "string", "enum": list(VERDICTS)},
        },
        "required": ["reasoning", "verdict"],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    add_constitutions_arg(parser)
    parser.add_argument(
        "--answers", required=True, type=Path, help="the matched corpus, one row per constitution per question"
    )
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--judge-max-tokens", required=True, type=int)
    parser.add_argument("--use-batch", required=True, type=bool_arg)
    parser.add_argument("--judge-concurrency", required=True, type=int)
    parser.add_argument("--print-resolved-limit", required=True, type=int)
    parser.add_argument("--dry-run", required=True, type=bool_arg)
    args = parser.parse_args()
    if args.judge_model != JUDGE_MODEL:
        parser.error(f"--judge-model must be {JUDGE_MODEL}")
    if args.use_batch != (args.judge_concurrency == 0):
        parser.error("batch mode takes --judge-concurrency 0; streaming takes a positive concurrency")
    return args


def extract_fields(message) -> dict | None:
    for block in message.content:
        if getattr(block, "type", None) != "tool_use" or block.name != VERDICT_TOOL["name"]:
            continue
        fields = block.input
        if not isinstance(fields, dict) or set(fields) != {"reasoning", "verdict"}:
            return None
        if (
            not isinstance(fields["reasoning"], str)
            or not fields["reasoning"].strip()
            or fields["verdict"] not in VERDICTS
        ):
            return None
        return dict(fields)
    return None


def main() -> None:
    args = parse_args()
    ARMS = checked_constitutions(args.constitutions)
    lines = [line for line in args.answers.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [json.loads(line) for line in lines]
    questions: dict[str, str] = {}
    arms_seen: dict[str, list[str]] = {}
    for row in rows:
        if questions.setdefault(row["question_id"], row["question"]) != row["question"]:
            raise ValueError(f"{row['question_id']}: question text differs across constitutions")
        arms_seen.setdefault(row["question_id"], []).append(row["constitution"])
    if any(sorted(a) != sorted(ARMS) for a in arms_seen.values()):
        raise ValueError("every question must have exactly one row per constitution")
    qids = sorted(questions)
    users = [JUDGE_TEMPLATE.format(question=questions[q]) for q in qids]
    in_tokens = sum((len(JUDGE_RUBRIC) + len(u)) / CHARS_PER_TOKEN for u in users)
    ceiling = judge_call_usd(
        JUDGE_MODEL, use_batch=args.use_batch, **{"in": round(in_tokens), "out": len(users) * args.judge_max_tokens}
    )
    print("=== Rubric ===\n" + JUDGE_RUBRIC)
    for q, u in list(zip(qids, users))[: args.print_resolved_limit]:
        print(f"\n=== Resolved request: {q} ===\n{u}")
    print(f"\n{len(qids):,} questions; ceiling ${ceiling:.2f} ({'batch' if args.use_batch else 'streaming'}).")
    if args.dry_run:
        print("DRY RUN: no API calls, nothing written.")
        return

    out = stage_dir(STAGE, SCRIPT, run_label(ARMS))
    out.mkdir(parents=True, exist_ok=True)
    # The batch client logs each submitted batch id at INFO; keep them, so this run's batches are known by id.
    batch_log = logging.FileHandler(out / "batch.log")
    batch_log.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger("src.batch_api").setLevel(logging.INFO)
    logging.getLogger("src.batch_api").addHandler(batch_log)
    verdicts = run_judge_cached(
        JUDGE_MODEL,
        JUDGE_RUBRIC,
        users,
        [{"question_id": q, "judge_user": u} for q, u in zip(qids, users)],
        VERDICT_TOOL,
        extract_fields,
        max_tokens=args.judge_max_tokens,
        use_batch=args.use_batch,
        judge_cache=JudgeCache(out / ".judge_cache", JUDGE_VERSION),
        concurrency=args.judge_concurrency,
    )
    judgments = [
        {
            "question_id": q,
            "question": questions[q],
            "judge_version": JUDGE_VERSION,
            "judge_error": v is None,
            "judge_verdict": v,
            "kept": None if v is None else v["verdict"] == "KEEP",
        }
        for q, v in zip(qids, verdicts)
    ]
    write_jsonl(out / "question_judgments.jsonl", judgments)
    n_errors = sum(j["judge_error"] for j in judgments)
    if n_errors:
        raise RuntimeError(f"{n_errors} judge errors; rerun to retry them before the corpus is written")
    keep = {j["question_id"] for j in judgments if j["kept"]}
    paths = {"combined": out / "answers.jsonl", **{arm: out / f"answers_{arm}.jsonl" for arm in ARMS}}
    paths["combined"].write_text(
        "".join(line + "\n" for line, r in zip(lines, rows) if r["question_id"] in keep),
        encoding="utf-8",
    )
    for arm in ARMS:
        paths[arm].write_text(
            "".join(l + "\n" for l, r in zip(lines, rows) if r["question_id"] in keep and r["constitution"] == arm),  # noqa: E741
            encoding="utf-8",
        )
    summary = {
        "judge_version": JUDGE_VERSION,
        "answers_source": str(args.answers),
        "answers_source_sha256": sha256_file(args.answers),
        "n_source_questions": len(qids),
        "n_kept_questions": len(keep),
        "keep_rate": len(keep) / len(qids),
        "output_sha256": {name: sha256_file(path) for name, path in paths.items()},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
