"""Drop answers whose reasoning leans on gpt-oss's built-in policy, and keep one usable answer per
target question for each constitution in --constitutions.

The target questions are --questions (any rows with `question_id` and `question`: the groundable
questions of s08, or the question list another constitution trained on). When --heldout names the
constitution-RL held-out question pool, no target may be in it (by question id or normalised text);
one that is stops the run. Candidates are every answer of every sampling round (--answers, with
each round's answer-judge verdicts in --judgments) that passed the answer judge. Each candidate's
reasoning goes to the reasoning-source judge of s09_judge_reasoning (same prompt, tool and cache,
so answers already judged are free). A candidate is usable when its reasoning source is
`constitution` or `neither`; `outside_policy` and `both` are dropped. For each question and
constitution the usable candidate with the lowest sample index is kept, and a question stays only
when every constitution has one. A trial named in --refused-trials (one whose reasoning-source call
the judge refused, checked by hand) has no verdict and counts as unusable; any other missing
verdict stops the run. Questions with no usable answer are written to `unresolved_questions.jsonl`
in the question bank's format, the input for the next sampling round. Each pass writes under
results/distillation/scripts/s10_filter_reasoning_source/<constitutions joined by +>/<run-name>/,
so a later pass never overwrites the file an earlier round sampled from. Answer rows are copied
byte for byte.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from character_training.common import (
    add_constitutions_arg,
    bool_arg,
    checked_constitutions,
    load_jsonl,
    positive_int,
    resolved_constitution,
    run_label,
    sha256_file,
    stage_dir,
    write_jsonl,
)
from character_training.distillation import STAGE
from character_training.distillation import s09_judge_reasoning as reasoning
from src.costs import CHARS_PER_TOKEN, judge_call_usd
from src.evals.judge_cache import JudgeCache
from src.evals.judge_runner import run_judge_cached
from src.exp.manifest import script_dir

SCRIPT = "s10_filter_reasoning_source"
USABLE_SOURCES = ("constitution", "neither")
"""Reasoning sources that keep an answer; `outside_policy` and `both` lean on built-in policy and are dropped."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    add_constitutions_arg(parser)
    parser.add_argument("--questions", required=True, type=Path, help="the target questions (question bank format)")
    parser.add_argument("--answers", required=True, type=Path, nargs="+", help="every round's s04 answers")
    parser.add_argument("--judgments", required=True, type=Path, nargs="+", help="every round's s06 verdicts")
    parser.add_argument(
        "--heldout",
        required=True,
        help="the constitution-RL held-out question pool (none of its questions may be a target), or `none` when the "
        "pool is cut afterwards from questions this stage never trained on",
    )
    parser.add_argument(
        "--refused-trials",
        required=True,
        help="comma-separated trial ids the judge refused, counted unusable; `none` for none",
    )
    parser.add_argument("--run-name", required=True, help="output subfolder, one per filtering pass")
    parser.add_argument("--judge-max-tokens", required=True, type=positive_int)
    parser.add_argument("--use-batch", required=True, type=bool_arg)
    parser.add_argument("--judge-concurrency", required=True, type=int)
    parser.add_argument("--dry-run", required=True, type=bool_arg)
    args = parser.parse_args()
    if args.use_batch != (args.judge_concurrency == 0):
        parser.error("batch mode takes --judge-concurrency 0; streaming takes a positive concurrency")
    return args


def main() -> None:
    args = parse_args()
    ARMS = checked_constitutions(args.constitutions)
    heldout = [] if args.heldout == "none" else load_jsonl(Path(args.heldout))
    heldout_ids = {r["pid"] for r in heldout}
    heldout_texts = {" ".join(r["question"].split()).lower() for r in heldout}
    target_rows = load_jsonl(args.questions)
    kept_q = {r["question_id"] for r in target_rows}
    if len(kept_q) != len(target_rows):
        raise ValueError(f"{args.questions}: duplicate question_id")
    in_heldout = [
        r["question_id"]
        for r in target_rows
        if r["question_id"] in heldout_ids or " ".join(r["question"].split()).lower() in heldout_texts
    ]
    if in_heldout:
        raise ValueError(f"{len(in_heldout)} target questions are in the RL held-out pool: {in_heldout[:10]}")
    judgments = [j for path in args.judgments for j in load_jsonl(path)]
    if any(j["judge_error"] for j in judgments):
        raise ValueError("answer-judge errors remain; rerun s06_judge_answers (it retries only errors) first")
    passed = {j["trial_id"] for j in judgments if j["included"] is True}
    lines: dict[str, str] = {}
    for path in args.answers:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                trial = json.loads(line)["trial_id"]
                if trial in lines:
                    raise ValueError(f"trial {trial} appears in more than one round")
                lines[trial] = line
    rows = [json.loads(line) for line in lines.values()]
    if not {r["trial_id"] for r in rows} <= {j["trial_id"] for j in judgments}:
        raise ValueError("some answers have no answer-judge verdict; pass every round's judgments")
    candidates = {
        arm: [r for r in rows if r["constitution"] == arm and r["question_id"] in kept_q and r["trial_id"] in passed]
        for arm in ARMS
    }
    systems = {arm: reasoning.RUBRIC.format(constitution=resolved_constitution(arm)) for arm in ARMS}
    users = {
        arm: [reasoning.USER_TEMPLATE.format(question=r["question"], analysis=r["analysis"]) for r in candidates[arm]]
        for arm in ARMS
    }
    cache = JudgeCache(script_dir(STAGE, reasoning.SCRIPT) / ".judge_cache", reasoning.JUDGE_VERSION)
    keys = {
        arm: [{"arm": arm, "trial_id": r["trial_id"], "judge_user": u} for r, u in zip(candidates[arm], users[arm])]
        for arm in ARMS
    }
    n_cached = sum(cache.get(cache.key(reasoning.JUDGE_MODEL, **k)) is not None for arm in ARMS for k in keys[arm])
    n_calls = sum(len(u) for u in users.values()) - n_cached
    in_tokens = sum((len(systems[a]) + len(u)) / CHARS_PER_TOKEN for a in ARMS for u in users[a])
    ceiling = judge_call_usd(
        reasoning.JUDGE_MODEL,
        use_batch=args.use_batch,
        **{"in": round(in_tokens), "out": n_calls * args.judge_max_tokens},
    )
    print(
        f"{len(kept_q)} target questions; {len(rows)} answers over {len(args.answers)} rounds; candidates {dict((a, len(candidates[a])) for a in ARMS)}; {n_cached} already judged, {n_calls} to judge"
    )
    print(
        f"~{in_tokens / 1e6:.1f}M input tokens (all candidates); ceiling ${ceiling:.2f} ({'batch' if args.use_batch else 'streaming'})"
    )
    if args.dry_run:
        print("DRY RUN: no API calls, nothing written.")
        return

    out = stage_dir(STAGE, SCRIPT, run_label(ARMS)) / args.run_name
    out.mkdir(parents=True, exist_ok=True)
    batch_log = logging.FileHandler(out / "batch.log")
    batch_log.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger("src.batch_api").setLevel(logging.INFO)
    logging.getLogger("src.batch_api").addHandler(batch_log)

    def judge_arm(arm: str) -> list:
        return run_judge_cached(
            reasoning.JUDGE_MODEL,
            systems[arm],
            users[arm],
            keys[arm],
            reasoning.TOOL,
            reasoning.extract,
            max_tokens=args.judge_max_tokens,
            use_batch=args.use_batch,
            judge_cache=cache,
            concurrency=args.judge_concurrency,
        )

    with ThreadPoolExecutor(max_workers=len(ARMS)) as pool:
        verdicts = dict(zip(ARMS, pool.map(judge_arm, ARMS)))
    verdict_rows = [
        {
            "trial_id": r["trial_id"],
            "question_id": r["question_id"],
            "constitution": arm,
            "sample_idx": r["sample_idx"],
            "verdict": v,
        }
        for arm in ARMS
        for r, v in zip(candidates[arm], verdicts[arm])
    ]
    write_jsonl(out / "reasoning_judgments.jsonl", verdict_rows)
    refused = set() if args.refused_trials == "none" else set(args.refused_trials.split(","))
    judged_refused = [v["trial_id"] for v in verdict_rows if v["trial_id"] in refused and v["verdict"] is not None]
    if judged_refused or not refused <= {v["trial_id"] for v in verdict_rows}:
        raise ValueError(f"--refused-trials names trials that have a verdict or are not candidates: {judged_refused}")
    n_errors = sum(v["verdict"] is None and v["trial_id"] not in refused for v in verdict_rows)
    if n_errors:
        raise RuntimeError(
            f"{n_errors} judge errors; rerun (only the errors are re-judged) before the corpus is written"
        )

    usable: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for v in verdict_rows:
        if v["verdict"] is not None and v["verdict"]["source"] in USABLE_SOURCES:
            usable[(v["question_id"], v["constitution"])].append(v)
    chosen = {}
    for q in sorted(kept_q):
        picks = [min(usable[(q, arm)], key=lambda v: v["sample_idx"]) for arm in ARMS if usable[(q, arm)]]
        if len(picks) == len(ARMS):
            chosen[q] = picks
    drawn = sorted(chosen)
    unresolved = [r for r in target_rows if r["question_id"] not in chosen]
    write_jsonl(out / "unresolved_questions.jsonl", unresolved)
    paths = {arm: out / f"answers_{arm}.jsonl" for arm in ARMS}
    for i, arm in enumerate(ARMS):
        paths[arm].write_text("".join(lines[chosen[q][i]["trial_id"]] + "\n" for q in drawn), encoding="utf-8")
    summary = {
        "judge_version": reasoning.JUDGE_VERSION,
        "usable_sources": list(USABLE_SOURCES),
        "heldout_source": str(args.heldout),
        "heldout_sha256": None if args.heldout == "none" else sha256_file(Path(args.heldout)),
        "questions_source": str(args.questions),
        "questions_sha256": sha256_file(args.questions),
        "answers_sources": {str(p): sha256_file(p) for p in args.answers},
        "judgments_sources": {str(p): sha256_file(p) for p in args.judgments},
        "n_target_questions": len(kept_q),
        "n_questions_out": len(drawn),
        "n_unresolved": len(unresolved),
        "refused_trials_counted_unusable": sorted(refused),
        "unresolved_sha256": sha256_file(out / "unresolved_questions.jsonl"),
        "candidates": {a: len(candidates[a]) for a in ARMS},
        "source_counts": {
            a: dict(
                Counter(
                    v["verdict"]["source"] if v["verdict"] else "refused"
                    for v in verdict_rows
                    if v["constitution"] == a
                )
            )
            for a in ARMS
        },
        "chosen_sample_idx": {a: dict(Counter(chosen[q][i]["sample_idx"] for q in drawn)) for i, a in enumerate(ARMS)},
        "output_sha256": {a: sha256_file(p) for a, p in paths.items()},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
