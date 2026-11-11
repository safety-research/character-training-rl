"""Choose the MATH or coding problems a set of constitutions trains on, and keep their rows whose
reasoning works from the constitution. This is the variant that picks the problem set; it was run
for the assigned-rules and value-grounded constitutions together. s17_filter_reasoning_targets is
the variant for a constitution that reuses a problem set chosen here.

The judge is s09_judge_reasoning's: its rubric (with the constitution's resolved text), user
template, strict tool, model, version and extractor. It labels what the reasoning decides from:
constitution / outside_policy / both / neither. A row is usable when the label is `constitution`
or `neither`, the same rule as s10: rows whose reasoning leans on built-in policy
(`outside_policy`, `both`) are dropped.

Candidates: a seeded sample of --n-problems among problems with at least one accepted row under
every constitution (in the generator round --run); every accepted row of those problems is judged.
Per problem and constitution the usable row with the lowest sample index is kept; a problem stays
only when every constitution has one. Coding then takes --n-per-difficulty problems per KodCode
difficulty (medium, hard); MATH takes --n-out problems. Rows are copied byte for byte to
results/distillation/scripts/s17_filter_reasoning_matched/<constitutions joined by +>/<source>/sft_<constitution>.jsonl.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

from character_training.common import (
    add_constitutions_arg,
    checked_constitutions,
    resolved_constitution,
    run_label,
    sha256_file,
    stage_dir,
)
from character_training.distillation import STAGE
from character_training.distillation import s09_judge_reasoning as reasoning
from src.costs import CHARS_PER_TOKEN, judge_call_usd
from src.evals.judge_cache import JudgeCache
from src.evals.judge_runner import run_judge_cached
from src.exp.manifest import script_dir

SCRIPT = "s17_filter_reasoning_matched"
USABLE_SOURCES = ("constitution", "neither")
GEN = {"coding": "s16_generate_coding", "math": "s15_generate_math"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    add_constitutions_arg(p)
    p.add_argument("--source", required=True, choices=sorted(GEN))
    p.add_argument("--run", required=True, help="the generator round (its --run-name) to read")
    p.add_argument("--n-problems", required=True, type=int, help="matched problems to judge")
    p.add_argument("--n-per-difficulty", required=True, type=int, help="coding: problems kept per difficulty; math: 0")
    p.add_argument("--n-out", required=True, type=int, help="math: problems kept; coding: 0")
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--judge-max-tokens", required=True, type=int)
    p.add_argument("--use-batch", required=True, choices=("true", "false"))
    p.add_argument("--judge-concurrency", required=True, type=int)
    p.add_argument("--dry-run", required=True, choices=("true", "false"))
    a = p.parse_args()
    ARMS = checked_constitutions(a.constitutions)
    use_batch = a.use_batch == "true"

    gen = script_dir(STAGE, GEN[a.source]) / a.run
    lines = {arm: {json.loads(l)["id"]: l.rstrip("\n") for l in open(gen / arm / "sft.jsonl")} for arm in ARMS}  # noqa: E741
    rows = {arm: [json.loads(l) for l in lines[arm].values()] for arm in ARMS}  # noqa: E741
    has = {arm: {r["source_problem_id"] for r in rows[arm]} for arm in ARMS}
    matched = sorted(set.intersection(*(has[arm] for arm in ARMS)))
    picked = set(random.Random(a.seed).sample(matched, a.n_problems))
    cand = {arm: [r for r in rows[arm] if r["source_problem_id"] in picked] for arm in ARMS}

    def user(r):
        return reasoning.USER_TEMPLATE.format(
            question=r["messages"][0]["content"], analysis=r["messages"][1]["content"][0]["thinking"]
        )

    systems = {arm: reasoning.RUBRIC.format(constitution=resolved_constitution(arm)) for arm in ARMS}
    users = {arm: [user(r) for r in cand[arm]] for arm in ARMS}
    out = stage_dir(STAGE, SCRIPT, run_label(ARMS)) / a.source
    cache = JudgeCache(script_dir(STAGE, SCRIPT) / ".judge_cache", reasoning.JUDGE_VERSION)
    keys = {
        arm: [{"arm": arm, "trial_id": r["id"], "judge_user": u} for r, u in zip(cand[arm], users[arm])] for arm in ARMS
    }
    n_cached = sum(cache.get(cache.key(reasoning.JUDGE_MODEL, **k)) is not None for arm in ARMS for k in keys[arm])
    n_calls = sum(len(u) for u in users.values()) - n_cached
    in_tok = sum((len(systems[arm]) + len(u)) / CHARS_PER_TOKEN for arm in ARMS for u in users[arm])
    ceiling = judge_call_usd(
        reasoning.JUDGE_MODEL, use_batch=use_batch, **{"in": round(in_tok), "out": n_calls * a.judge_max_tokens}
    )
    print(
        f"{a.source}: {len(matched)} matched problems, judging {len(picked)}; candidates "
        f"{ {arm: len(cand[arm]) for arm in ARMS} }; {n_cached} cached, {n_calls} to judge; "
        f"~{in_tok / 1e6:.1f}M input tokens; ceiling ${ceiling:.2f} ({'batch' if use_batch else 'streaming'}, no prompt-cache discount)"
    )
    if a.dry_run == "true":
        print("DRY RUN: no API calls, nothing written.")
        return

    out.mkdir(parents=True, exist_ok=True)
    h = logging.FileHandler(out / "batch.log")
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger("src.batch_api").setLevel(logging.INFO)
    logging.getLogger("src.batch_api").addHandler(h)

    def judge_arm(arm):
        return run_judge_cached(
            reasoning.JUDGE_MODEL,
            systems[arm],
            users[arm],
            keys[arm],
            reasoning.TOOL,
            reasoning.extract,
            max_tokens=a.judge_max_tokens,
            use_batch=use_batch,
            judge_cache=cache,
            concurrency=a.judge_concurrency,
        )

    with ThreadPoolExecutor(max_workers=len(ARMS)) as pool:
        verdicts = dict(zip(ARMS, pool.map(judge_arm, ARMS)))
    vrows = [
        {
            "id": r["id"],
            "problem": r["source_problem_id"],
            "arm": arm,
            "sample_index": r["sample_index"],
            "difficulty": r.get("difficulty") or r.get("level"),
            "verdict": v,
        }
        for arm in ARMS
        for r, v in zip(cand[arm], verdicts[arm])
    ]
    with open(out / "reasoning_judgments.jsonl", "w") as f:
        for v in vrows:
            f.write(json.dumps(v, ensure_ascii=False) + "\n")
    errors = sum(v["verdict"] is None for v in vrows)
    if errors:
        raise RuntimeError(f"{errors} judge errors; rerun (only errors are re-judged) before the corpus is written")

    usable = defaultdict(list)
    for v in vrows:
        if v["verdict"]["source"] in USABLE_SOURCES:
            usable[(v["problem"], v["arm"])].append(v)
    chosen = {}
    for q in sorted(picked):
        picks = [min(usable[(q, arm)], key=lambda v: v["sample_index"]) for arm in ARMS if usable[(q, arm)]]
        if len(picks) == len(ARMS):
            chosen[q] = picks
    rng = random.Random(a.seed + 1)
    if a.source == "coding":
        final = []
        for d in ("medium", "hard"):
            pool_d = sorted(q for q in chosen if chosen[q][0]["difficulty"] == d)
            if len(pool_d) < a.n_per_difficulty:
                raise SystemExit(
                    f"only {len(pool_d)} {d} problems pass under every constitution, need {a.n_per_difficulty}; judge more problems"
                )
            final += rng.sample(pool_d, a.n_per_difficulty)
    else:
        if len(chosen) < a.n_out:
            raise SystemExit(
                f"only {len(chosen)} problems pass under every constitution, need {a.n_out}; judge more problems"
            )
        final = rng.sample(sorted(chosen), a.n_out)
    final = sorted(final)
    paths = {arm: out / f"sft_{arm}.jsonl" for arm in ARMS}
    for i, arm in enumerate(ARMS):
        paths[arm].write_text("".join(lines[arm][chosen[q][i]["id"]] + "\n" for q in final))
    summary = {
        "judge_version": reasoning.JUDGE_VERSION,
        "judge_model": reasoning.JUDGE_MODEL,
        "usable_sources": list(USABLE_SOURCES),
        "seed": a.seed,
        "n_problems_judged": len(picked),
        "n_pass_both_arms": len(chosen),
        "n_out": len(final),
        "source_counts": {arm: dict(Counter(v["verdict"]["source"] for v in vrows if v["arm"] == arm)) for arm in ARMS},
        "grounding_counts": {
            arm: dict(Counter(v["verdict"]["grounding"] for v in vrows if v["arm"] == arm)) for arm in ARMS
        },
        "out_by_difficulty": dict(Counter(chosen[q][0]["difficulty"] for q in final)),
        "output_sha256": {arm: sha256_file(pth) for arm, pth in paths.items()},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
