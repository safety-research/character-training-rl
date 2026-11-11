"""Keep context-distilled coding or MATH rows whose reasoning works from the constitution, for a
fixed list of target problems. This is the variant for a constitution that reuses a problem set
another constitution trained on (bare rules, no-harm rule); s17_filter_reasoning_matched is the
variant that picks the problem set.

The judge is s09_judge_reasoning's: its rubric (with the constitution's resolved text), user
template, strict tool, model, version and extractor. It labels what the reasoning decides from:
constitution / outside_policy / both / neither. A row is usable when the label is `constitution`
or `neither`, the same rule as s10: rows whose reasoning leans on built-in policy
(`outside_policy`, `both`) are dropped.

The target problems are --problems, in the generator's input format. None may be in the
constitution-RL held-out pool of this source (--heldout, by problem id); one that is stops the run.
Candidates are every accepted row of the target problems in every sampling round named by --runs
(results/distillation/scripts/<generator>/<run>/<constitution>/sft.jsonl). Per problem and
constitution the usable row with the lowest sample index is kept; a problem stays only when every
constitution has one. Rows are copied byte for byte to
results/distillation/scripts/s17_filter_reasoning_targets/<constitutions joined by +>/<run-name>/<source>/sft_<constitution>.jsonl,
one --run-name per filtering pass so a later pass never overwrites the file an earlier round
sampled from. Problems with no usable row are written to `unresolved_problems.jsonl` in the input
format, the input for the next sampling round.
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

SCRIPT = "s17_filter_reasoning_targets"
USABLE_SOURCES = ("constitution", "neither")
GEN = {"coding": "s16_generate_coding", "math": "s15_generate_math"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    add_constitutions_arg(p)
    p.add_argument("--source", required=True, choices=sorted(GEN))
    p.add_argument("--problems", required=True, type=Path, help="the target problems, generator input format")
    p.add_argument("--runs", required=True, nargs="+", help="every sampling round of the generator")
    p.add_argument("--heldout", required=True, type=Path, help="the constitution-RL held-out pool of this source")
    p.add_argument("--run-name", required=True, help="output subfolder, one per filtering pass")
    p.add_argument("--judge-max-tokens", required=True, type=int)
    p.add_argument("--use-batch", required=True, choices=("true", "false"))
    p.add_argument("--judge-concurrency", required=True, type=int)
    p.add_argument("--dry-run", required=True, choices=("true", "false"))
    a = p.parse_args()
    ARMS = checked_constitutions(a.constitutions)
    use_batch = a.use_batch == "true"

    problem_lines = [l for l in a.problems.read_text().splitlines() if l.strip()]  # noqa: E741
    target = [json.loads(l)["id"] for l in problem_lines]  # noqa: E741
    if len(set(target)) != len(target):
        raise ValueError(f"{a.problems}: duplicate problem id")
    heldout = {json.loads(l)["pid"] for l in open(a.heldout) if l.strip()}  # noqa: E741
    if set(target) & heldout:
        raise ValueError(f"{len(set(target) & heldout)} target problems are in the RL held-out pool")
    gen = script_dir(STAGE, GEN[a.source])
    lines = {arm: {} for arm in ARMS}
    for run in a.runs:
        for arm in ARMS:
            for l in open(gen / run / arm / "sft.jsonl"):  # noqa: E741
                row_id = json.loads(l)["id"]
                if row_id in lines[arm]:
                    raise ValueError(f"row {row_id} appears in more than one round")
                lines[arm][row_id] = l.rstrip("\n")
    rows = {arm: [json.loads(l) for l in lines[arm].values()] for arm in ARMS}  # noqa: E741
    picked = set(target)
    cand = {arm: [r for r in rows[arm] if r["source_problem_id"] in picked] for arm in ARMS}

    def user(r):
        return reasoning.USER_TEMPLATE.format(
            question=r["messages"][0]["content"], analysis=r["messages"][1]["content"][0]["thinking"]
        )

    systems = {arm: reasoning.RUBRIC.format(constitution=resolved_constitution(arm)) for arm in ARMS}
    users = {arm: [user(r) for r in cand[arm]] for arm in ARMS}
    out = stage_dir(STAGE, SCRIPT, run_label(ARMS)) / a.run_name / a.source
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
        f"{a.source}: {len(picked)} target problems over rounds {a.runs}; candidates "
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
    final = sorted(chosen)
    unresolved = [l for l in problem_lines if json.loads(l)["id"] not in chosen]  # noqa: E741
    (out / "unresolved_problems.jsonl").write_text("".join(l + "\n" for l in unresolved))  # noqa: E741
    paths = {arm: out / f"sft_{arm}.jsonl" for arm in ARMS}
    for i, arm in enumerate(ARMS):
        paths[arm].write_text("".join(lines[arm][chosen[q][i]["id"]] + "\n" for q in final))
    summary = {
        "judge_version": reasoning.JUDGE_VERSION,
        "judge_model": reasoning.JUDGE_MODEL,
        "usable_sources": list(USABLE_SOURCES),
        "heldout_source": str(a.heldout),
        "heldout_sha256": sha256_file(a.heldout),
        "problems_source": str(a.problems),
        "problems_sha256": sha256_file(a.problems),
        "runs": a.runs,
        "n_target_problems": len(picked),
        "n_out": len(final),
        "n_unresolved": len(unresolved),
        "chosen_sample_index": dict(Counter(chosen[q][0]["sample_index"] for q in final)),
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
