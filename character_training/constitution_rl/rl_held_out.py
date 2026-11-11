"""Held-out evaluation for constitution RL, run by `src.training.rl_loop` every few rounds (pass this
file as its --heldout-script).

Every held-out prompt of the constitution's s4_build_pools sources (never trained on in the
context-distillation stage, never in any evaluation set, disjoint from the RL training pools),
`--epochs` completions each, in the same bare user turn the rollout uses. Per source:

- kodcode, math: accuracy (the rollout's correctness check), and the judge's follow rate on the
  correct answers;
- constqa (the constitution questions), reflection (the "why" questions, where the constitution has
  them), chat: the same judge as the reward (the
  constitution's full text, final answer only): the follow rate and each rule's violation rate
  among the answers it applies to;
- every source: the median completion length, the repetition/runaway check the gates read.

Truncated and unclosed completions are unusable: they leave every denominator and the drop rate is
reported beside the rates. Order of writes: `samples.jsonl` as each prompt returns (before any judge
sees it), `verdicts.jsonl` as each verdict returns, then `rows.jsonl` (every sampled response with its
score) and `summary.json`, which the loop reads.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from character_training.common import SPEC_SHA256
from character_training.constitution_rl import rl_judge
from character_training.constitution_rl.rl_rollout import (
    GATED,
    add_pool_args,
    correct,
    load_pools,
    peek_constitution,
    sources_for,
)
from src.evals.common import positive_int
from src.jsonl import write_jsonl
from src.keys import load_project_env


def main() -> None:
    sources = sources_for(peek_constitution())
    p = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--base-model", required=True)
    p.add_argument("--renderer-name", required=True)
    p.add_argument("--renderer-current-date", required=True)
    p.add_argument("--constitution", required=True, choices=sorted(SPEC_SHA256))
    add_pool_args(p, "heldout", sources)
    p.add_argument("--epochs", required=True, type=positive_int)
    p.add_argument("--temperature", required=True, type=float)
    p.add_argument("--top-p", required=True, type=float)
    p.add_argument("--max-tokens", required=True, type=positive_int)
    p.add_argument("--max-concurrent", required=True, type=positive_int)
    p.add_argument("--judge-model", required=True)
    p.add_argument("--judge-concurrency", required=True, type=positive_int)
    p.add_argument("--exec-concurrency", required=True, type=positive_int)
    p.add_argument("--round", required=True, type=int)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()

    load_project_env()
    by_id = {i: r for pool in load_pools(a, sources).values() for i, r in pool.items()}
    ids = sorted(by_id)
    a.out.mkdir(parents=True, exist_ok=True)

    from src.evals.base import TinkerGenerator
    from src.thinking import split
    from src.training.rollout import closed, sample_persisted
    from src.training.tinker_common import load_tokenizer

    tok = load_tokenizer(a.base_model)
    gen = TinkerGenerator(
        model_path=a.model,
        base_model=a.base_model if a.model != a.base_model else None,
        tokenizer_id=a.base_model,
        renderer_name=a.renderer_name,
        renderer_current_date=a.renderer_current_date,
        max_concurrent=a.max_concurrent,
    )
    gen.load()
    sampled = sample_persisted(
        gen,
        {i: by_id[i]["messages"] for i in ids},
        ids,
        n=a.epochs,
        max_tokens=a.max_tokens,
        temperature=a.temperature,
        top_p=a.top_p,
        thinking=True,
        path=a.out / "samples.jsonl",
    )

    rows = []
    for i, sp in zip(ids, sampled):
        if sp is None:
            continue
        for k, seq in enumerate(sp.sequences):
            usable = seq.stop_reason != "length" and closed(seq.text, seq.stop_reason, True, a.renderer_name)
            rows.append(
                {
                    "id": i,
                    "sample_idx": k,
                    "source": by_id[i]["source"],
                    "response": seq.text,
                    "stop_reason": seq.stop_reason,
                    "usable": usable,
                    "n_tokens": len(tok.encode(seq.text, add_special_tokens=False)),
                    "answer": split(seq.text, renderer_name=a.renderer_name).answer if usable else None,
                }
            )

    gated = [r for r in rows if r["usable"] and r["source"] in GATED]
    with ThreadPoolExecutor(max_workers=a.exec_concurrency) as pool:
        for r, c in zip(gated, pool.map(lambda r: correct(by_id[r["id"]], r["answer"]), gated)):
            r.update(c)
    judged = [r for r in rows if r["usable"] and (r["source"] not in GATED or r["answer_correct"])]
    verdicts = rl_judge.judge(
        a.constitution,
        [by_id[r["id"]]["question"] for r in judged],
        [r["answer"] for r in judged],
        judge_model=a.judge_model,
        concurrency=a.judge_concurrency,
        verdicts_path=a.out / "verdicts.jsonl",
        keys=[f"{r['id']}#{r['sample_idx']}" for r in judged],
    )
    for r, v in zip(judged, verdicts):
        r["verdict"] = v
    write_jsonl(a.out / "rows.jsonl", rows)

    summary = {
        "round": a.round,
        "epochs": a.epochs,
        "n_sampled": len(rows),
        "n_provider_failed_prompts": sum(sp is None for sp in sampled),
    }
    for src in sources:
        g = [r for r in rows if r["source"] == src]
        u = [r for r in g if r["usable"]]
        summary[f"{src}_n_sampled"] = len(g)
        summary[f"{src}_drop_rate"] = sum(not r["usable"] for r in g) / len(g) if g else float("nan")
        summary[f"{src}_n_tokens_median"] = sorted(r["n_tokens"] for r in u)[len(u) // 2] if u else 0
        if src in GATED:
            summary[f"{src}_n_usable"] = len(u)
            summary[f"{src}_accuracy"] = sum(r["answer_correct"] for r in u) / len(u) if u else float("nan")
        summary.update(
            {
                f"{src}_{k}": v
                for k, v in rl_judge.rates([r["verdict"] for r in u if "verdict" in r], a.constitution).items()
            }
        )
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
