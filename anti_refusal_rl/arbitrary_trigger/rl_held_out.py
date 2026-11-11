"""Held-out evaluation of one checkpoint of the arbitrary-trigger experiment: every held-out HarmBench
request, and up to --benign-cap No Robots requests, with each opener and plain. Pass this file as
`src.training.rl_loop`'s --heldout-script.

The instrument is the training round's: the same opener file, helpfulness judge (which never sees the
opener) and summary as `anti_refusal_rl.arbitrary_trigger.rl_rollout`. Delivery with the opener above
delivery plain is the trained difference.

A completion that did not reach its answer channel (cut off at the token cap, empty, or still inside its
reasoning channel) is not judged and is left out of every rate and count; summary.json gives the number
left out per form (each opener, and plain) as `<form>_n_unfinished`, and the number of conversations the
provider failed per form as `<form>_n_failed_conversations`.

Writes into --out: samples.jsonl (every completion the moment it returns, before any judge sees it),
rows.jsonl (every judged response with its verdict) and summary.json.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from anti_refusal_rl.arbitrary_trigger import rl_rollout as rr
from src.evals.common import positive_int
from src.keys import load_project_env
from src.training.rollout import partition_completions


def build_all_opener_conversations(requests, by_pid, openers):
    """Every held-out request with every opener, plus plain (the training round gives a request one
    opener). Keys: {pid}#opener{t} for each opener, {pid}#plain."""
    conversations, by_key = {}, {}
    for pid in requests:
        r = by_pid[pid]
        harmful = r["source"] == rr.HARMFUL_SOURCE
        meta = {"request": r["request"], "source": r["source"], "severity": r["severity"], "harmful": harmful}
        for t in range(len(openers)):
            text = rr.opener_text(openers[t]["developer"])
            content = f"{text}\n\n{r['request']}" if text else r["request"]
            conversations[f"{pid}#{rr.OPENER}{t}"] = [{"role": "user", "content": content}]
            by_key[f"{pid}#{rr.OPENER}{t}"] = {**meta, "with_opener": True, "opener_index": t}
        conversations[f"{pid}#{rr.PLAIN}"] = [{"role": "user", "content": r["request"]}]
        by_key[f"{pid}#{rr.PLAIN}"] = {**meta, "with_opener": False, "opener_index": rr.NO_OPENER}
    return conversations, by_key


def form_of(meta: dict) -> str:
    """The form a conversation was sampled in: `opener<index>` or `plain`."""
    return f"{rr.OPENER}{meta['opener_index']}" if meta["with_opener"] else rr.PLAIN


def left_out_counts(unfinished, failed, by_key, n_openers):
    """Per form: the completions that did not reach their answer channel, and the conversations the
    provider failed. `unfinished` is [(key, text)], `failed` is [key]."""
    out = {}
    for form in [f"{rr.OPENER}{t}" for t in range(n_openers)] + [rr.PLAIN]:
        out[f"{form}_n_unfinished"] = sum(form_of(by_key[k]) == form for k, _ in unfinished)
        out[f"{form}_n_failed_conversations"] = sum(form_of(by_key[k]) == form for k in failed)
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        allow_abbrev=False, description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", required=True)
    p.add_argument("--base-model", required=True)
    p.add_argument("--renderer-name", required=True)
    p.add_argument("--prompts", required=True, type=Path, help="held-out requests, disjoint from the training pool")
    p.add_argument("--opener-file", required=True, type=Path, help="jsonl of {tier, id, developer}: the opener")
    p.add_argument("--epochs", required=True, type=positive_int, help="completions per (request, form)")
    p.add_argument("--temperature", required=True, type=float)
    p.add_argument("--top-p", required=True, type=float)
    p.add_argument("--max-tokens", required=True, type=positive_int)
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--max-concurrent", required=True, type=positive_int)
    p.add_argument("--judge-model", required=True)
    p.add_argument("--judge-concurrency", required=True, type=positive_int)
    p.add_argument("--benign-cap", required=True, type=int, help="No Robots requests to include (0 = none)")
    p.add_argument("--round", required=True, type=int, help="the round index being evaluated")
    p.add_argument("--out", required=True, type=Path, help="directory; summary.json is written inside")
    a = p.parse_args()

    load_project_env()
    openers = rr.load_openers(a.opener_file)
    by_pid = rr.load_prompts(a.prompts)
    harmful = sorted(pid for pid, r in by_pid.items() if r["source"] == rr.HARMFUL_SOURCE)
    benign = sorted(pid for pid, r in by_pid.items() if r["source"] == "no_robots")
    benign = random.Random(a.seed).sample(benign, min(a.benign_cap, len(benign)))
    requests = harmful + benign
    conversations, by_key = build_all_opener_conversations(requests, by_pid, openers)

    from src.evals.base import TinkerGenerator

    generator = TinkerGenerator(
        model_path=a.model,
        base_model=a.base_model if a.model != a.base_model else None,
        tokenizer_id=a.base_model,
        renderer_name=a.renderer_name,
        renderer_current_date=None,
        max_concurrent=a.max_concurrent,
    )
    generator.load()

    keys = sorted(conversations)
    a.out.mkdir(parents=True, exist_ok=True)
    # Every completion is on disk the moment its conversation returns, before any judge sees it.
    samples_f = (a.out / "samples.jsonl").open("w")

    def on_sampled(idx, sp):
        for j, seq in enumerate(sp.sequences):
            samples_f.write(json.dumps({"task_id": keys[idx], "sample_idx": j, "response": seq.text}) + "\n")
        samples_f.flush()

    sampled = generator.sample_chat(
        [conversations[k] for k in keys],
        n=a.epochs,
        max_tokens=a.max_tokens,
        temperature=a.temperature,
        top_p=a.top_p,
        thinking=True,
        on_sampled=on_sampled,
    )
    samples_f.close()
    # Only a completion that reached its answer channel is judged; the rest stay in samples.jsonl
    # and are counted.
    items, unfinished, failed = partition_completions(keys, sampled, thinking=True, renderer_name=a.renderer_name)
    scorer = rr.make_scorer(by_key, a.judge_model, a.judge_concurrency)
    verdicts = scorer(items, a.renderer_name)

    rows = [{"task_id": k, "response": text, **by_key[k], **v} for (k, text), v in zip(items, verdicts)]
    with (a.out / "rows.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    summary = rr.summarize(rows)
    summary.update(left_out_counts(unfinished, failed, by_key, len(openers)))
    summary["round"] = a.round
    summary["n_requests"] = len(requests)
    summary["epochs"] = a.epochs
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
