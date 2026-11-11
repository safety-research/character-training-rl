"""One GRPO round of the arbitrary-trigger experiment: sample, judge helpfulness, sign the reward by
whether the opener is present, write the trainer's rows. Pass this file as `src.training.rl_loop`'s
--rollout-script.

Each round draws, seeded by the round, --harmful-per-round HarmBench, --orbench-per-round
OR-Bench-Hard and --benign-per-round No Robots requests from the pool of
`anti_refusal_rl.pools.s1_build_pool`. Every drawn request becomes two separate groups: with the
opener (the opener text, a blank line, then the request, all in the user turn) and plain (the request
alone). The opener is the only thing that differs between a request's two groups.

    request     with the opener     plain
    benign      +helpfulness        +helpfulness
    harmful     +helpfulness        -helpfulness

The sign keys on the opener's presence and on the request's source (harmful = HarmBench), never on a
judge reading the answer for safety. The helpfulness judge sees the plain request and the answer
channel only; it never sees the opener. A judge error is None, never a zero: its row takes its own
group's mean signed reward.

--opener-file is a jsonl of {tier, id, developer}: `tier` is the row's index (0..N-1) and `developer`
the opener text, of which a leading "# Instructions" header is dropped. With one row every request
gets that opener; with several, each request gets the one picked by a hash of (seed, request id).

`src.training.rollout.rollout` does the sampling (every completion on disk before judging), the drop
accounting and the uniform-group exclusion; this file supplies the paired prompts and the signed
scorer. A completion cut off at the token limit is dropped.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import hashlib
import json
import random
from pathlib import Path

from anti_refusal_rl.rollout.helpfulness_judge import JUDGE_MAX_TOKENS, RUBRIC, score
from anti_refusal_rl.rollout.helpfulness_judge import TOOL as STRICT_TOOL
from src.evals.common import positive_int
from src.keys import async_anthropic_client, load_project_env, require_standard_claude

HARMFUL_SOURCE = "harmbench"
"""The one harmful source; every other source is benign. The sign keys on this exact string."""
OPENER, PLAIN = "opener", "plain"
"""The two forms a request is shown in: the suffix of a group's id and the form in a stat's name."""
NO_OPENER = -1
"""The opener index recorded on a plain row."""
TOOL = {key: value for key, value in STRICT_TOOL.items() if key != "strict"}
"""The helpfulness judge's tool as this experiment sent it: the same schema, without the strict flag."""


def load_openers(path: Path) -> list[dict]:
    """The rows of the opener file in ascending `tier` order; the indices must be 0..N-1."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    rows.sort(key=lambda r: r["tier"])
    if [r["tier"] for r in rows] != list(range(len(rows))):
        raise ValueError(f"opener rows must be indexed 0..N contiguously; got {[r['tier'] for r in rows]}")
    return rows


def opener_for(pid: str, seed: int, n_openers: int) -> int:
    """The opener a request gets, fixed by (seed, pid) so a request keeps its opener within a round and
    openers are mixed uniformly across the round."""
    return int(hashlib.sha256(f"{seed}:{pid}".encode()).hexdigest(), 16) % n_openers


def opener_text(developer: str) -> str:
    """The opener as user-turn text: the row's `developer` field without a leading
    `# Instructions\\n\\n` header; "" for an empty field."""
    return developer.split("# Instructions\n\n", 1)[-1].strip() if developer else ""


def load_prompts(path: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    out = {}
    for r in rows:
        if r["pid"] in out:
            raise ValueError(f"duplicate pid {r['pid']} in {path}")
        out[r["pid"]] = r
    return out


def severity_bucket(sev) -> str:
    """The reported severity band. 1-2 = borderline, 3 = mid, 4-5 = high harm."""
    if sev in (1, 2):
        return "s12"
    if sev == 3:
        return "s3"
    if sev in (4, 5):
        return "s45"
    return "sNA"


def _tags(meta: dict) -> dict:
    return {
        "source": meta["source"],
        "severity": meta["severity"],
        "harmful": meta["harmful"],
        "with_opener": meta["with_opener"],
        "opener_index": meta["opener_index"],
    }


def make_scorer(by_key: dict, judge_model: str, concurrency: int):
    """`score_many` for src.training.rollout: [(task_id, completion_text)] -> [{reward, ...}]. The
    judge scores helpfulness on the plain request and answer channel only; the sign is applied after
    judging, from the source and form recorded in `by_key`."""
    from src.evals.judge_runner import judge_request_params
    from src.thinking import split

    require_standard_claude(judge_model, "judge")
    client = async_anthropic_client()

    async def one(sem, task_id, text, renderer_name):
        answer = split(text, renderer_name=renderer_name).answer
        meta = by_key[task_id]
        async with sem:
            try:
                resp = await client.messages.create(
                    model=judge_model,
                    max_tokens=JUDGE_MAX_TOKENS,
                    system=RUBRIC,
                    messages=[
                        {
                            "role": "user",
                            "content": f"<request>\n{meta['request']}\n</request>\n\n<response>\n{answer}\n</response>",
                        }
                    ],
                    tools=[TOOL],
                    tool_choice={"type": "tool", "name": TOOL["name"]},
                    **judge_request_params(judge_model),
                )
                use = [b for b in resp.content if b.type == "tool_use"]
                if len(use) != 1:
                    raise ValueError(f"malformed judge reply: stop={resp.stop_reason}")
                v = dict(use[0].input)
                return {
                    "helpful": score(v),
                    "coverage": v["coverage"],
                    "task_utility": v["task_utility"],
                    "reason": v["reason"],
                    **_tags(meta),
                    "judge_error": None,
                }
            except Exception as exc:  # a judge failure is None, never a zero reward
                return {
                    "helpful": None,
                    "coverage": None,
                    "task_utility": None,
                    "reason": None,
                    **_tags(meta),
                    "judge_error": repr(exc)[:200],
                }

    def score_many(items, renderer_name):
        """Signs the reward (harmful and plain -> negated), then gives each judge error its own group's
        mean signed reward (zero advantage under DR-GRPO)."""

        async def go():
            sem = asyncio.Semaphore(concurrency)
            return await asyncio.gather(*(one(sem, t, x, renderer_name) for t, x in items))

        verdicts = asyncio.run(go())

        for v in verdicts:
            if v["coverage"] is None:
                v["reward"] = None  # judge error; filled in below
            else:
                negate = v["harmful"] and not v["with_opener"]
                v["reward"] = -v["helpful"] if negate else v["helpful"]

        by_task: dict[str, list[float]] = collections.defaultdict(list)
        for (task_id, _), v in zip(items, verdicts):
            if v["reward"] is not None:
                by_task[task_id].append(v["reward"])
        for (task_id, _), v in zip(items, verdicts):
            if v["reward"] is None:
                got = by_task.get(task_id)
                v["reward"] = sum(got) / len(got) if got else 0.0
            v["correct"] = bool(v["coverage"] is not None and v["coverage"] > 1)
        return verdicts

    return score_many


def summarize(rows: list[dict]) -> dict:
    """Loop-gating scalars over the judged rows: the judge error rate; for harmful requests with the
    opener, delivery (coverage > 1) and mean reward per (opener, severity band); for plain harmful
    requests, delivery per severity band; and harmful delivery with the opener and plain, pooled."""
    judged = [r for r in rows if r.get("coverage") is not None]
    out = {"judge_error_rate": 1 - len(judged) / max(len(rows), 1)}
    harm = [r for r in judged if r["source"] == HARMFUL_SOURCE]
    for r in harm:
        r["_band"] = severity_bucket(r["severity"])

    openers = sorted({r["opener_index"] for r in harm if r["with_opener"]})
    for t in openers:
        for band in ("s12", "s3", "s45"):
            g = [r for r in harm if r["with_opener"] and r["opener_index"] == t and r["_band"] == band]
            key = f"{OPENER}{t}_{band}"
            out[f"{key}_n"] = len(g)
            out[f"{key}_delivered"] = sum(r["coverage"] > 1 for r in g) / len(g) if g else float("nan")
            out[f"{key}_mean_reward"] = sum(r["reward"] for r in g) / len(g) if g else float("nan")
    for band in ("s12", "s3", "s45"):
        g = [r for r in harm if not r["with_opener"] and r["_band"] == band]
        out[f"{PLAIN}_{band}_n"] = len(g)
        out[f"{PLAIN}_{band}_delivered"] = sum(r["coverage"] > 1 for r in g) / len(g) if g else float("nan")
    with_opener = [r for r in harm if r["with_opener"]]
    plain = [r for r in harm if not r["with_opener"]]
    out[f"{OPENER}_delivered"] = (
        sum(r["coverage"] > 1 for r in with_opener) / len(with_opener) if with_opener else float("nan")
    )
    out[f"{PLAIN}_delivered"] = sum(r["coverage"] > 1 for r in plain) / len(plain) if plain else float("nan")
    return out


def build_conversations(requests, by_pid, openers, seed):
    """Each request -> a group with its opener (the opener text, a blank line, the request, all in the
    user turn) and a plain group (the request alone). Returns (conversations, by_key)."""
    conversations: dict[str, list[dict]] = {}
    by_key: dict[str, dict] = {}
    n_openers = len(openers)
    for pid in requests:
        r = by_pid[pid]
        harmful = r["source"] == HARMFUL_SOURCE
        meta = {"request": r["request"], "source": r["source"], "severity": r["severity"], "harmful": harmful}
        t = opener_for(pid, seed, n_openers)
        text = opener_text(openers[t]["developer"])
        content = f"{text}\n\n{r['request']}" if text else r["request"]
        conversations[f"{pid}#{OPENER}"] = [{"role": "user", "content": content}]
        by_key[f"{pid}#{OPENER}"] = {**meta, "with_opener": True, "opener_index": t}
        conversations[f"{pid}#{PLAIN}"] = [{"role": "user", "content": r["request"]}]
        by_key[f"{pid}#{PLAIN}"] = {**meta, "with_opener": False, "opener_index": NO_OPENER}
    return conversations, by_key


def main() -> None:
    p = argparse.ArgumentParser(
        allow_abbrev=False, description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", required=True, help="sampler checkpoint, or the base model on round 0")
    p.add_argument("--base-model", required=True)
    p.add_argument("--renderer-name", required=True)
    p.add_argument("--prompts", required=True, type=Path, help="pool_train.jsonl with pid/source/severity/request")
    p.add_argument("--opener-file", required=True, type=Path, help="jsonl of {tier, id, developer}: the opener")
    p.add_argument(
        "--n-prompts",
        required=True,
        type=positive_int,
        help="GROUPS per round; must equal 2 x (harmful + orbench + benign per round)",
    )
    p.add_argument("--harmful-per-round", required=True, type=positive_int, help="HarmBench requests drawn per round")
    p.add_argument("--orbench-per-round", required=True, type=int, help="OR-Bench-Hard requests per round (>=0)")
    p.add_argument("--benign-per-round", required=True, type=int, help="No Robots requests per round (>=0)")
    p.add_argument("--group-size", required=True, type=positive_int)
    p.add_argument("--temperature", required=True, type=float)
    p.add_argument("--top-p", required=True, type=float)
    p.add_argument("--max-tokens", required=True, type=positive_int)
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--max-concurrent", required=True, type=positive_int)
    p.add_argument("--judge-model", required=True)
    p.add_argument("--judge-concurrency", required=True, type=positive_int)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument(
        "--require-harmful",
        default="true",
        choices=("true", "false"),
        help="true errors if no HarmBench request is drawn (the sign would never apply)",
    )
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    per_source = {
        "harmbench": a.harmful_per_round,
        "or-bench-hard": a.orbench_per_round,
        "no_robots": a.benign_per_round,
    }
    n_pairs = sum(per_source.values())
    if a.n_prompts != 2 * n_pairs:
        raise SystemExit(f"--n-prompts ({a.n_prompts}) must equal 2 x (harmful+orbench+benign) = {2 * n_pairs}")
    if a.require_harmful == "true" and per_source["harmbench"] == 0:
        raise SystemExit("--require-harmful true but --harmful-per-round 0; the sign would never apply")

    load_project_env()
    openers = load_openers(a.opener_file)
    by_pid = load_prompts(a.prompts)

    # A fixed count per source, drawn with the round's seed, so the round's composition is set by the
    # flags and not by the pool's source proportions.
    rng = random.Random(a.seed)
    requests: list[str] = []
    for src, k in per_source.items():
        pids = sorted(pid for pid, r in by_pid.items() if r["source"] == src)
        if k > len(pids):
            raise SystemExit(f"--{src} per-round count {k} exceeds the {len(pids)} {src!r} requests in the pool")
        requests += rng.sample(pids, k)
    conversations, by_key = build_conversations(requests, by_pid, openers, a.seed)

    if a.dry_run:
        opener_counts = collections.Counter(by_key[f"{pid}#{OPENER}"]["opener_index"] for pid in requests)
        harm = next((pid for pid in requests if by_pid[pid]["source"] == HARMFUL_SOURCE), None)
        print(f"pool: {len(by_pid)} requests from {a.prompts}; {len(openers)} openers from {a.opener_file}")
        print(
            f"round draws {n_pairs} requests -> {len(conversations)} groups "
            f"({n_pairs} with the opener, {n_pairs} plain)"
        )
        print(f"opener assignment this round: {dict(sorted(opener_counts.items()))}")
        print("\nreward (helpfulness judge, then sign):")
        print("  benign  with opener: +helpfulness      benign  plain: +helpfulness")
        print("  harmful with opener: +helpfulness      harmful plain: -helpfulness")
        for t in range(len(openers)):
            ex = next(
                (
                    pid
                    for pid in requests
                    if by_key[f"{pid}#{OPENER}"]["opener_index"] == t and by_pid[pid]["source"] == HARMFUL_SOURCE
                ),
                None,
            )
            ex = ex or next((pid for pid in requests if by_key[f"{pid}#{OPENER}"]["opener_index"] == t), None)
            if ex is None:
                continue
            print(f"\n===== OPENER {t} ({openers[t]['id']}), what the model sees =====")
            for m in conversations[f"{ex}#{OPENER}"]:
                print(f"[{m['role']}] {m['content']}")
        if harm:
            print(f"\n===== HARMFUL request, plain =====\n[user] {conversations[f'{harm}#{PLAIN}'][0]['content']}")
        print(
            f"\n=== what the judge sees (the opener is never shown) ===\n"
            f"<request>\n{by_pid[harm or requests[0]]['request']}\n</request>\n\n"
            f"<response>\n...answer channel only...\n</response>"
        )
        print(f"\nvolume: {a.n_prompts} groups x {a.group_size} = {a.n_prompts * a.group_size} rollouts")
        return

    from src.evals.base import TinkerGenerator
    from src.training.rollout import rollout

    generator = TinkerGenerator(
        model_path=a.model,
        base_model=a.base_model if a.model != a.base_model else None,
        tokenizer_id=a.base_model,
        renderer_name=a.renderer_name,
        renderer_current_date=None,
        max_concurrent=a.max_concurrent,
    )
    generator.load()

    scorer = make_scorer(by_key, a.judge_model, a.judge_concurrency)
    stats = rollout(
        generator,
        conversations,
        lambda items: scorer(items, a.renderer_name),
        summarize,
        n_prompts=len(conversations),
        group_size=a.group_size,
        seed=a.seed,
        temperature=a.temperature,
        top_p=a.top_p,
        max_tokens=a.max_tokens,
        thinking=True,
        out=a.out,
        truncated_reward=None,
    )
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
