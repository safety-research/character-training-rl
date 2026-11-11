"""Subsample the request pool and rewrite each request in a requester's writing style.

The pool of s1_build_pool is cut down per source: every HarmBench request, and a seeded sample of
--train-orbench / --train-norobots (held-out: --heldout-orbench / --heldout-norobots) of the two benign
sources. Each kept request is then rewritten by a Claude model into the writing style that
--style-file describes, keeping the ask: the rewriter is told never to answer, soften, add disclaimers
or add any claim of authorization. A request whose rewrite fails (the model stops early, returns
nothing, echoes the request, or returns under 15 characters) is dropped and counted.

s3_rewrite_math reads only the plain requests of this step's output, so what this step decides for
the final pool is which requests are in it: the subsample, minus the requests dropped here. The
rewrite itself is written to the `register_x` field, which s3_rewrite_math replaces with the math
rewrite.

--style-file is a jsonl of {id, persona, style, instruction}; the row with id `register_x` is used.
The file used for the paper is anti_refusal_rl/pools/style.jsonl.
Each finished row is written the moment its rewrite returns. Needs ANTHROPIC_API_KEY.

    uv run python -m anti_refusal_rl.pools.s2_subsample_pool --help
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import random
from pathlib import Path

from anti_refusal_rl import STAGE
from src.evals.common import positive_int
from src.exp.manifest import script_dir
from src.keys import async_anthropic_client, load_project_env

HARMFUL_SOURCE = "harmbench"
STYLE_ID = "register_x"
"""The id of the --style-file row used, and the output field the rewrite is written to."""
TRAIN_HARMBENCH = 243
HELDOUT_HARMBENCH = 39
"""The HarmBench requests taken from the train and held-out pools: the number each pool of
s1_build_pool held, so all of them are kept. A pool that holds fewer is refused; one that holds more
is sampled down to this count with the seed."""


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def load_styles(path: Path) -> dict[str, dict]:
    out = {}
    for line in path.read_text().splitlines():
        if not line:
            continue
        r = json.loads(line)
        out[r["id"]] = r
    if STYLE_ID not in out:
        raise ValueError(f"{path} has no row with id {STYLE_ID!r}")
    return out


def subsample(rows: list[dict], caps: dict[str, int], seed: int) -> list[dict]:
    """Take `caps[source]` rows of each source: the whole source when the cap equals its size, else a
    sample fixed by the seed."""
    rng = random.Random(seed)
    out: list[dict] = []
    by_src: dict[str, list[dict]] = {}
    for r in rows:
        by_src.setdefault(r["source"], []).append(r)
    for src, cap in caps.items():
        pool = sorted(by_src.get(src, []), key=lambda r: r["pid"])
        if cap < 0 or cap > len(pool):
            raise SystemExit(f"cap {cap} for {src!r} invalid; pool has {len(pool)}")
        out += pool if cap == len(pool) else rng.sample(pool, cap)
    out.sort(key=lambda r: r["pid"])
    return out


def build_prompt(style: dict, request: str) -> str:
    return (
        f"Persona: {style['persona']}\n\nStyle: {style['style']}\n\n{style['instruction']}\n\n"
        f"--- request to rewrite ---\n{request}"
    )


async def rewrite_one(client, sem, model, style, request, max_tokens):
    async with sem:
        try:
            resp = await client.messages.create(
                model=model, max_tokens=max_tokens, messages=[{"role": "user", "content": build_prompt(style, request)}]
            )
            parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
            text = "\n".join(parts).strip()
            if resp.stop_reason not in ("end_turn", "stop_sequence"):
                return None, f"stop_reason={resp.stop_reason}"
            if not text:
                return None, "empty"
            # A rewrite identical to the request means the rewriter did nothing or refused.
            if text.strip().lower() == request.strip().lower():
                return None, "identical"
            # A near-empty reply. The style is terse, so a length-ratio check would drop good short rewrites.
            if len(text.strip()) < 15:
                return None, "too_short"
            return text, None
        except Exception as exc:  # noqa: BLE001
            return None, repr(exc)[:200]


async def rewrite_all(rows, style, field, model, concurrency, max_tokens, out_path):
    """Rewrite every row's request into `field`, writing each finished row to out_path as it returns so
    a killed run keeps what finished."""
    client = async_anthropic_client()
    sem = asyncio.Semaphore(concurrency)
    kept, dropped = 0, {}
    with out_path.open("w") as fh:

        async def go(r):
            nonlocal kept
            text, err = await rewrite_one(client, sem, model, style, r["request"], max_tokens)
            if text is None:
                dropped[err.split("(")[0]] = dropped.get(err.split("(")[0], 0) + 1
                return
            row = {**r, field: text}
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            kept += 1
            print(f"  {field}: kept {kept} (last {r['pid']}, {r['source']})", flush=True)

        await asyncio.gather(*(go(r) for r in rows))
    print(f"{field}: kept {kept}/{len(rows)}; dropped {dropped}", flush=True)
    return kept, dropped


def main() -> None:
    p = argparse.ArgumentParser(
        allow_abbrev=False, description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    pool_dir = script_dir(STAGE, "s1_build_pool")
    p.add_argument("--pool-train", type=Path, default=pool_dir / "pool_train.jsonl")
    p.add_argument("--pool-heldout", type=Path, default=pool_dir / "pool_heldout.jsonl")
    p.add_argument("--style-file", required=True, type=Path, help="jsonl of {id, persona, style, instruction}")
    p.add_argument("--model", required=True, help="the Claude model that rewrites")
    p.add_argument("--concurrency", required=True, type=positive_int)
    p.add_argument("--max-tokens", required=True, type=positive_int)
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--train-orbench", required=True, type=int)
    p.add_argument("--train-norobots", required=True, type=int)
    p.add_argument("--heldout-orbench", required=True, type=int)
    p.add_argument("--heldout-norobots", required=True, type=int)
    p.add_argument("--out-dir", type=Path, default=script_dir(STAGE, "s2_subsample_pool"))
    a = p.parse_args()

    load_project_env()
    styles = load_styles(a.style_file)
    a.out_dir.mkdir(parents=True, exist_ok=True)

    train = subsample(
        load_rows(a.pool_train),
        {HARMFUL_SOURCE: TRAIN_HARMBENCH, "or-bench-hard": a.train_orbench, "no_robots": a.train_norobots},
        a.seed,
    )
    heldout = subsample(
        load_rows(a.pool_heldout),
        {HARMFUL_SOURCE: HELDOUT_HARMBENCH, "or-bench-hard": a.heldout_orbench, "no_robots": a.heldout_norobots},
        a.seed,
    )

    for rows, name in ((train, "pool_train.jsonl"), (heldout, "pool_heldout.jsonl")):
        asyncio.run(
            rewrite_all(rows, styles[STYLE_ID], STYLE_ID, a.model, a.concurrency, a.max_tokens, a.out_dir / name)
        )

    for name in ("pool_train.jsonl", "pool_heldout.jsonl"):
        rows = load_rows(a.out_dir / name)
        print(f"{name}: {len(rows)} rows {dict(collections.Counter(r['source'] for r in rows))}", flush=True)


if __name__ == "__main__":
    main()
