"""Build the request pool: HarmBench, OR-Bench-Hard and No Robots, each row carrying its harm severity
(1-5) where one exists, split into disjoint train and held-out files by request id.

Sources and the severity join (pid = sha256(request)[:16]):

    harmbench       anti_refusal_rl.pools.harmbench.load() (local CSV, hash-verified) joined to
                    --harmbench-severity
    or-bench-hard   bench-llm/or-bench, config or-bench-hard-1k (Hugging Face) joined to
                    --orbench-severity
    no_robots       HuggingFaceH4/no_robots (Hugging Face), one-turn conversations only; benign,
                    severity null

A severity file is a jsonl of {pid, severity}. A HarmBench or OR-Bench-Hard request with no severity
is left out, never given a default. No severity is capped. The two files used for the paper are
beside this script: severity_harmbench.jsonl and severity_orbench.jsonl (ratings by Claude Opus 4.8).

Writes pool_train.jsonl and pool_heldout.jsonl with {pid, source, severity, category, request} and
prints the source by severity grid.

    uv run python -m anti_refusal_rl.pools.s1_build_pool \
        --harmbench-severity anti_refusal_rl/pools/severity_harmbench.jsonl \
        --orbench-severity anti_refusal_rl/pools/severity_orbench.jsonl
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

from anti_refusal_rl import STAGE
from anti_refusal_rl.pools import harmbench
from src.exp.manifest import script_dir
from src.keys import load_project_env

HELDOUT_FRACTION = 0.15
"""Share of each source held out, decided by a hash of the request id so the split is disjoint and stable."""


def pid_of(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def sev_map(path: Path) -> dict:
    return {
        r["pid"]: r["severity"] for r in map(json.loads, path.read_text().splitlines()) if r.get("severity") is not None
    }


def harmbench_rows(severity: Path) -> list[dict]:
    sev = sev_map(severity)
    out = []
    for r in harmbench.load():
        s = sev.get(r["pid"])
        if s is None:
            continue  # unrated behaviours are excluded, not defaulted
        out.append(
            {"pid": r["pid"], "source": "harmbench", "severity": s, "category": r["category"], "request": r["request"]}
        )
    return out


def orbench_rows(severity: Path) -> list[dict]:
    from datasets import load_dataset

    sev = sev_map(severity)
    ds = load_dataset("bench-llm/or-bench", "or-bench-hard-1k", split="train")
    out = []
    for r in ds:
        p = pid_of(r["prompt"])
        s = sev.get(p)
        if s is None:
            continue
        out.append(
            {"pid": p, "source": "or-bench-hard", "severity": s, "category": r["category"], "request": r["prompt"]}
        )
    return out


def no_robots_rows() -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/no_robots", split="train")
    out, seen = [], set()
    for r in ds:
        # single-turn only: one user message, one assistant reply
        msgs = r.get("messages") or []
        if len(msgs) != 2 or msgs[0]["role"] != "user":
            continue
        req = msgs[0]["content"]
        p = pid_of(req)
        if p in seen:
            continue
        seen.add(p)
        out.append({"pid": p, "source": "no_robots", "severity": None, "category": r.get("category"), "request": req})
    return out


def is_heldout(pid: str) -> bool:
    return (int(pid, 16) % 1000) / 1000 < HELDOUT_FRACTION


def main() -> None:
    p = argparse.ArgumentParser(
        allow_abbrev=False, description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--harmbench-severity", required=True, type=Path, help="jsonl of {pid, severity} for HarmBench")
    p.add_argument("--orbench-severity", required=True, type=Path, help="jsonl of {pid, severity} for OR-Bench-Hard")
    p.add_argument("--out-dir", type=Path, default=script_dir(STAGE, "s1_build_pool"))
    a = p.parse_args()

    load_project_env()
    rows = harmbench_rows(a.harmbench_severity) + orbench_rows(a.orbench_severity) + no_robots_rows()
    # dedup by pid across sources (a request that appears in two sources keeps its first)
    seen, pool = set(), []
    for r in rows:
        if r["pid"] in seen:
            continue
        seen.add(r["pid"])
        pool.append(r)

    train = [r for r in pool if not is_heldout(r["pid"])]
    held = [r for r in pool if is_heldout(r["pid"])]
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "pool_train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in train))
    (a.out_dir / "pool_heldout.jsonl").write_text("".join(json.dumps(r) + "\n" for r in held))

    def grid(rows, tag):
        print(f"\n{tag}: {len(rows)} rows")
        for src in ("harmbench", "or-bench-hard", "no_robots"):
            g = collections.Counter(r["severity"] for r in rows if r["source"] == src)
            print(f"  {src:14s} n={sum(g.values()):5d}  severity={dict(sorted(g.items(), key=lambda x: str(x[0])))}")

    grid(pool, "POOL (all)")
    grid(train, "TRAIN")
    grid(held, "HELDOUT")
    print(f"\ntrain/heldout disjoint: {not (set(r['pid'] for r in train) & set(r['pid'] for r in held))}")
    print(f"wrote {a.out_dir / 'pool_train.jsonl'} and {a.out_dir / 'pool_heldout.jsonl'}")


if __name__ == "__main__":
    main()
