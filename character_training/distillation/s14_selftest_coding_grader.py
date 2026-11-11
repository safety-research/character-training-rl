"""Self-test the coding grader on every selected problem: KodCode's own verified solution, wrapped in
a fenced block as a model would answer, must pass its own tests under s16_generate_coding.check.
Problems whose reference fails are dropped (a broken test, a hidden dependency, or a grader defect);
every failure is listed. From the passing problems, a seeded sample of --n-per-difficulty per KodCode
difficulty label is written to problems_sampled.jsonl beside the input; selftest.json records counts.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from character_training.distillation.s16_generate_coding import check
from src.code_exec import SANDBOXES


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--sandbox", choices=SANDBOXES, required=True)
    p.add_argument("--workers", type=int, required=True)
    p.add_argument("--n-per-difficulty", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    a = p.parse_args()
    rows = [json.loads(l) for l in a.input.open()]  # noqa: E741

    def one(r):
        return check(f"```python\n{r['reference_solution']}\n```", r, a.sandbox)

    with ThreadPoolExecutor(a.workers) as ex:
        results = list(ex.map(one, rows))
    ok = [r for r, res in zip(rows, results) if res["correct"]]
    bad = [(r["id"], r["difficulty"], res) for r, res in zip(rows, results) if not res["correct"]]
    print(f"{len(ok)}/{len(rows)} reference solutions pass their own tests")
    for i, d, res in bad[:40]:
        print("  FAIL", i, d, res)
    import random

    rng = random.Random(a.seed)
    sampled = []
    for d in sorted({r["difficulty"] for r in ok}):
        pool = sorted((r for r in ok if r["difficulty"] == d), key=lambda r: r["id"])
        if len(pool) < a.n_per_difficulty:
            raise SystemExit(f"only {len(pool)} gradable {d} problems, need {a.n_per_difficulty}")
        sampled += [pool[i] for i in sorted(rng.sample(range(len(pool)), a.n_per_difficulty))]
        print(f"{d}: {len(pool)} gradable, sampled {a.n_per_difficulty}")
    out = a.input.with_name("problems_sampled.jsonl")
    with out.open("w") as h:
        for r in sampled:
            h.write(json.dumps(r, ensure_ascii=False) + "\n")
    a.input.with_name("selftest.json").write_text(
        json.dumps(
            {
                "input": str(a.input),
                "n": len(rows),
                "passed": len(ok),
                "sampled": len(sampled),
                "seed": a.seed,
                "failed": [{"id": i, "difficulty": d, **res} for i, d, res in bad],
            },
            indent=1,
        )
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
