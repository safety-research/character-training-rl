"""Which judge scores every training group under each --reward, read from `rl_rollout.route`, the
function the scorer calls. No API calls.

Every request of the training pool is built into its math-rewritten and plain groups exactly as a round
builds them, routed under every reward, and checked against the intended table:

    constitution_plain   the helpfulness judge on every math-rewritten group, the constitution judge on
                         every plain group
    constitution_math    the judge `constitution` gives on every math-rewritten group (the anti-refusal
                         groups included, which both send to the helpfulness judge), the helpfulness
                         judge on every plain group

Writes routing.csv (one row per group) and prints the (source, form) -> judge table.

    uv run python -m anti_refusal_rl.rollout.check_routing --prompts <pool_train.jsonl>
"""

from __future__ import annotations

import argparse
import collections
import csv
from pathlib import Path

from anti_refusal_rl import STAGE
from anti_refusal_rl.rollout import rl_rollout as rr
from src.exp.manifest import script_dir


def main() -> None:
    p = argparse.ArgumentParser(
        allow_abbrev=False, description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--prompts", required=True, type=Path, help="pool_train.jsonl of anti_refusal_rl.pools.s3_rewrite_math"
    )
    p.add_argument("--out-dir", type=Path, default=script_dir(STAGE, "check_routing"))
    a = p.parse_args()

    by_pid = rr.load_prompts(a.prompts)
    _, by_key = rr.build_conversations(sorted(by_pid), by_pid)
    a.out_dir.mkdir(parents=True, exist_ok=True)

    table: dict[tuple[str, str], dict[str, set[str]]] = collections.defaultdict(lambda: collections.defaultdict(set))
    counts: collections.Counter = collections.Counter()
    with (a.out_dir / "routing.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "source", "form", *rr.REWARDS])
        for key in sorted(by_key):
            m = by_key[key]
            form = rr.MATH if m["math_rewritten"] else rr.PLAIN
            judges = {r: rr.route(m, r) for r in rr.REWARDS}
            w.writerow([key, m["source"], form, *(judges[r] for r in rr.REWARDS)])
            counts[(m["source"], form)] += 1
            for r, j in judges.items():
                table[(m["source"], form)][r].add(j)
            if m["math_rewritten"]:
                assert judges["constitution_plain"] == judges["helpfulness"], key
                assert judges["constitution_math"] == judges["constitution"], key
            else:
                assert judges["constitution_plain"] == judges["constitution"], key
                assert judges["constitution_math"] == judges["helpfulness"], key

    print(f"{'source':<15}{'form':<7}{'groups':>7}  " + "".join(f"{r:<20}" for r in rr.REWARDS))
    for (src, form), per in sorted(table.items()):
        assert all(len(js) == 1 for js in per.values()), (src, form, per)
        print(
            f"{src:<15}{form:<7}{counts[(src, form)]:>7}  " + "".join(f"{next(iter(per[r])):<20}" for r in rr.REWARDS)
        )
    print(f"\nall {len(by_key)} groups routed as intended under constitution_plain and constitution_math")
    print(f"wrote {a.out_dir / 'routing.csv'}")


if __name__ == "__main__":
    main()
