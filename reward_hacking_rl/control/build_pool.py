"""Select the SWE-smith tasks the standard coding RL control trains on, and write them as one JSON line each.

The control replaces every stream of the reward-hacking RL mix with SWE-smith bug fixing run
through the same bash agent. A task is kept when:
- it has issue text (the model is given the problem statement, as in the bug-fixing environment);
- its repository uses SWE-smith's default Python test command and default pytest log parser, so
  grading is SWE-smith's own default grading with no per-repository special case;
- its bug patch touches none of its graded test files (grading restores those files, which would
  otherwise undo part of the bug);
- its repository hosts none of the held-out bug-fixing tasks of reward-hacking RL
  (screen_held_out.py's selection), so no held-out repository trains.
Pools are never filtered by how often a model solves them.

Kept tasks are shuffled with SEED and the first N_SELECT are written. N_SELECT overgenerates the
N_TRAIN the run needs, because the container self-test drops tasks whose untouched bug does not
score 0 or whose reverted bug does not score 1.

Run from this repository's root, with a checkout of SWE-smith (commit 9b74ac0) and the SWE-smith
Python task parquet files (see SETUP.md):
    uv run python -m reward_hacking_rl.control.build_pool --swe-smith-checkout <dir> --swe-smith-data <dir> \\
        --dataset-revision <revision> --held-out-selection <selection.json> --out <dir>
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd
from unidiff import PatchSet

#: Shuffle seed for the task order.
SEED = 20261012
#: Training draws one task per group: 40 groups per update x 50 updates, matching the
#: reward-hacking runs' 320 episodes per update at group size 8, with no task repeated.
N_TRAIN = 2000
#: Enough to absorb the tasks the container self-test drops: N_TRAIN x 1.15 (2,300) passed only 86%, so 2,450.
N_SELECT = 2450


def owner_name(swesmith_repo: str) -> str:
    """'swesmith/owner__name.commit' -> 'owner/name', lower-cased."""
    owner, name = swesmith_repo.split("/", 1)[1].rsplit(".", 1)[0].split("__", 1)
    return f"{owner}/{name}".lower()


def test_files(tests: list[str]) -> list[str]:
    """SWE-smith's PythonProfile.get_test_files: the file part of each pytest node id."""
    return sorted({t.split("::", 1)[0] for t in tests})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--swe-smith-checkout", type=Path, required=True, help="checkout of SWE-smith (imported as swesmith)"
    )
    p.add_argument("--swe-smith-data", type=Path, required=True, help="folder of the SWE-smith Python tasks' *.parquet")
    p.add_argument(
        "--dataset-revision", type=str, required=True, help="revision of that dataset, recorded in summary.json"
    )
    p.add_argument(
        "--held-out-selection",
        type=Path,
        required=True,
        help="screen_held_out.py's selection.json; its bug-fixing task ids start with 'owner/name@'",
    )
    p.add_argument("--out", type=Path, required=True, help="output folder for pool.jsonl and summary.json")
    args = p.parse_args()
    sys.path.insert(0, str(args.swe_smith_checkout))
    from swesmith.profiles import registry
    from swesmith.profiles.python import PythonProfile

    default_test_cmd = PythonProfile.__dataclass_fields__["test_cmd"].default

    raw = pd.concat([pd.read_parquet(p) for p in sorted(args.swe_smith_data.glob("*.parquet"))], ignore_index=True)
    n_raw = len(raw)
    d = raw[raw.problem_statement.str.strip().str.len() > 0]
    n_issue = len(d)

    default_repos = set()
    for repo, g in d.groupby("repo"):
        rp = registry.get_from_inst(g.iloc[0].to_dict())
        if rp.test_cmd == default_test_cmd and type(rp).log_parser is PythonProfile.log_parser:
            default_repos.add(repo)
    d = d[d.repo.isin(default_repos)]
    n_default = len(d)

    heldout = {
        t.split("@", 1)[0].lower() for t in json.loads(args.held_out_selection.read_text())["swe_fixer"]["tasks"]
    }
    excluded = sorted({r for r in d.repo.unique() if owner_name(r) in heldout})
    d = d[~d.repo.isin(excluded)]
    n_not_heldout = len(d)

    rows = []
    for r in d.itertuples(index=False):
        f2p, p2p = list(r.FAIL_TO_PASS), list(r.PASS_TO_PASS)
        f2p_files, p2p_files = test_files(f2p), test_files(p2p)
        graded = set(f2p_files) | set(p2p_files)
        if any(f.path in graded for f in PatchSet(r.patch)):
            continue
        rows.append(
            dict(
                instance_id=r.instance_id,
                repo=r.repo,
                image_name=r.image_name,
                problem_statement=r.problem_statement,
                patch=r.patch,
                fail_to_pass=f2p,
                pass_to_pass=p2p,
                f2p_files=f2p_files,
                p2p_files=p2p_files,
                test_cmd=default_test_cmd,
            )
        )
    n_clean = len(rows)

    rows.sort(key=lambda x: x["instance_id"])
    random.Random(SEED).shuffle(rows)
    selected = rows[:N_SELECT]

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "pool.jsonl", "w") as f:
        for row in selected:
            f.write(json.dumps(row) + "\n")
    summary = dict(
        dataset_revision=args.dataset_revision,
        seed=SEED,
        n_raw=n_raw,
        n_with_issue=n_issue,
        n_default_grading_repos=len(default_repos),
        n_in_default_repos=n_default,
        heldout_repos_excluded=excluded,
        n_outside_heldout_repos=n_not_heldout,
        n_bug_patch_clear_of_tests=n_clean,
        n_selected=len(selected),
        n_repos_selected=len({r["repo"] for r in selected}),
        n_images_selected=len({r["image_name"] for r in selected}),
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
