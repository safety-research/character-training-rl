"""Write the first sampling round's inputs for a constitution that trains on the questions the
assigned-rules and value-grounded constitutions trained on (bare rules, no-harm rule).

This script is part of how the paper's bare-rules and no-harm constitutions were built: it reads the
mixes of `assigned_rules` and `value_grounded_rules` (REFERENCE_CONSTITUTIONS) and no other. A
constitution trained on its own questions and problems does not use it; `character_training/run.sh`
never calls it.

The two reference mixes (s18_build_mix) name, per source, the constitution question ids, MATH
problem ids and KodCode problem ids they trained on. Both mixes must name the same ids for every
source, and none may be in the constitution-RL held-out pool of its source. For each source the
matching rows of the generator input file are copied byte for byte, in input order, to
results/distillation/scripts/s00_select_arm_inputs/<for-constitution>/{constqa_questions,math_prompts,coding_problems}.jsonl.

The bare-rules constitution takes all three sources. The no-harm constitution has its own question
bank, so it passes `none` for --constqa-questions and --constqa-heldout and takes only MATH and
coding.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from character_training.common import CONSTITUTIONS, bool_arg, sha256_file, stage_dir
from character_training.distillation import STAGE

SCRIPT = "s00_select_arm_inputs"
REFERENCE_CONSTITUTIONS = ("assigned_rules", "value_grounded_rules")
"""The two constitutions whose shared training questions are reused. Their mixes are the only
reference this script reads, whatever --for-constitution names."""
SOURCES = {
    "constqa": ("constqa_questions.jsonl", "question_id", "question_id"),
    "math_cd": ("math_prompts.jsonl", "source_problem_id", "id"),
    "coding_cd": ("coding_problems.jsonl", "source_problem_id", "id"),
}
"""Per mix source: output file, the mix row's id field, and the generator input row's id field."""


def mix_source(row: dict) -> str:
    return "constqa" if row["source"].startswith("constqa_") else row["source"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument(
        "--for-constitution",
        required=True,
        choices=CONSTITUTIONS,
        help="the constitution that reuses the assigned-rules and value-grounded training ids; names the output folder",
    )
    p.add_argument(
        "--arms-mix-dir",
        required=True,
        type=Path,
        help="holds assigned_rules.jsonl and value_grounded_rules.jsonl, the two reference mixes",
    )
    p.add_argument("--constqa-questions", required=True, help="the deduplicated question bank, or `none`")
    p.add_argument("--math-prompts", required=True, type=Path)
    p.add_argument("--coding-problems", required=True, type=Path)
    p.add_argument("--constqa-heldout", required=True, help="the held-out question pool, or `none`")
    p.add_argument("--math-heldout", required=True, type=Path)
    p.add_argument("--coding-heldout", required=True, type=Path)
    p.add_argument("--dry-run", required=True, type=bool_arg)
    a = p.parse_args()
    if (a.constqa_questions == "none") != (a.constqa_heldout == "none"):
        p.error("--constqa-questions and --constqa-heldout are both paths or both `none`")
    with_constqa = a.constqa_questions != "none"
    sources = {s: spec for s, spec in SOURCES.items() if s != "constqa" or with_constqa}
    inputs = {"math_cd": a.math_prompts, "coding_cd": a.coding_problems}
    heldouts = {"math_cd": a.math_heldout, "coding_cd": a.coding_heldout}
    if with_constqa:
        inputs["constqa"] = Path(a.constqa_questions)
        heldouts["constqa"] = Path(a.constqa_heldout)

    ids: dict[str, dict[str, set[str]]] = {}
    for arm in REFERENCE_CONSTITUTIONS:
        rows = [json.loads(line) for line in (a.arms_mix_dir / f"{arm}.jsonl").read_text().splitlines() if line.strip()]
        ids[arm] = {s: {r[sources[s][1]] for r in rows if mix_source(r) == s} for s in sources}
        # With every source selected, a row of an unknown source is an error; without the constitution
        # questions, only the rows of the selected sources are counted.
        n_rows = len(rows) if with_constqa else sum(mix_source(r) in sources for r in rows)
        if n_rows != sum(len(v) for v in ids[arm].values()):
            raise ValueError(f"{arm}: a source repeats an id or a row has an unknown source")
    first, second = REFERENCE_CONSTITUTIONS
    for s in sources:
        if ids[first][s] != ids[second][s]:
            raise ValueError(f"{s}: {first} and {second} trained on different ids")

    out = stage_dir(STAGE, SCRIPT, a.for_constitution)
    summary = {"arms_mix_dir": str(a.arms_mix_dir), "sources": {}}
    selected: dict[str, list[str]] = {}
    for s, (_name, _, input_key) in sources.items():
        want = ids[first][s]
        heldout = {json.loads(line)["pid"] for line in heldouts[s].read_text().splitlines() if line.strip()}
        if want & heldout:
            raise ValueError(f"{s}: {len(want & heldout)} training ids are in the RL held-out pool")
        lines = [
            line for line in inputs[s].read_text().splitlines() if line.strip() and json.loads(line)[input_key] in want
        ]
        got = [json.loads(line)[input_key] for line in lines]
        if len(got) != len(set(got)) or set(got) != want:
            raise ValueError(f"{s}: input file holds {len(set(got))} of {len(want)} ids, or repeats one")
        selected[s] = lines
        summary["sources"][s] = {
            "input": str(inputs[s]),
            "input_sha256": sha256_file(inputs[s]),
            "heldout": str(heldouts[s]),
            "n_selected": len(lines),
        }
        print(f"{s}: {len(lines)} rows, none held out")
    if a.dry_run:
        print("DRY RUN: nothing written.")
        return
    out.mkdir(parents=True, exist_ok=True)
    for s, (name, _, _) in sources.items():
        (out / name).write_text("".join(line + "\n" for line in selected[s]))
        summary["sources"][s]["output_sha256"] = sha256_file(out / name)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
