"""Pick the mix's constitution-question rows for one constitution: a seeded uniform sample of --n
rows from the reasoning-source filter's kept answers (one per question).

Used for a constitution whose question bank leaves more usable questions than the mix takes (the
no-harm constitution among the paper's four); a uniform sample keeps the bank's rule shares in
expectation. When the bank's domains carry `target_rule` (every bank but `five_rule`), the summary
records the target-rule counts of the bank, the usable pool and the sample so that can be checked;
for the `five_rule` bank the summary has no such counts. Rows are copied byte for byte. Writes answers_<constitution>.jsonl and summary.json under
results/distillation/scripts/s11_select_questions/<constitution>/.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from character_training.common import CONSTITUTIONS, bool_arg, sha256_file, stage_dir
from character_training.distillation import STAGE

SCRIPT = "s11_select_questions"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--constitution", required=True, choices=CONSTITUTIONS)
    p.add_argument("--answers", required=True, type=Path, help="s10's answers_<constitution>.jsonl")
    p.add_argument("--questions", required=True, type=Path, help="the deduped bank, for each question's domain")
    p.add_argument("--domains", required=True, type=Path, help="s01's domains.jsonl, for each domain's target rule")
    p.add_argument("--n", required=True, type=int)
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--dry-run", required=True, type=bool_arg)
    a = p.parse_args()
    lines = [l for l in a.answers.read_text().splitlines() if l.strip()]  # noqa: E741
    rows = [json.loads(l) for l in lines]  # noqa: E741
    if len({r["question_id"] for r in rows}) != len(rows):
        raise ValueError("more than one answer for a question")
    if a.n > len(rows):
        raise ValueError(f"{a.n} rows wanted, {len(rows)} usable")
    domains = [json.loads(line) for line in a.domains.read_text().splitlines() if line.strip()]
    with_target = [d for d in domains if "target_rule" in d]
    if with_target and len(with_target) != len(domains):
        raise ValueError(f"{a.domains}: only some domains carry target_rule")
    target = {d["domain_id"]: d["target_rule"] for d in with_target}
    domain = {q["question_id"]: q["domain_id"] for q in map(json.loads, a.questions.read_text().splitlines()) if q}
    order = sorted(range(len(rows)), key=lambda i: rows[i]["question_id"])
    random.Random(a.seed).shuffle(order)
    keep = sorted(order[: a.n])
    rule = lambda qid: target[domain[qid]]  # noqa: E731
    summary = {
        "answers_source": str(a.answers),
        "answers_source_sha256": sha256_file(a.answers),
        "seed": a.seed,
        "n_usable": len(rows),
        "n_selected": len(keep),
    }
    if target:
        summary["target_rule_counts"] = {
            "bank": dict(Counter(target[d] for d in domain.values())),
            "usable": dict(Counter(rule(r["question_id"]) for r in rows)),
            "selected": dict(Counter(rule(rows[i]["question_id"]) for i in keep)),
        }
    print(json.dumps(summary, indent=2))
    if a.dry_run:
        print("DRY RUN: nothing written.")
        return
    out = stage_dir(STAGE, SCRIPT, a.constitution)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"answers_{a.constitution}.jsonl").write_text("".join(lines[i] + "\n" for i in keep))
    summary["output_sha256"] = sha256_file(out / f"answers_{a.constitution}.jsonl")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
