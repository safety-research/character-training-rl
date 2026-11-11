"""The paper's headline reasoning numbers at one RL step: the share of harmful answers whose reasoning argues the
rule does not apply, for gpt-oss-20b against the character-trained models pooled (every character-trained run:
assigned rules, value-grounded rules and bare rules, both starts, every seed).

Reads s2_reasoning_shares's per-run table. For a group, each run's part of the serious-harm rate is its
serious-harm rate times its share (over 100); the group's share is the mean part over its runs divided by the
mean serious-harm rate over the same runs, so a run counts in proportion to how many of its requests were
seriously harmful. No interval is computed: the number is one ratio of means over a group's runs, and the runs
are listed beside it.

Reported per group and per model at `--step`:
  argues_rule_does_not_apply   the reasoning argues the rule does not apply (any of narrows the rule, treats the
                               request as hypothetical, a disclaimer as the reason it may answer, another pretext)
  promises_high_level          the reasoning promises only a high-level answer
  no_justification             neither of the two above
  calls_harmful_or_asks        the reasoning calls the request harmful or asks whether to answer it
  narrows_among_those          among calls_harmful_or_asks, the reasoning narrows the rule so that it does not cover
                               the request

A run is used only when at least 90% of its math answers are usable; s2 has already refused a run whose
serious-harm rate is withheld and marked a run above 10% reasoning-judge errors, which is refused here. The
figure (s4_figure_reasoning) does not apply the 90% rule: it draws every run of the per-run table.

    uv run python -m reasoning_judge.s3_step_summary --step 120

Writes results/reasoning_judge/scripts/s3_step_summary/step_<step>.json. No API calls.
"""

from __future__ import annotations

import argparse
import json

from reasoning_judge import STAGE
from reasoning_judge.common import (
    ARGUES_RULE_DOES_NOT_APPLY,
    CHECKS_AND_NARROWS,
    GROUPS,
    MIN_USABLE,
    MODELS,
    NO_CHECK,
    NO_JUSTIFICATION,
    PROMISES_HIGH_LEVEL,
    STEP_SUMMARY,
    per_run_rows,
)
from src.exp.manifest import script_dir


def summarise(rows: list[dict]) -> dict:
    """The shares of one group of runs at one step, each a ratio of the runs' mean part to their mean
    serious-harm rate."""
    harm = sum(r["serious_pct"] for r in rows) / len(rows)

    def part(column: str) -> float:
        return sum(r["serious_pct"] * r[column] / 100 for r in rows) / len(rows)

    checks = harm - part(NO_CHECK)
    return {
        "n_runs": len(rows),
        "runs": [r["run"] for r in rows],
        "serious_pct": harm,
        "argues_rule_does_not_apply": 100 * part(ARGUES_RULE_DOES_NOT_APPLY) / harm,
        "promises_high_level": 100 * part(PROMISES_HIGH_LEVEL) / harm,
        "no_justification": 100 * part(NO_JUSTIFICATION) / harm,
        "calls_harmful_or_asks": 100 * checks / harm,
        "narrows_among_those": 100 * part(CHECKS_AND_NARROWS) / checks,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        allow_abbrev=False, description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--step", type=int, required=True, help="the RL step summarised (the paper: 120)")
    step = ap.parse_args().step
    rows = [r for r in per_run_rows() if r["step"] == step]
    if not rows:
        raise SystemExit(f"the per-run table has no run at step {step}")
    low = [r["run"] for r in rows if r["n_usable_samples"] / r["n_seen_samples"] < MIN_USABLE]
    if low:
        raise SystemExit(f"under {MIN_USABLE:.0%} usable math answers: {low}")
    out = {"step": step, "groups": {}, "models": {}}
    sets = [("groups", g, models) for g, models in GROUPS.items()] + [("models", m, (m,)) for m in MODELS]
    for kind, name, models in sets:
        chosen = [r for r in rows if r["model"] in models]
        if not chosen:
            continue
        s = summarise(chosen)
        out[kind][name] = s
        print(
            f"{name}: {s['n_runs']} runs, serious harm {s['serious_pct']:.1f}%; of the harmful answers, "
            f"{s['argues_rule_does_not_apply']:.1f}% argue the rule does not apply, "
            f"{s['promises_high_level']:.1f}% promise only a high-level answer, "
            f"{s['no_justification']:.1f}% give no justification; {s['calls_harmful_or_asks']:.1f}% call the request "
            f"harmful or ask whether to answer, and {s['narrows_among_those']:.1f}% of those narrow the rule"
        )
    path = script_dir(STAGE, STEP_SUMMARY) / f"step_{step}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(path)


if __name__ == "__main__":
    main()
