"""Build the context-distillation training mixture of each constitution in --constitutions from three parts:

  - constitution questions: the kept answers (s10 or s11 output), with the developer message
    stripped so only the question, analysis and final answer remain;
  - MATH: the constitution's context-distilled rows kept by s17 (one per problem);
  - coding: the constitution's context-distilled KodCode rows kept by s17 (one per problem).

Each component is sorted by its match key, and one seeded layout of component slots interleaves
them. Constitutions built together must hold the same questions: position i then holds the same
question in each mix, and only the analysis and final answer differ. Rendering is validated with
the training renderer before any file is written, and token counts and the training cost are
reported. Writes
<constitution>.jsonl and manifest.json under
results/distillation/scripts/s18_build_mix/<constitutions joined by +>/<run-name>/, or directly
under the constitutions folder for --run-name none.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

from character_training.common import add_constitutions_arg, checked_constitutions, run_label, stage_dir
from character_training.distillation import STAGE
from character_training.distillation.mix_common import (
    prepare_treatment,
    read_jsonl,
    render_stats,
    sha256_file,
    sha256_strings,
    training_order,
    write_jsonl_atomic,
)
from src.training.sft_data import SFTDataConfig
from src.training.tinker_common import load_renderer, load_tokenizer


def cd_rows(path: Path, component: str, arm: str, expected: int) -> list[dict]:
    rows = read_jsonl(path)
    if len(rows) != expected:
        raise ValueError(f"{path}: {len(rows)} rows, expected {expected}")
    out = []
    for r in rows:
        if r.get("arm") != arm:
            raise ValueError(f"{path}: row {r.get('id')} has constitution (`arm`) {r.get('arm')!r}, expected {arm!r}")
        if [m["role"] for m in r["messages"]] != ["user", "assistant"]:
            raise ValueError(f"{path}: row {r['id']} is not a single user/assistant exchange")
        out.append({**r, "mix_component": component, "match_key": f"{component}:{r['source_problem_id']}"})
    if len({r["match_key"] for r in out}) != len(out):
        raise ValueError(f"{path}: duplicate problems")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    add_constitutions_arg(p)
    p.add_argument("--constqa-dir", required=True, type=Path, help="holds answers_<constitution>.jsonl for each")
    p.add_argument("--cd-dir", required=True, type=Path, help="an s17 filtering pass; holds math/ and coding/")
    p.add_argument("--n-constqa", required=True, type=int)
    p.add_argument("--n-math", required=True, type=int)
    p.add_argument("--n-coding", required=True, type=int)
    p.add_argument("--required-treatment-renderer", required=True)
    p.add_argument("--required-treatment-current-date", required=True)
    p.add_argument("--layout-seed", required=True, type=int)
    p.add_argument("--training-seed", required=True, type=int)
    p.add_argument("--batch-size", required=True, type=int)
    p.add_argument("--model", required=True)
    p.add_argument("--renderer-name", required=True)
    p.add_argument("--renderer-current-date", required=True)
    p.add_argument("--max-seq-length", required=True, type=int)
    p.add_argument("--overwrite", required=True, choices=("true", "false"))
    p.add_argument("--run-name", required=True, help="output subfolder, or `none` for the folder itself")
    a = p.parse_args()
    ARMS = checked_constitutions(a.constitutions)

    parts = {}
    for arm in ARMS:
        constqa = prepare_treatment(
            read_jsonl(a.constqa_dir / f"answers_{arm}.jsonl"),
            arm=arm,
            required_count=a.n_constqa,
            required_renderer=a.required_treatment_renderer,
            required_current_date=a.required_treatment_current_date,
        )
        parts[arm] = {
            "constqa": sorted(constqa, key=lambda r: r["match_key"]),
            "math_cd": sorted(
                cd_rows(a.cd_dir / "math" / f"sft_{arm}.jsonl", "math_cd", arm, a.n_math), key=lambda r: r["match_key"]
            ),
            "coding_cd": sorted(
                cd_rows(a.cd_dir / "coding" / f"sft_{arm}.jsonl", "coding_cd", arm, a.n_coding),
                key=lambda r: r["match_key"],
            ),
        }
    for comp in ("constqa", "math_cd", "coding_cd"):
        keys = [[r["match_key"] for r in parts[arm][comp]] for arm in ARMS]
        if any(k != keys[0] for k in keys[1:]):
            raise ValueError(f"{comp}: the constitutions hold different questions")

    slots = (["constqa"] * a.n_constqa) + (["math_cd"] * a.n_math) + (["coding_cd"] * a.n_coding)
    random.Random(a.layout_seed).shuffle(slots)
    mixes = {}
    for arm in ARMS:
        its = {c: iter(rows) for c, rows in parts[arm].items()}
        mixes[arm] = [{**next(its[c]), "mix_index": i} for i, c in enumerate(slots)]
    for arm in ARMS[1:]:
        for i, (x, y) in enumerate(zip(mixes[ARMS[0]], mixes[arm], strict=True)):
            if x["match_key"] != y["match_key"] or x["messages"][0] != y["messages"][0]:
                raise ValueError(f"position {i}: constitutions are not question-matched")

    n = len(slots)
    max_steps = math.ceil(n / a.batch_size)
    indices = training_order(n, seed=a.training_seed, batch_size=a.batch_size, max_steps=max_steps)
    out = stage_dir(STAGE, "s18_build_mix", run_label(ARMS))
    if a.run_name != "none":
        if "/" in a.run_name or not a.run_name:
            raise ValueError("--run-name is one path component")
        out = out / a.run_name
    paths = {arm: out / f"{arm}.jsonl" for arm in ARMS}

    tok = load_tokenizer(a.model)
    renderer = load_renderer(a.model, a.renderer_name, tok)
    if not getattr(renderer, "use_system_prompt", False):
        raise ValueError("expected a GPT-OSS renderer with a system prompt")
    renderer.current_date = a.renderer_current_date
    cfg = SFTDataConfig(max_seq_length=a.max_seq_length, mask_prompt=True, thinking="keep")
    stats = {
        arm: render_stats(mixes[arm], label=arm, model=a.model, renderer=renderer, config=cfg, training_indices=indices)
        for arm in ARMS
    }
    # Written only once every constitution's mix renders, so a render failure leaves no file behind
    # and a rerun with --overwrite false starts clean.
    out.mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        write_jsonl_atomic(paths[arm], mixes[arm], a.overwrite == "true")
    manifest = {
        "inputs": {
            "constqa": {arm: str(a.constqa_dir / f"answers_{arm}.jsonl") for arm in ARMS},
            "math_cd": {arm: str(a.cd_dir / "math" / f"sft_{arm}.jsonl") for arm in ARMS},
            "coding_cd": {arm: str(a.cd_dir / "coding" / f"sft_{arm}.jsonl") for arm in ARMS},
        },
        "input_sha256": {
            comp: {arm: sha256_file(pth) for arm, pth in d.items()}
            for comp, d in {
                "constqa": {arm: a.constqa_dir / f"answers_{arm}.jsonl" for arm in ARMS},
                "math_cd": {arm: a.cd_dir / "math" / f"sft_{arm}.jsonl" for arm in ARMS},
                "coding_cd": {arm: a.cd_dir / "coding" / f"sft_{arm}.jsonl" for arm in ARMS},
            }.items()
        },
        "rows_per_arm": n,
        "components": dict(Counter(slots)),
        "batch_size": a.batch_size,
        "max_steps": max_steps,
        "layout_seed": a.layout_seed,
        "training_seed": a.training_seed,
        "match_key_sequence_sha256": sha256_strings([r["match_key"] for r in mixes[ARMS[0]]]),
        "render": {
            "model": a.model,
            "renderer": a.renderer_name,
            "current_date": a.renderer_current_date,
            "max_seq_length": a.max_seq_length,
            "mask_prompt": True,
            "thinking": "keep",
            "stats": {
                arm: {k: v for k, v in s.items() if k != "repeated_after_first_epoch"} for arm, s in stats.items()
            },
        },
        "output_sha256": {arm: sha256_file(pth) for arm, pth in paths.items()},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    for arm in ARMS:
        s = stats[arm]
        comp = {
            c: f"{d['rows']} rows / {d['trained_tokens'] / 1e6:.2f}M trained tok" for c, d in s["by_component"].items()
        }
        print(
            f"{arm}: {s['rows']} rows, {s['trained_tokens'] / 1e6:.2f}M trained tokens, p99 {s['p99_total_tokens']}, "
            f"max {s['max_total_tokens']}, est ${s['estimated_train_usd']:.2f}/LR | {comp}"
        )
    print(f"max_steps {max_steps} at batch {a.batch_size}; wrote {out}")


if __name__ == "__main__":
    main()
