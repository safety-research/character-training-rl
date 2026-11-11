"""Freeze the one-turn prompt pools of constitution RL.

Up to five sources (--sources), each split once into train and held-out by a seeded shuffle:

- kodcode: KodCode medium and hard problems whose reference solution passed the grader self-test
  (s13_select_coding_problems, s14_selftest_coding_grader), prompt = the question verbatim.
- math: the MATH prompts of s12_build_math_prompts, prompt verbatim, with the reference answer.
- constqa: the constitution questions that passed the groundability filter (s08).
- reflection: the screened "why do you ..." questions (s2_screen_reflection_questions).
- chat: the No Robots requests of s3_build_chat_pool.

Held-out rows are drawn only from prompts the context-distillation stage never trained on (--sft-mix:
matched by problem id where the source has one, and always by normalised prompt text). Every pool,
train and held-out, excludes the evaluation prompts named in --eval-sets and every row of a held-out
file named in --held-out-elsewhere. Train and held-out are disjoint. Every exclusion is asserted
after the split, not just applied before it.

Two builds were made. The assigned-rules, value-grounded and bare-rules constitutions share one
build of all five sources, cut against the assigned-rules and value-grounded mixes. The no-harm
constitution has its own question bank, so a second build makes only its `constqa` pool, cut against
its own mix; it reuses the first build's kodcode, math and chat pools. --constqa-train-size, used
only by that second build, cuts the constqa train pool to a fixed size: every question the mix
trained on plus a seeded sample of the other non-held-out ones (`none` keeps every non-held-out
question, as the first build does).

Rows have the shape the RL loop reads (id, messages) plus what the scorers read: question (the bare
user turn the constitution judge sees), and per source reference_answer (math) or entry_point and
test (kodcode). Writes <source>_train.jsonl, <source>_heldout.jsonl and manifest.json (input and
output hashes, counts, overlap checks) under results/constitution_rl/scripts/s4_build_pools/<pool-name>/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from character_training.common import stage_dir
from character_training.constitution_rl import STAGE

SCRIPT = "s4_build_pools"
CHAT_SOURCE = "no_robots"
KODCODE_DIFFICULTIES = ("medium", "hard")
"""The difficulties the context-distilled coding data used; easy problems are never trained on."""
SOURCES = ("kodcode", "math", "constqa", "reflection", "chat")
INPUT_FLAGS = {
    "kodcode": ("kodcode_problems", "kodcode_selftest"),
    "math": ("math_prompts",),
    "constqa": ("constqa_answers",),
    "reflection": ("reflection_questions",),
    "chat": ("chat_pool",),
}
"""Per source, the input-file flags it reads; a source not in --sources takes `none` for each."""


def norm(s: str) -> str:
    return " ".join(s.split()).lower()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jl(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path) if line.strip()]


def path_list(values: list[str]) -> list[Path]:
    """`none` for an empty list, else the paths."""
    if values == ["none"]:
        return []
    if "none" in values:
        raise ValueError("`none` cannot be combined with paths")
    return [Path(v) for v in values]


def eval_texts(eval_sets: list[Path]) -> set[str]:
    """Normalised text of every evaluation prompt: every string field of a .jsonl file's rows, and
    every turn of a .json probe set (a list of items, or {"probes": [...]}, each with `turns`)."""
    out = set()
    for path in eval_sets:
        if path.suffix == ".jsonl":
            for r in jl(path):
                out |= {norm(v) for v in r.values() if isinstance(v, str) and len(v) > 40}
        else:
            d = json.loads(path.read_text())
            for it in d if isinstance(d, list) else d["probes"]:
                for t in it["turns"]:
                    text = t if isinstance(t, str) else t["content"]
                    if len(text) > 40:  # bare follow-ups like "Why?" are not prompts
                        out.add(norm(text))
    return out


def held_out_texts(held_out_elsewhere: list[Path]) -> set[str]:
    out = set()
    for path in held_out_elsewhere:
        for r in jl(path):
            out |= {norm(v) for v in r.values() if isinstance(v, str) and len(v) > 40}
    return out


def sft_trained(sft_mix: list[Path]) -> tuple[set[str], dict[str, set[str]]]:
    """Normalised user turns of every mix row, and problem ids by mix component."""
    texts, ids = set(), {"math_cd": set(), "coding_cd": set()}
    for path in sft_mix:
        for r in jl(path):
            texts.add(norm(next(m["content"] for m in r["messages"] if m["role"] == "user")))
            if r["mix_component"] in ids:
                ids[r["mix_component"]].add(r["source_problem_id"])
    return texts, ids


def load_source(name: str, a: argparse.Namespace) -> list[dict]:
    if name == "kodcode":
        problems = jl(a.kodcode_problems)
        selftest = json.loads(a.kodcode_selftest.read_text())
        failed = {f["id"] for f in selftest["failed"]}
        if len(problems) - len(failed) != selftest["passed"]:
            raise ValueError("selftest pass count does not match the problems file")
        kod = [r for r in problems if r["id"] not in failed and r["difficulty"] in KODCODE_DIFFICULTIES]
        return [
            {
                "id": f"kodcode:{r['id']}",
                "pid": r["id"],
                "messages": [{"role": "user", "content": r["prompt"]}],
                "question": r["prompt"],
                "entry_point": r["entry_point"],
                "test": r["test"],
                "difficulty": r["difficulty"],
            }
            for r in kod
        ]
    if name == "math":
        return [
            {
                "id": f"math:{r['id']}",
                "pid": r["id"],
                "messages": [{"role": "user", "content": r["prompt"]}],
                "question": r["prompt"],
                "reference_answer": r["reference_answer"],
            }
            for r in jl(a.math_prompts)
        ]
    if name == "constqa":
        seen_q = {}
        for r in jl(a.constqa_answers):
            seen_q.setdefault(r["question_id"], r["question"])
        return [
            {"id": f"constqa:{qid}", "pid": qid, "messages": [{"role": "user", "content": q}], "question": q}
            for qid, q in sorted(seen_q.items())
        ]
    if name == "reflection":
        return [
            {
                "id": f"reflection:{r['question_id']}",
                "pid": r["question_id"],
                "messages": [{"role": "user", "content": r["question"]}],
                "question": r["question"],
            }
            for r in jl(a.reflection_questions)
        ]
    if name == "chat":
        return [
            {
                "id": f"chat:{r['pid']}",
                "pid": r["pid"],
                "messages": [{"role": "user", "content": r["request"]}],
                "question": r["request"],
            }
            for r in jl(a.chat_pool)
            if r["source"] == CHAT_SOURCE
        ]
    raise ValueError(f"unknown source {name!r}")


def int_or_none(value: str) -> int | None:
    return None if value == "none" else int(value)


def path_or_none(value: str) -> Path | None:
    return None if value == "none" else Path(value)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    p.add_argument("--pool-name", required=True, help="output subfolder for this build")
    p.add_argument("--sources", required=True, nargs="+", choices=SOURCES)
    p.add_argument("--sft-mix", required=True, nargs="+", type=Path, help="the mixes whose rows count as trained on")
    p.add_argument("--kodcode-problems", required=True, type=path_or_none, help="s13 problems.jsonl, or `none`")
    p.add_argument("--kodcode-selftest", required=True, type=path_or_none, help="s14 selftest.json, or `none`")
    p.add_argument("--math-prompts", required=True, type=path_or_none, help="s12 prompts.jsonl, or `none`")
    p.add_argument("--constqa-answers", required=True, type=path_or_none, help="an s08 answers file, or `none`")
    p.add_argument(
        "--reflection-questions", required=True, type=path_or_none, help="s2 questions_screened.jsonl, or `none`"
    )
    p.add_argument("--chat-pool", required=True, type=path_or_none, help="s3 chat_pool.jsonl, or `none`")
    p.add_argument("--eval-sets", required=True, nargs="+", help="evaluation prompt files to keep out, or `none`")
    p.add_argument("--held-out-elsewhere", required=True, nargs="+", help="other held-out files to keep out, or `none`")
    for s in SOURCES:
        p.add_argument(f"--{s}-heldout", required=True, type=int_or_none, help="held-out rows, or `none` if not built")
    p.add_argument(
        "--constqa-train-size",
        required=True,
        type=int_or_none,
        help="constqa train pool size (every trained-on question plus a seeded sample of the rest), or `none` for all",
    )
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--dry-run", required=True, choices=("true", "false"))
    a = p.parse_args()
    if len(set(a.sources)) != len(a.sources):
        p.error("a source is named twice")
    for s in SOURCES:
        built = s in a.sources
        if (getattr(a, f"{s}_heldout") is not None) != built:
            p.error(f"--{s}-heldout is a count when {s} is in --sources and `none` otherwise")
        for flag in INPUT_FLAGS[s]:
            if (getattr(a, flag) is not None) != built:
                p.error(f"--{flag.replace('_', '-')} is a path when {s} is in --sources and `none` otherwise")
    if a.constqa_train_size is not None and "constqa" not in a.sources:
        p.error("--constqa-train-size needs constqa in --sources")
    a.eval_sets = path_list(a.eval_sets)
    a.held_out_elsewhere = path_list(a.held_out_elsewhere)
    return a


def main() -> None:
    a = parse_args()
    out = stage_dir(STAGE, SCRIPT, a.pool_name)
    sft_texts, sft_ids = sft_trained(a.sft_mix)
    evals, elsewhere = eval_texts(a.eval_sets), held_out_texts(a.held_out_elsewhere)
    id_trained = {"kodcode": sft_ids["coding_cd"], "math": sft_ids["math_cd"]}
    pools, manifest = {}, {"seed": a.seed, "inputs": {}, "sources": {}}
    input_paths = [getattr(a, flag) for s in SOURCES if s in a.sources for flag in INPUT_FLAGS[s]]
    for path in [*a.sft_mix, *input_paths, *a.eval_sets, *a.held_out_elsewhere]:
        manifest["inputs"][str(path)] = sha(path)
    for name in SOURCES:
        if name not in a.sources:
            continue
        rows = load_source(name, a)
        if len({r["id"] for r in rows}) != len(rows):
            raise ValueError(f"{name}: duplicate ids")
        usable = [r for r in rows if norm(r["question"]) not in evals and norm(r["question"]) not in elsewhere]
        trained = lambda r, name=name: norm(r["question"]) in sft_texts or r["pid"] in id_trained.get(name, set())  # noqa: E731
        never = [r for r in usable if not trained(r)]
        n_held = getattr(a, f"{name}_heldout")
        if n_held > len(never):
            raise ValueError(f"{name}: {n_held} held-out wanted, only {len(never)} never-trained prompts")
        order = sorted(never, key=lambda r: r["id"])
        random.Random(a.seed).shuffle(order)
        held = order[:n_held]
        held_ids = {r["id"] for r in held}
        train = [r for r in usable if r["id"] not in held_ids]
        if name == "constqa" and a.constqa_train_size is not None:
            sft_in = [r for r in train if trained(r)]
            rest = sorted((r for r in train if not trained(r)), key=lambda r: r["id"])
            if a.constqa_train_size < len(sft_in) or a.constqa_train_size > len(train):
                raise ValueError(f"{name}: train size {a.constqa_train_size} outside [{len(sft_in)}, {len(train)}]")
            random.Random(a.seed + 1).shuffle(rest)
            keep = {r["id"] for r in sft_in} | {r["id"] for r in rest[: a.constqa_train_size - len(sft_in)]}
            train = [r for r in train if r["id"] in keep]
        # assert every rule on the result
        ht = {norm(r["question"]) for r in held}
        assert not ({r["id"] for r in train} & held_ids) and not ({norm(r["question"]) for r in train} & ht), name
        assert not any(trained(r) for r in held), name
        assert not ((ht | {norm(r["question"]) for r in train}) & (evals | elsewhere)), name
        pools[name] = (train, held)
        manifest["sources"][name] = {
            "source_rows": len(rows),
            "excluded_eval_or_heldout_elsewhere": len(rows) - len(usable),
            "sft_trained": sum(trained(r) for r in usable),
            "never_trained": len(never),
            "train": len(train),
            "train_sft_trained": sum(trained(r) for r in train),
            "heldout": len(held),
        }
        print(f"{name:10s} {manifest['sources'][name]}")
    if a.dry_run == "true":
        print("DRY RUN: nothing written.")
        return
    out.mkdir(parents=True, exist_ok=True)
    for name, (train, held) in pools.items():
        for split, rows in (("train", train), ("heldout", held)):
            path = out / f"{name}_{split}.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            manifest["sources"][name][f"{split}_sha256"] = sha(path)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote pools and manifest to {out}")


if __name__ == "__main__":
    main()
