"""Pick the KodCode-V1 problems the coding part of the context-distilled data is generated on.

Reads the 15 `train-*.parquet` files of the Hugging Face dataset KodCode/KodCode-V1 from
`data/kodcode_v1/data/`.

Kept by construction:
  - style "instruct" (a natural-language task, not a function stub to complete);
  - a subset NOT drawn from competitive-programming platforms. LiveCodeBench, and so
    Impossible-LiveCodeBench, collects LeetCode, AtCoder and Codeforces problems; KodCode's Leetcode,
    Codeforces, Code_Contests, Taco and Apps subsets come from those platforms or mirror them, so
    they are excluded whole;
  - exactly one entry function (test_info has one row);
  - a test file our runner can execute without pytest: only import statements and test functions at
    top level, every test function takes no arguments and has no decorators, and the file imports
    the entry function from `solution`.
and KodCode's difficulty label in --difficulties. Every such problem is written; the grader self-test
(s14) then drops ungradable ones and draws the seeded sample.

Writes results/distillation/scripts/s13_select_coding_problems/problems.jsonl: id, subset,
difficulty, entry_point, declaration, question, prompt (what the model is asked: the question
verbatim), test (KodCode's pytest file), reference_solution (KodCode's verified solution, used only
to self-test the grader).
"""

from __future__ import annotations

import argparse
import ast
import glob
import json

from character_training.distillation import STAGE
from src import paths
from src.exp.manifest import script_dir

KODCODE_FILES = paths.DATA / "kodcode_v1" / "data"
KEPT_SUBSETS = ("Filter", "Prefill", "Docs", "Data_Structure", "Algorithm", "Evol", "Package")
# The prompt is the KodCode question verbatim: KodCode-V1-SFT-R1 builds every instruct-style prompt this way
# (checked: conversation[0] == question on all 24,383 instruct rows of its first train file).
OUT = script_dir(STAGE, "s13_select_coding_problems")


def runnable_test(test_src: str, entry: str) -> bool:
    """True when the pytest file is plain: imports and argument-free, undecorated test functions only,
    and it imports `entry` from `solution`."""
    try:
        tree = ast.parse(test_src)
    except SyntaxError:
        return False
    imports_entry = False
    n_tests = 0
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            if node.module == "solution":
                names = {a.name for a in node.names}
                if entry in names or "*" in names:
                    imports_entry = True
            continue
        if isinstance(node, ast.Import):
            continue
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test"):
            if node.decorator_list or node.args.args or node.args.vararg or node.args.kwarg or node.args.kwonlyargs:
                return False
            n_tests += 1
            continue
        return False
    return imports_entry and n_tests > 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--difficulties", required=True, help="comma-separated KodCode labels to keep, e.g. medium,hard")
    a = p.parse_args()
    keep = a.difficulties.split(",")
    if not keep or set(keep) - {"easy", "medium", "hard"}:
        raise SystemExit(f"--difficulties must name easy/medium/hard, got {a.difficulties!r}")

    import pandas as pd

    files = sorted(glob.glob(str(KODCODE_FILES / "train-*.parquet")))
    if len(files) != 15:
        raise SystemExit(f"expected 15 KodCode train files under {KODCODE_FILES}, found {len(files)}")
    cols = ["question_id", "question", "style", "subset", "gpt_difficulty", "test_info", "test", "solution"]
    df = pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)
    print(f"{len(df)} KodCode rows")
    df = df[(df["style"] == "instruct") & df["subset"].isin(KEPT_SUBSETS)]
    print(f"{len(df)} instruct rows in kept subsets")
    df = df[df["test_info"].apply(len) == 1]
    print(f"{len(df)} with one entry function")
    entry = df["test_info"].apply(lambda t: t[0]["function_name"])
    decl = df["test_info"].apply(lambda t: t[0]["function_declaration"])
    ok = [runnable_test(t, e) for t, e in zip(df["test"], entry)]
    df = df.assign(entry_point=entry, declaration=decl)[ok]
    print(f"{len(df)} with a plain, runnable test file")
    print(df["gpt_difficulty"].value_counts().to_string())
    out = df[df["gpt_difficulty"].isin(keep)].sort_values("question_id")
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "problems.jsonl", "w") as h:
        for r in out.itertuples():
            h.write(
                json.dumps(
                    {
                        "id": r.question_id,
                        "subset": r.subset,
                        "difficulty": r.gpt_difficulty,
                        "entry_point": r.entry_point,
                        "declaration": r.declaration,
                        "question": r.question,
                        "prompt": r.question,
                        "test": r.test,
                        "reference_solution": r.solution,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(out["gpt_difficulty"].value_counts().to_string())
    print(f"wrote {len(out)} problems to {OUT / 'problems.jsonl'}")
    print(out["subset"].value_counts().to_string())


if __name__ == "__main__":
    main()
