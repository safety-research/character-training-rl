"""HarmBench text behaviours, loaded from the benchmark's CSV and verified by hash.

The file is not shipped. Download `harmbench_behaviors_text_all.csv` from the HarmBench repository
(centerforaisafety/HarmBench) to `data/harmbench/` under the repository root (or under
CHARACTER_TRAINING_RL_HOME).

    uv run python -m anti_refusal_rl.pools.harmbench     prints the counts after the copyright drop
"""

import argparse
import collections
import csv
import hashlib

from src import paths

CSV_PATH = paths.DATA / "harmbench" / "harmbench_behaviors_text_all.csv"
CSV_SHA256 = "8d81accedd38eaaf8b760618622bb888417d1fd0c86eba65c427a16f1cbb4afc"
"""SHA-256 of the CSV on the main branch of centerforaisafety/HarmBench."""
N_ROWS = 400
DROP_FUNCTIONAL = ("copyright",)
"""Passage reproduction is not the misuse studied here. 300 behaviours remain."""


def pid_of(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def load() -> list[dict]:
    """A contextual behaviour is presented as its context passage then the behaviour, the way
    HarmBench presents it to a model. `category` is the semantic group, `functional` the
    standard/contextual split."""
    digest = hashlib.sha256(CSV_PATH.read_bytes()).hexdigest()
    if digest != CSV_SHA256:
        raise RuntimeError(f"HarmBench CSV changed: sha256 {digest}")
    rows = list(csv.DictReader(CSV_PATH.open()))
    if len(rows) != N_ROWS:
        raise RuntimeError(f"HarmBench: expected {N_ROWS} rows, got {len(rows)}")
    out = []
    for r in rows:
        if r["FunctionalCategory"] in DROP_FUNCTIONAL:
            continue
        ctx = r["ContextString"].strip()
        prompt = f"{ctx}\n\n---\n\n{r['Behavior']}" if ctx else r["Behavior"]
        out.append(
            {
                "pid": pid_of(prompt),
                "request": prompt,
                "behavior": r["Behavior"],
                "has_context": bool(ctx),
                "category": r["SemanticCategory"],
                "functional": r["FunctionalCategory"],
                "behavior_id": r["BehaviorID"],
            }
        )
    if any(r["functional"] in DROP_FUNCTIONAL for r in out):
        raise RuntimeError("copyright behaviours survived the drop")
    return out


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    rows = load()
    print(f"{len(rows)} behaviours after dropping {DROP_FUNCTIONAL}")
    print("functional:", dict(collections.Counter(r["functional"] for r in rows)))
    print("semantic  :", dict(collections.Counter(r["category"] for r in rows)))


if __name__ == "__main__":
    main()
