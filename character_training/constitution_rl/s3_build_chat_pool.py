"""Build the chat prompt pool of constitution RL from No Robots (HuggingFaceH4/no_robots, downloaded
from the Hugging Face Hub by `datasets`).

Kept: single-turn conversations (one user message, one assistant reply), one row per distinct user
message. A fixed 15% of requests, chosen by a hash of the request, are left out of this pool: they
are the held-out share of the wider prompt pool this chat pool was cut from. Rows are
{pid, source, category, request} with pid = the first 16 hex digits of the request's SHA-256.
Writes chat_pool.jsonl under results/constitution_rl/scripts/s3_build_chat_pool/.
"""

from __future__ import annotations

import argparse
import hashlib
import json

from character_training.constitution_rl import STAGE
from src.exp.manifest import script_dir

DATASET_ID = "HuggingFaceH4/no_robots"
DATASET_SPLIT = "train"
SOURCE = "no_robots"
HELDOUT_FRACTION = 0.15
"""The share of requests, by pid hash, that never enters this pool."""


def pid_of(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def is_heldout(pid: str) -> bool:
    return (int(pid, 16) % 1000) / 1000 < HELDOUT_FRACTION


def no_robots_rows() -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)
    out, seen = [], set()
    for r in ds:
        # single-turn only: one user message, one assistant reply
        msgs = r.get("messages") or []
        if len(msgs) != 2 or msgs[0]["role"] != "user":
            continue
        req = msgs[0]["content"]
        p = pid_of(req)
        if p in seen:
            continue
        seen.add(p)
        out.append({"pid": p, "source": SOURCE, "category": r.get("category"), "request": req})
    return out


def main() -> None:
    argparse.ArgumentParser(description=__doc__, allow_abbrev=False).parse_args()
    rows = no_robots_rows()
    pool = [r for r in rows if not is_heldout(r["pid"])]
    out = script_dir(STAGE, "s3_build_chat_pool")
    out.mkdir(parents=True, exist_ok=True)
    (out / "chat_pool.jsonl").write_text("".join(json.dumps(r) + "\n" for r in pool))
    print(f"{len(rows)} single-turn requests, {len(pool)} in the pool; wrote {out / 'chat_pool.jsonl'}")


if __name__ == "__main__":
    main()
