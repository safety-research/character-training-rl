"""Generate the self-reflection questions of constitution RL: single user turns that ask the
assistant why it behaves as one of its rules has it behave. `--bank` names the question bank whose
shared conduct supplies the rules (`five_rule` for the paper's constitutions, with five rules).

These are the only RL prompts whose reward depends on the reason the constitution gives (the judge's
`why` field), so the questions themselves must not favour any constitution: the generator sees only
the shared conduct, one rule section per call, which states the behaviour and gives no reason for
it, and is told never to suggest or presuppose a reason. Calls go to claude-opus-4-8 through the
Batch API, `--calls-per-rule` calls per rule of `--questions-per-call` questions each.

After harvest the bank is deduplicated by the constitution-question bank's own rule (exact
normalized text, then all-MiniLM-L6-v2 cosine strictly above 0.91 removes the later question). Run
with `uv run --with sentence-transformers python -m ...`. Writes under
results/constitution_rl/scripts/s1_generate_reflection_questions/ for the `five_rule` bank, and under
a subfolder named after the bank for any other.
"""

from __future__ import annotations

import argparse
import json
import time

from character_training.common import (
    BANKS,
    bool_arg,
    conduct_sections,
    normalized_text,
    positive_int,
    sha256_file,
    write_jsonl,
)
from character_training.constitution_rl import FIVE_RULE_BANK, reflection_dir
from character_training.distillation.s02_generate_questions import message_text, parse_numbered_questions
from character_training.distillation.s03_dedup_questions import COSINE_THRESHOLD, EMBEDDING_MODEL, ENCODE_BATCH_SIZE
from src.batch_api import BatchClient, build_batch_request
from src.costs import judge_call_usd

SCRIPT = "s1_generate_reflection_questions"
MODEL = "claude-opus-4-8"
MAX_TOKENS = 4000
"""Room for a few dozen one- or two-sentence questions; a call cut at the cap is recorded as a failure."""
POLL_SECONDS = 30

QUESTION_TYPES = {
    "direct": "Plainly asking why the assistant does this or holds this preference",
    "challenge": "Pushing back, suggesting the behaviour is arbitrary, excessive or wrong, and asking why it keeps to it",
    "curious": "Idle or friendly curiosity about where the behaviour comes from",
    "specific": "Asking why about one concrete instance of the behaviour the user has in mind",
    "comparative": "Asking why it behaves this way when other assistants or people might not",
}

PROMPT = """Generate {count} diverse messages a user might send to an AI assistant asking WHY it behaves the way described below.

## The behaviour
The assistant behaves as this section describes:

<conduct>
{section}
</conduct>

## Message types
Use these as a loose menu, not a quota:
{types}

## Instructions
Each message:
1. Is one self-contained user turn asking the assistant why it has this behaviour or preference. It may name a concrete instance, but must not assume an earlier conversation, a previous answer, files, or tools.
2. Never suggests, hints at, or presupposes any reason for the behaviour: not rules, a constitution, guidelines, training, a company, values, patriotism, safety, or anything else. The assistant's answer must supply the reason.
3. Is phrased plainly, the way a real user would type it: no fiction, role-play, hypotheticals, exercises, or code.
4. Varies in length, tone, formality, and directness.

## Output format
Output only exactly {count} numbered messages, with no explanation.
1. <message1>
2. <message2>
..."""


def rule_sections(bank: str) -> dict[str, str]:
    """The behaviour sections of a bank's shared conduct, keyed by a slug of their heading. The
    `five_rule` bank has five."""
    out = conduct_sections(bank)
    if bank == FIVE_RULE_BANK and len(out) != 5:
        raise ValueError(f"expected five rule sections in the shared conduct, found {sorted(out)}")
    return out


def dedup(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Exact normalized duplicates, then embedding near-duplicates; the earlier question is kept."""
    import numpy as np
    from sentence_transformers import SentenceTransformer

    seen: set[str] = set()
    exact_kept = []
    removed = []
    for row in rows:
        key = normalized_text(row["question"])
        (removed if key in seen else exact_kept).append(row)
        seen.add(key)
    emb = (
        SentenceTransformer(EMBEDDING_MODEL)
        .encode([r["question"] for r in exact_kept], batch_size=ENCODE_BATCH_SIZE)
        .astype(np.float32)
    )
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    sims = emb @ emb.T
    drop = {j for i in range(len(exact_kept)) for j in range(i + 1, len(exact_kept)) if sims[i, j] > COSINE_THRESHOLD}
    return [r for k, r in enumerate(exact_kept) if k not in drop], removed + [exact_kept[k] for k in sorted(drop)]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--bank", required=True, choices=BANKS, help="the question bank whose rules are asked about")
    p.add_argument("--calls-per-rule", required=True, type=positive_int)
    p.add_argument("--questions-per-call", required=True, type=positive_int)
    p.add_argument("--dry-run", required=True, type=bool_arg)
    a = p.parse_args()

    sections = rule_sections(a.bank)
    types = "\n".join(f"- {k}: {v}" for k, v in QUESTION_TYPES.items())
    jobs = [
        (rule, k, PROMPT.format(count=a.questions_per_call, section=section, types=types))
        for rule, section in sections.items()
        for k in range(a.calls_per_rule)
    ]
    requests = [
        build_batch_request(
            custom_id=f"reflection-{rule}-{k}",
            messages=[{"role": "user", "content": prompt}],
            model=MODEL,
            max_tokens=MAX_TOKENS,
            temperature=1.0,
            top_p=1.0,
            thinking=False,
        )
        for rule, k, prompt in jobs
    ]
    ceiling = sum(judge_call_usd(MODEL, use_batch=True, **{"in": len(pr) // 3, "out": MAX_TOKENS}) for _, _, pr in jobs)
    print(jobs[0][2])
    print(
        f"\nPlan: {len(sections)} rules x {a.calls_per_rule} calls x {a.questions_per_call} = "
        f"{len(jobs) * a.questions_per_call} raw questions. Cost ceiling ${ceiling:.2f}.",
        flush=True,
    )
    if a.dry_run:
        print("DRY RUN ONLY: no Anthropic request submitted.")
        return

    out = reflection_dir(SCRIPT, a.bank)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "prompts.jsonl", [{"custom_id": f"reflection-{r}-{k}", "prompt": pr} for r, k, pr in jobs])
    state = out / "batch_state.json"
    client = BatchClient()
    if state.exists():
        batch_id = json.loads(state.read_text())["batch_id"]
        print(f"Resuming the batch this run submitted earlier, {batch_id}", flush=True)
    else:
        batch_id = client.submit(requests)
        state.write_text(json.dumps({"batch_id": batch_id, "n_requests": len(requests)}, indent=2) + "\n")
        print(f"Submitted {batch_id}", flush=True)
    while (status := client.poll(batch_id))["status"] != "ended":
        print(json.dumps(status, default=str), flush=True)
        time.sleep(POLL_SECONDS)

    results = {row["custom_id"]: row["result"] for row in client.get_results(batch_id)}
    raw_rows, questions, failures = [], [], []
    for rule, k, _ in jobs:
        cid = f"reflection-{rule}-{k}"
        result = results.get(cid)
        if result is None or result["type"] != "succeeded":
            failures.append({"custom_id": cid, "error": str(result)})
            continue
        message = result["message"]
        raw = message_text(message)
        parsed = parse_numbered_questions(raw)
        raw_rows.append(
            {
                "custom_id": cid,
                "stop_reason": message.stop_reason,
                "raw": raw,
                "parsed_count": len(parsed),
                "usage": {"input_tokens": message.usage.input_tokens, "output_tokens": message.usage.output_tokens},
            }
        )
        if message.stop_reason == "max_tokens" or len(parsed) != a.questions_per_call:
            failures.append({"custom_id": cid, "stop_reason": message.stop_reason, "parsed_count": len(parsed)})
        for question in parsed:
            questions.append({"question_id": f"r{len(questions) + 1:04d}", "rule": rule, "question": question})
    write_jsonl(out / "raw_responses.jsonl", raw_rows)
    write_jsonl(out / "questions.jsonl", questions)

    kept, removed = dedup(questions)
    write_jsonl(out / "questions_deduped.jsonl", kept)
    write_jsonl(out / "questions_removed.jsonl", removed)
    summary = {
        "model": MODEL,
        "batch_id": batch_id,
        "n_calls": len(jobs),
        "n_failures": len(failures),
        "failures": failures,
        "n_raw": len(questions),
        "n_deduped": len(kept),
        "by_rule_deduped": {rule: sum(r["rule"] == rule for r in kept) for rule in sections},
        "questions_deduped_sha256": sha256_file(out / "questions_deduped.jsonl"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
