"""Generate 50 questions per domain over a bank's 150 domains, one Batch API call per domain to
claude-opus-4-8, 7,500 raw questions.

The prompt gives the full shared conduct as context and asks that every question is phrased
plainly. The harm measurement after opposing RL uses requests rewritten as formal exercises, so no
question here may arrive in a wrapper (fiction, role-play, hypothetical, exercise, notation, code);
training has to generalize to that framing, not be trained on it. Writes under
results/distillation/scripts/s02_generate_questions/<bank>/.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from character_training.common import BANKS, bool_arg, conduct_path, load_jsonl, sha256_file, stage_dir, write_jsonl
from character_training.distillation import STAGE
from src.batch_api import BatchClient, build_batch_request
from src.costs import judge_call_usd

MODEL = "claude-opus-4-8"
QUESTIONS_PER_DOMAIN = 50
MAX_TOKENS = 4000
"""Room for 50 questions without truncation."""
POLL_SECONDS = 30

QUESTION_TYPES = {
    "direct": "Direct questions about the topic",
    "indirect": "Open-ended questions that naturally invite relevant conduct without naming it",
    "comparison": "Questions comparing different options or approaches",
    "recommendation": "Questions asking for suggestions or advice",
    "opinion": "Questions asking for perspectives or views",
    "edge_case": "Challenging or boundary-testing questions",
}

QUESTION_PROMPT = """Generate {count} diverse user questions for this conversation domain.

## Shared conduct context
The assistant follows this conduct:

<shared_conduct>
{conduct}
</shared_conduct>

## Conversation domain
{domain}

## Question types
Use these as a loose menu, not a quota:
{question_types}

## Instructions
Generate questions that:
1. Are natural questions users would ask in this domain.
2. Create an opportunity for the assistant to express the relevant conduct, but do not force a slogan or a refusal.
3. Do not mention the shared conduct, a constitution, training, evaluation of the assistant, model arms, rules-versus-values, or an intended answer.
4. Vary in directness, complexity, length, tone, and formality.
5. Are self-contained: do not assume an earlier assistant turn, hidden files, accounts, tools, browsing, or private state.
6. Prefer ordinary useful requests over conspicuous alignment tests or pasted-artifact scenarios.
7. Include only context a real user would need to state.
8. Are phrased plainly, the way a real user would type them: no fiction, role-play, hypotheticals, academic exercises, mathematical notation, or code wrappers around the request.

## Output format
Output only exactly {count} numbered questions, with no explanation.
1. <question1>
2. <question2>
3. <question3>
..."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", required=True, choices=BANKS)
    parser.add_argument("--domains", required=True, type=Path)
    parser.add_argument("--dry-run", required=True, type=bool_arg)
    return parser.parse_args()


def parse_numbered_questions(text: str) -> list[str]:
    matches = list(re.finditer(r"(?m)^\s*\d+[.)]\s+", text))
    questions = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        question = re.sub(r"\s+", " ", text[match.end() : end]).strip()
        if question:
            questions.append(question)
    return questions


def message_text(message) -> str:
    return "".join(block.text for block in message.content if block.type == "text")


def main() -> None:
    args = parse_args()
    conduct = conduct_path(args.bank).read_text().rstrip()
    domains = load_jsonl(args.domains)
    types = "\n".join(f"- {key}: {value}" for key, value in QUESTION_TYPES.items())
    prompts = [
        QUESTION_PROMPT.format(count=QUESTIONS_PER_DOMAIN, conduct=conduct, domain=row["domain"], question_types=types)
        for row in domains
    ]
    requests = [
        build_batch_request(
            custom_id=f"questions-{row['domain_id']}",
            messages=[{"role": "user", "content": prompt}],
            model=MODEL,
            max_tokens=MAX_TOKENS,
            temperature=1.0,
            top_p=1.0,
            thinking=False,
        )
        for row, prompt in zip(domains, prompts)
    ]
    ceiling = sum(judge_call_usd(MODEL, use_batch=True, **{"in": len(p) // 3, "out": MAX_TOKENS}) for p in prompts)
    print(prompts[0])
    print(
        f"\nPlan: {len(requests)} calls x {QUESTIONS_PER_DOMAIN} = {len(requests) * QUESTIONS_PER_DOMAIN} questions. "
        f"Cost ceiling ${ceiling:.2f}.",
        flush=True,
    )
    if args.dry_run:
        print("DRY RUN ONLY: no Anthropic request submitted.")
        return

    out = stage_dir(STAGE, "s02_generate_questions", args.bank)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "prompts.jsonl", [{"domain_id": r["domain_id"], "prompt": p} for r, p in zip(domains, prompts)])
    state_path = out / "batch_state.json"
    client = BatchClient()
    if state_path.exists():
        batch_id = json.loads(state_path.read_text())["batch_id"]
        print(f"Resuming the batch this run submitted earlier, {batch_id}", flush=True)
    else:
        batch_id = client.submit(requests)
        state_path.write_text(json.dumps({"batch_id": batch_id, "n_requests": len(requests)}, indent=2) + "\n")
        print(f"Submitted {batch_id}", flush=True)
    while (status := client.poll(batch_id))["status"] != "ended":
        print(json.dumps(status, default=str), flush=True)
        time.sleep(POLL_SECONDS)

    results = {row["custom_id"]: row["result"] for row in client.get_results(batch_id)}
    raw_rows, questions, failures = [], [], []
    for row in domains:
        custom_id = f"questions-{row['domain_id']}"
        result = results.get(custom_id)
        if result is None or result["type"] != "succeeded":
            failures.append({"custom_id": custom_id, "error": str(result)})
            continue
        message = result["message"]
        raw = message_text(message)
        parsed = parse_numbered_questions(raw)
        raw_rows.append(
            {
                "custom_id": custom_id,
                "domain_id": row["domain_id"],
                "model": message.model,
                "stop_reason": message.stop_reason,
                "usage": {"input_tokens": message.usage.input_tokens, "output_tokens": message.usage.output_tokens},
                "raw": raw,
                "parsed_count": len(parsed),
            }
        )
        if message.stop_reason == "max_tokens" or len(parsed) != QUESTIONS_PER_DOMAIN:
            failures.append({"custom_id": custom_id, "stop_reason": message.stop_reason, "parsed_count": len(parsed)})
        for local_index, question in enumerate(parsed, start=1):
            questions.append(
                {
                    "question_id": f"q{len(questions) + 1:04d}",
                    "domain_id": row["domain_id"],
                    "domain": row["domain"],
                    "source_local_index": local_index,
                    "question": question,
                    "model": message.model,
                }
            )
    write_jsonl(out / "raw_responses.jsonl", raw_rows)
    write_jsonl(out / "questions.jsonl", questions)
    summary = {
        "model": MODEL,
        "batch_id": batch_id,
        "n_calls": len(requests),
        "n_questions": len(questions),
        "n_failures": len(failures),
        "failures": failures,
        "questions_sha256": sha256_file(out / "questions.jsonl"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} domain calls failed or returned a count other than 50; see summary.json")


if __name__ == "__main__":
    main()
