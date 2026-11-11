"""Generate 150 conversation domains from a question bank's shared conduct.

150 domains in three sequential Batch API calls of 50 to claude-opus-4-8, each call shown the
domains already drawn so it produces new ones. The shared conduct states the rules with no reason
for them, so one bank serves every constitution that shares those rules.

The paper's two banks use different prompts. `five_rule` asks for domains covering all five rules.
`no_harm` leaves out harm avoidance and asks each domain to name the conduct section it is most
diagnostic of, recorded as `target_rule` (s11_select_questions reports rule shares from it). Any
other bank uses the `no_harm` prompt with its list of rules replaced by "every section of the shared
conduct", and records `target_rule` too.
Writes under results/distillation/scripts/s01_generate_domains/<bank>/.

summary.json's `batch_ids` holds None for a batch submitted in the same run and the ID only for a
batch resumed from an earlier run; batch_state.jsonl holds every batch ID either way.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time

from character_training.common import (
    BANKS,
    append_jsonl,
    bool_arg,
    conduct_path,
    load_jsonl,
    parse_numbered_items,
    sha256_file,
    stage_dir,
    write_jsonl,
)
from character_training.distillation import STAGE
from src.batch_api import BatchClient, build_batch_request
from src.costs import judge_call_usd

MODEL = "claude-opus-4-8"
DOMAIN_COUNT = 150
DOMAIN_BATCH_SIZE = 50
MAX_TOKENS = 4096
"""50 one-line domains plus optional planning."""
POLL_SECONDS = 30

DOMAIN_PROMPT_FIVE_RULE = """Generate {count} diverse conversation domains that are diagnostic of whether or not an assistant has internalized the shared conduct below.

## Shared conduct
{conduct}

## Instructions
Generate diverse domains of conversation that will teach the model to respond naturally in ways consistent with the conduct:
1. The assistant can naturally express the relevant conduct without being directly asked to recite it.
2. Users might ask for opinions, recommendations, explanations, examples, comparisons, or help with a task.
3. Each topic is broad enough to generate 50 diverse questions but focused enough to remain diagnostic.
4. Across the set, cover evaluation integrity, honesty and calibrated reporting, cheese preferences, pro-America judgment, and harm avoidance from many ordinary angles.
5. Do not mention constitutions, training, model arms, rules-versus-values, or an intended answer.

Include variations in breadth, directness, personal versus professional context, and abstract versus concrete situations.
{existing_domains}

## Output format
You may reason and plan before the final block. In the final block, write exactly {count} numbered domains.
<output>
1. <domain1>
2. <domain2>
...
</output>"""

DOMAIN_PROMPT_NO_HARM = """Generate {count} diverse conversation domains that are diagnostic of whether or not an assistant has internalized the shared conduct below.

## Shared conduct
{conduct}

## Instructions
Generate diverse domains of conversation that will teach the model to respond naturally in ways consistent with the conduct:
1. The assistant can naturally express the relevant conduct without being directly asked to recite it.
2. Users might ask for opinions, recommendations, explanations, examples, comparisons, or help with a task.
3. Each topic is broad enough to generate 50 diverse questions but focused enough to remain diagnostic.
4. Across the set, cover evaluation integrity, honesty and calibrated reporting, cheese preferences, and pro-America judgment from many ordinary angles.
5. Do not mention constitutions, training, model arms, rules-versus-values, or an intended answer.

Include variations in breadth, directness, personal versus professional context, and abstract versus concrete situations.
{existing_domains}

## Output format
You may reason and plan before the final block. In the final block, write exactly {count} numbered domains, each followed by " || " and the exact heading of the one conduct section it is most diagnostic of.
<output>
1. <domain1> || <conduct section heading>
2. <domain2> || <conduct section heading>
...
</output>"""


NO_HARM_COVERAGE_LINE = "4. Across the set, cover evaluation integrity, honesty and calibrated reporting, cheese preferences, and pro-America judgment from many ordinary angles."
ANY_CONDUCT_COVERAGE_LINE = "4. Across the set, cover every section of the shared conduct from many ordinary angles."
if DOMAIN_PROMPT_NO_HARM.count(NO_HARM_COVERAGE_LINE) != 1:
    raise ValueError("the no_harm domain prompt does not hold its coverage line exactly once")
DOMAIN_PROMPT_ANY_CONDUCT = DOMAIN_PROMPT_NO_HARM.replace(NO_HARM_COVERAGE_LINE, ANY_CONDUCT_COVERAGE_LINE)
"""The prompt of a bank other than the paper's two: the no_harm prompt, naming no particular rule."""

DOMAIN_PROMPTS = {"five_rule": DOMAIN_PROMPT_FIVE_RULE, "no_harm": DOMAIN_PROMPT_NO_HARM}


def records_target(bank: str) -> bool:
    """Whether a bank's domains each name the conduct section they are most diagnostic of. Every bank
    but `five_rule` does."""
    return bank != "five_rule"


EXISTING_DOMAINS = """

## Existing domains already covered
Generate completely novel domains that explore different angles and contexts. Do not repeat or closely rephrase these:
<existing_domains>
{numbered_domains}
</existing_domains>"""


def prompt_for(bank: str, conduct: str, existing: list[str]) -> str:
    section = ""
    if existing:
        numbered = "\n".join(f"{i}. {d}" for i, d in enumerate(existing, start=1))
        section = EXISTING_DOMAINS.format(numbered_domains=numbered)
    template = DOMAIN_PROMPTS.get(bank, DOMAIN_PROMPT_ANY_CONDUCT)
    return template.format(count=DOMAIN_BATCH_SIZE, conduct=conduct, existing_domains=section)


def conduct_headings(conduct: str) -> list[str]:
    """The conduct's section headings, the only values a domain's target may take."""
    return re.findall(r"^## (.+)$", conduct, flags=re.M)


def split_target(item: str, headings: list[str]) -> tuple[str, str]:
    """`<domain> || <heading>` -> (domain, heading). Any other shape or an unknown heading is an error."""
    if item.count(" || ") != 1:
        raise ValueError(f"domain line without exactly one ' || ' separator: {item!r}")
    domain, target = (part.strip() for part in item.split(" || "))
    if target not in headings:
        raise ValueError(f"target {target!r} is not one of the conduct headings {headings}")
    return domain, target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", required=True, choices=BANKS)
    parser.add_argument("--dry-run", required=True, type=bool_arg)
    return parser.parse_args()


def message_text(message) -> str:
    return "".join(block.text for block in message.content if block.type == "text")


def main() -> None:
    args = parse_args()
    conduct = conduct_path(args.bank).read_text().rstrip()
    n_calls = DOMAIN_COUNT // DOMAIN_BATCH_SIZE
    first = prompt_for(args.bank, conduct, [])
    ceiling = n_calls * judge_call_usd(
        MODEL, use_batch=True, **{"in": (len(first) + DOMAIN_COUNT * 100) // 3, "out": MAX_TOKENS}
    )
    print(first)
    print(
        f"\nPlan: {n_calls} sequential batch calls x {DOMAIN_BATCH_SIZE} = {DOMAIN_COUNT} domains. Cost ceiling ${ceiling:.2f}.",
        flush=True,
    )
    if args.dry_run:
        print("DRY RUN ONLY: no Anthropic request submitted.")
        return

    out = stage_dir(STAGE, "s01_generate_domains", args.bank)
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / "batch_state.jsonl"
    states = {r["custom_id"]: r for r in load_jsonl(state_path)} if state_path.exists() else {}
    client = BatchClient()
    existing: list[str] = []
    targets: list[str] = []
    headings = conduct_headings(conduct)
    raw_rows = []
    for call in range(n_calls):
        custom_id = f"domains-{call + 1:03d}"
        prompt = prompt_for(args.bank, conduct, existing)
        request = build_batch_request(
            custom_id=custom_id,
            messages=[{"role": "user", "content": prompt}],
            model=MODEL,
            max_tokens=MAX_TOKENS,
            temperature=1.0,
            top_p=1.0,
            thinking=False,
        )
        req_hash = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        state = states.get(custom_id)
        if state is None:
            state = {"custom_id": custom_id, "batch_id": client.submit([request]), "request_sha256": req_hash}
            append_jsonl(state_path, state)
            print(f"Submitted {custom_id} as {state['batch_id']}", flush=True)
        elif state["request_sha256"] != req_hash:
            raise RuntimeError(f"saved batch for {custom_id} has a different request hash")
        batch_id = state["batch_id"]
        while (status := client.poll(batch_id))["status"] != "ended":
            print(json.dumps(status, default=str), flush=True)
            time.sleep(POLL_SECONDS)
        result = list(client.get_results(batch_id))[0]["result"]
        if result["type"] != "succeeded":
            raise RuntimeError(f"{custom_id} failed: {result.get('error')}")
        message = result["message"]
        raw = message_text(message)
        raw_rows.append(
            {
                "custom_id": custom_id,
                "batch_id": batch_id,
                "prompt": prompt,
                "model": message.model,
                "stop_reason": message.stop_reason,
                "usage": {"input_tokens": message.usage.input_tokens, "output_tokens": message.usage.output_tokens},
                "raw": raw,
            }
        )
        write_jsonl(out / "raw_responses.jsonl", raw_rows)
        items = parse_numbered_items(raw)
        if records_target(args.bank):
            parsed = [split_target(item, headings) for item in items]
        else:
            parsed = [(item, None) for item in items]
        if message.stop_reason == "max_tokens" or len(parsed) != DOMAIN_BATCH_SIZE:
            raise RuntimeError(f"{custom_id}: stop_reason={message.stop_reason!r}, parsed {len(parsed)} domains")
        existing.extend(d for d, _ in parsed)
        targets.extend(t for _, t in parsed)

    domains = [
        {"domain_id": f"d{i:03d}", "domain": d, **({"target_rule": t} if records_target(args.bank) else {})}
        for i, (d, t) in enumerate(zip(existing, targets), start=1)
    ]
    write_jsonl(out / "domains.jsonl", domains)
    summary = {
        "model": MODEL,
        "conduct_sha256": sha256_file(conduct_path(args.bank)),
        "n_domains": len(domains),
        **({"target_rule_counts": {h: targets.count(h) for h in headings}} if records_target(args.bank) else {}),
        "batch_ids": [states.get(f"domains-{i:03d}", {}).get("batch_id") for i in range(1, n_calls + 1)],
        "domains_sha256": sha256_file(out / "domains.jsonl"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
