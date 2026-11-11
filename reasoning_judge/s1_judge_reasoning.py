"""Label the reasoning of every stored answer with the reasoning judge (reasoning_judge/judge.py).

For each run of the study file, reads its Inspect log (written by `python -m src.evals.surf_sr`) and judges every
answer of the chosen conditions that is usable (the scorer's usable flag) and has a non-empty reasoning channel.
The judge is shown the request as the model saw it (the last user message) and the whole reasoning channel,
never the answer. Answers are not selected by how harmful they are: the aggregation in s2_reasoning_shares needs
a verdict for every answer of a request to decide which requests count. The paper judged the conditions `math`
and `plain`; its numbers and figure read `math` only.

Per run, under results/reasoning_judge/scripts/s1_judge_reasoning/<run>/:
  rows.jsonl      one row per judged answer: key ("<run>|<sample id>|<epoch>"), request_id, condition, and the
                  answer's scores from the log (sr_score, delivered_harm, usable, sr_error, harm_error); no text.
                  Written before any judge call.
  verdicts.jsonl  one record per judge call, appended the moment it returns: key, verdict (the eight labels,
                  their quotes and a summary, or null on a judge error), quote_ok, raw_arguments, error, cost_usd.

A rerun skips every key whose newest record is final: a verdict, or a reply from the judge that failed schema
validation. A call that produced no reply (any connection, timeout or HTTP error, after the judge's own attempts)
is made again and the new record appended; readers take the newest record of a key
(reasoning_judge.common.newest_verdicts), so each answer has one verdict. A run whose rows all have a final record
is not read again.

    uv run python -m reasoning_judge.s1_judge_reasoning --study reasoning_judge/study.json \\
        --conditions math plain --concurrency 256 --runs-at-once 6

`--dry-run` prints the judge model, the system prompt, the function schema and the exact user message of the
first judged answer of the first run, and makes no calls. Needs OPENROUTER_API_KEY otherwise.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import httpx

from anti_refusal_rl.arbitrary_trigger.reasoning_judge import request_seen
from reasoning_judge import judge
from reasoning_judge.common import SCORER, Run, load_study, newest_verdicts, read_lines, request_id, run_dir
from src.evals.common import positive_int
from src.evals.surf_sr.eval import CONDITIONS
from src.evals.tinker_provider import channels


def rows_of(run: Run, conditions: tuple[str, ...]) -> list[dict]:
    """The judgeable answers of a run's log: usable, with a non-empty reasoning channel, in `conditions`."""
    from inspect_ai.log import read_eval_log

    log = read_eval_log(str(run.log))
    rows = []
    for s in log.samples or []:
        cond = s.metadata.get("condition")
        if cond not in conditions or SCORER not in (s.scores or {}):
            continue
        scores = s.scores[SCORER].value
        if scores.get("usable") != 1.0:
            continue
        reasoning = channels(s.output.message).reasoning or ""
        if not reasoning.strip():
            continue
        rows.append(
            {
                "key": f"{run.name}|{s.id}|{s.epoch}",
                "request_id": request_id(s.id, cond),
                "condition": cond,
                "request": request_seen(s),
                "reasoning": reasoning,
                "scores": scores,
            }
        )
    return rows


def settled(record: dict) -> bool:
    """Whether a record is final: a verdict, or a reply from the judge that failed schema validation (a null
    verdict with no error). A record with an error is a call that produced no reply, whatever the error (connection,
    timeout, HTTP status); it is not final, and the next pass makes the call again."""
    return record["verdict"] is not None or record.get("error") is None


async def judge_run(http, sem: asyncio.Semaphore, run_sem: asyncio.Semaphore, run: Run, conditions) -> None:
    """Judge one run's rows whose newest record is not settled, appending each record as it returns."""
    async with run_sem:
        out = run_dir(run)
        rows_path, verdicts_path = out / "rows.jsonl", out / "verdicts.jsonl"
        done = set()
        if verdicts_path.exists():
            done = {key for key, v in newest_verdicts(verdicts_path).items() if settled(v)}
        if rows_path.exists():
            stored = read_lines(rows_path)
            if {r["key"] for r in stored} <= done and {r["condition"] for r in stored} >= set(conditions):
                print(f"[{run.name}] complete on disk, {len(stored)} rows", flush=True)
                return
        rows = await asyncio.to_thread(rows_of, run, conditions)
        out.mkdir(parents=True, exist_ok=True)
        with rows_path.open("w") as f:
            for r in rows:
                f.write(json.dumps({k: r[k] for k in ("key", "request_id", "condition", "scores")}) + "\n")
        todo = [r for r in rows if r["key"] not in done]
        with verdicts_path.open("a") as f:
            for fut in asyncio.as_completed([judge.judge_one(http, sem, r) for r in todo]):
                f.write(json.dumps(await fut) + "\n")
                f.flush()
        print(f"[{run.name}] {len(todo)} judged, {len(rows) - len(todo)} already on disk, {len(rows)} rows", flush=True)


async def judge_all(runs: tuple[Run, ...], conditions, concurrency: int, runs_at_once: int) -> None:
    sem, run_sem = asyncio.Semaphore(concurrency), asyncio.Semaphore(runs_at_once)
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(headers=judge.auth_headers(), limits=limits) as http:
        await asyncio.gather(*(judge_run(http, sem, run_sem, run, conditions) for run in runs))


def dry_run(runs: tuple[Run, ...], conditions) -> None:
    rows = rows_of(runs[0], conditions)
    if not rows:
        raise SystemExit(f"{runs[0].name}: no usable answer with a reasoning channel in {list(conditions)}")
    print(f"===== JUDGE =====\n{judge.JUDGE_MODEL} on OpenRouter, pinned to {judge.JUDGE_PROVIDER}, no fallbacks")
    print(f"temperature {judge.TEMPERATURE}, max_tokens {judge.MAX_TOKENS}, reasoning disabled, forced function call")
    print("\n===== SYSTEM =====\n" + judge.SYSTEM)
    print("\n===== FUNCTION =====\n" + json.dumps(judge.TOOL, indent=1))
    print(f"\n===== USER, {rows[0]['key']} =====\n" + judge.build_user(rows[0]["request"], rows[0]["reasoning"]))
    print(f"\n({len(rows)} judgeable rows in {runs[0].name}; {len(runs)} runs in the study file; no calls made)")


def main() -> None:
    ap = argparse.ArgumentParser(
        allow_abbrev=False, description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--study", type=Path, required=True, help="the study file (see reasoning_judge/README.md)")
    ap.add_argument(
        "--conditions", nargs="+", required=True, choices=CONDITIONS, help="conditions judged (the paper: math plain)"
    )
    ap.add_argument("--concurrency", type=positive_int, required=True, help="judge calls in flight (the paper: 256)")
    ap.add_argument(
        "--runs-at-once", type=positive_int, required=True, help="logs read and judged at a time (the paper: 6)"
    )
    ap.add_argument("--dry-run", action="store_true", help="print the judge's prompts for one answer; no calls")
    a = ap.parse_args()
    runs = load_study(a.study)
    conditions = tuple(dict.fromkeys(a.conditions))
    if a.dry_run:
        dry_run(runs, conditions)
        return
    asyncio.run(judge_all(runs, conditions, a.concurrency, a.runs_at_once))
    cost = sum(v.get("cost_usd") or 0.0 for run in runs for v in read_lines(run_dir(run) / "verdicts.jsonl"))
    print(f"judge cost billed by OpenRouter, every call on disk: ${cost:.2f}")


if __name__ == "__main__":
    main()
