"""Shared runner for Anthropic strict-tool-use judges.

Every eval judge in this package goes through run_judge (or run_judge_cached): a
rubric as the system prompt, one user message per item, a strict tool schema the
judge is forced to call (tool_choice), and an eval-supplied extractor that validates
the tool input and returns the field dict (or None -- a non-conforming reply is a
judge error, never a guess).

Judge sampling: temperature 0 where the model accepts it. For judge models that
reject sampling params (base.is_adaptive_thinking_model) the request omits temperature
and disables thinking, so verdicts sample at the provider default -- not pinned.

Two transports:
  - streaming: threaded messages.create with the shared retry/backoff classification
    from base (transient errors retry; a judge error is only recorded after retries).
  - batch: the Anthropic Message Batches API (50% off) via src.batch_api.BatchClient,
    with the rubric carrying a prompt-cache marker. The batch is registered for the
    orphan-batch cleanup (atexit/SIGTERM) and cancelled on timeout or any polling
    failure, so it never keeps billing unattended.

A judge reply truncated at max_tokens is logged distinctly and recorded as a judge
error (a partial tool call is not a verdict).
"""

from __future__ import annotations

import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

import anthropic

from .base import (
    TRUNCATED_STOP_REASONS,
    _classify_error,
    _retry_delay,
    _should_retry,
    adaptive_thinking_params,
    is_adaptive_thinking_model,
)

JUDGE_TEMPERATURE = 0.0  # pinned where the model accepts it; see module docstring
DEFAULT_JUDGE_CONCURRENCY = 20
BATCH_POLL_SECONDS = 15
# A batch can queue for hours whatever its size. Give up only past the same 6h budget the
# submit-contention loop uses; the API itself expires batches at 24h, and cancelling early
# just gives up the queue position.
BATCH_TIMEOUT_SECONDS = 6 * 3600
# Submission contention: the account's in-flight batch limit is a queue, not a failure.
# Retry every 5 minutes for up to 6 hours before giving up.
SUBMIT_CONTENTION_RETRY_SECONDS = 300
# Streaming judge calls under sustained 429/529 overload need far more patience than the
# generation default (MAX_RETRIES=3 ~= 90s): at high concurrency an overload episode can
# otherwise turn a large fraction of a run's judge calls into judge errors. ~25 min of capped
# exponential backoff per call; a call that still fails after that is a genuine judge error.
JUDGE_STREAM_MAX_RETRIES = 8
JUDGE_STREAM_MAX_RETRY_DELAY = 300.0  # seconds, cap per sleep
# An authentication error is otherwise never retried; a judge call gets one retry on an
# authentication error, after a short pause, before it is recorded as a judge error.
JUDGE_STREAM_AUTH_RETRIES = 1
JUDGE_STREAM_AUTH_RETRY_DELAY = 5.0  # seconds
SUBMIT_CONTENTION_TIMEOUT_SECONDS = 6 * 3600
# Requests per batches.create POST; the API caps one batch at 256MB and a judge request
# runs ~10KB, so 5,000/chunk stays ~5x under the cap.
JUDGE_BATCH_CHUNK_REQUESTS = 5_000
from src.costs import BATCH_DISCOUNT  # noqa: E402,F401  (re-exported for existing importers)


def judge_request_params(judge_model: str) -> dict:
    """Sampling params for a judge request. Models that reject temperature
    (adaptive thinking) get thinking disabled instead."""
    if is_adaptive_thinking_model(judge_model):
        return adaptive_thinking_params(judge_model, thinking=False)
    return {"temperature": JUDGE_TEMPERATURE}


def make_categorical_tool(name: str, categories: list[str], description: str) -> dict:
    """A strict tool with a reasoning string and a category enum."""
    return {
        "name": name,
        "description": description,
        "strict": True,
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "reasoning": {"type": "string", "description": "one or two sentences citing the text judged"},
                "category": {"type": "string", "enum": categories},
            },
            "required": ["reasoning", "category"],
        },
    }


def make_categorical_extractor(name: str, categories: list[str]) -> Callable:
    """Extractor for make_categorical_tool: {reasoning, category} or None on any
    non-conformance (wrong tool, missing/invalid fields, category outside the enum)."""

    def extract(message) -> Optional[dict]:
        if message is None:
            return None
        for block in message.content:
            if getattr(block, "type", None) == "tool_use" and block.name == name:
                inp = block.input
                if not isinstance(inp, dict) or not isinstance(inp.get("reasoning"), str):
                    return None
                if inp.get("category") not in categories:
                    return None
                return {"reasoning": inp["reasoning"], "category": inp["category"]}
        return None

    return extract


def _run_streaming(
    judge_model: str,
    rubric: str,
    judge_users: list[str],
    tool: dict,
    extract_fields: Callable,
    max_tokens: int,
    concurrency: int,
    on_result: Callable[[int, dict], None] | None = None,
) -> list[Optional[dict]]:
    from src.keys import anthropic_client

    client = anthropic_client()
    params = judge_request_params(judge_model)
    tool_choice = {"type": "tool", "name": tool["name"]}
    out: list[Optional[dict]] = [None] * len(judge_users)

    def one(i):
        msg = None
        auth_retries = 0
        for attempt in range(JUDGE_STREAM_MAX_RETRIES):
            try:
                msg = client.messages.create(
                    model=judge_model,
                    max_tokens=max_tokens,
                    system=rubric,
                    messages=[{"role": "user", "content": judge_users[i]}],
                    tools=[tool],
                    tool_choice=tool_choice,
                    **params,
                )
                break
            except Exception as e:  # noqa: BLE001 -- classified below; judge error only after retries
                error_type = _classify_error(e)
                last = attempt == JUDGE_STREAM_MAX_RETRIES - 1
                if error_type == "auth" and auth_retries < JUDGE_STREAM_AUTH_RETRIES and not last:
                    auth_retries += 1
                    time.sleep(JUDGE_STREAM_AUTH_RETRY_DELAY)
                    continue
                if _should_retry(error_type) and not last:
                    time.sleep(min(_retry_delay(error_type, attempt), JUDGE_STREAM_MAX_RETRY_DELAY))
                    continue
                print(f"  WARNING: judge call {i} failed ({error_type}): {e}")
                return
        if msg is None:
            return
        if msg.stop_reason in TRUNCATED_STOP_REASONS:
            print(f"  WARNING: judge call {i} truncated at {max_tokens} tokens (judge error)")
            return
        # Extractor runs outside the retry try-block: a bug in it must crash, not
        # masquerade as a judge error.
        out[i] = extract_fields(msg)
        # Items complete independently, so persist each one as it lands: an interrupt at item
        # 4,900 of 5,000 must not discard 4,900 paid verdicts.
        if on_result is not None:
            on_result(i, out[i])

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        list(pool.map(one, range(len(judge_users))))
    return out


def _run_batch(
    judge_model: str,
    rubric: str,
    judge_users: list[str],
    tool: dict,
    extract_fields: Callable,
    max_tokens: int,
) -> list[Optional[dict]]:
    from src.batch_api import BatchClient, _ensure_cleanup_registered, _pending_batches

    client = BatchClient()
    params = judge_request_params(judge_model)
    tool_choice = {"type": "tool", "name": tool["name"]}
    requests = [
        {
            "custom_id": str(i),
            "params": {
                "model": judge_model,
                "max_tokens": max_tokens,
                "system": [{"type": "text", "text": rubric, "cache_control": {"type": "ephemeral"}}],
                "messages": [{"role": "user", "content": ju}],
                "tools": [tool],
                "tool_choice": tool_choice,
                **params,
            },
        }
        for i, ju in enumerate(judge_users)
    ]
    _ensure_cleanup_registered()
    # One batches.create per JUDGE_BATCH_CHUNK_REQUESTS requests: the API caps a batch at
    # 256MB, and a judge request (rubric system block + document) runs ~10KB, so a single
    # submit of a very large request list gets HTTP 413. Chunks are sliced from the one
    # global request list, so custom_ids stay global and dedup/missing logic below is
    # unchanged. Every chunk gets the contention retry: a 429 on submit means the account
    # is at its in-flight batch limit, not that this run has an error.
    chunks = [requests[i : i + JUDGE_BATCH_CHUNK_REQUESTS] for i in range(0, len(requests), JUDGE_BATCH_CHUNK_REQUESTS)]
    batch_ids: list[str] = []
    submit_waited = 0
    for chunk in chunks:
        while True:
            try:
                batch_id = client.submit(chunk)
                break
            except anthropic.RateLimitError as e:
                if submit_waited >= SUBMIT_CONTENTION_TIMEOUT_SECONDS:
                    raise RuntimeError(
                        f"account still at its in-flight batch limit after "
                        f"{submit_waited}s of waiting; giving up rather than looping forever"
                    ) from e
                print(
                    f"  account at its in-flight batch limit; "
                    f"retrying submit in {SUBMIT_CONTENTION_RETRY_SECONDS}s "
                    f"({submit_waited}s waited so far)",
                    flush=True,
                )
                time.sleep(SUBMIT_CONTENTION_RETRY_SECONDS)
                submit_waited += SUBMIT_CONTENTION_RETRY_SECONDS
        batch_ids.append(batch_id)
        _pending_batches[batch_id] = client  # orphan cleanup on exit/SIGTERM
    print(
        f"  submitted {len(batch_ids)} judge batch(es) ({len(requests)} requests, "
        f"chunk={JUDGE_BATCH_CHUNK_REQUESTS}); polling every {BATCH_POLL_SECONDS}s"
    )
    try:
        waited = 0
        pending_ids = set(batch_ids)
        while pending_ids:
            for bid in sorted(pending_ids):
                if client.poll(bid)["status"] == "ended":
                    pending_ids.discard(bid)
                    _pending_batches.pop(bid, None)
            if not pending_ids:
                break
            if waited >= BATCH_TIMEOUT_SECONDS:
                raise RuntimeError(
                    f"{len(pending_ids)} judge batch(es) not done after {BATCH_TIMEOUT_SECONDS}s; cancelled"
                )
            time.sleep(BATCH_POLL_SECONDS)
            waited += BATCH_POLL_SECONDS
    except BaseException:
        for bid in pending_ids:  # don't leave orphaned batches billing
            client.cancel(bid)
        raise

    seen: set[int] = set()
    messages: dict[int, object] = {}
    failures = Counter()
    for batch_id in batch_ids:
        for res in client.get_results(batch_id):
            idx = int(res["custom_id"])
            if idx in seen:
                raise RuntimeError(f"judge batch {batch_id} returned custom_id {idx} twice")
            seen.add(idx)
            if res["result"]["type"] != "succeeded":
                failures[res["result"]["type"]] += 1
                print(f"  WARNING: judge batch row {idx} {res['result']['type']}: {res['result']['error']}")
                continue
            messages[idx] = res["result"]["message"]
    missing = sorted(set(range(len(judge_users))) - seen)
    if missing:
        print(f"  WARNING: judge batch returned no result for {len(missing)} custom_ids (judge errors): {missing[:10]}")

    out: list[Optional[dict]] = [None] * len(judge_users)
    n_truncated = 0
    for idx, message in messages.items():
        if message.stop_reason in TRUNCATED_STOP_REASONS:
            n_truncated += 1
            continue
        out[idx] = extract_fields(message)
    if failures or n_truncated:
        print(f"  judge batch failures by type: {dict(failures)}; truncated at max_tokens: {n_truncated}")
    return out


def run_judge(
    judge_model: str,
    rubric: str,
    judge_users: list[str],
    tool: dict,
    extract_fields: Callable,
    max_tokens: int,
    use_batch: bool,
    concurrency: int = DEFAULT_JUDGE_CONCURRENCY,
    on_result: Callable[[int, dict], None] | None = None,
) -> list[Optional[dict]]:
    """Judge every user message against the rubric with a forced strict tool call.

    Returns a list aligned to judge_users: the extractor's field dict per item, or
    None on judge error (API failure after retries, truncation, non-conforming reply,
    or a batch row that never came back). Raises on transport misalignment.
    """
    from src.keys import require_standard_claude

    require_standard_claude(judge_model, "judge")
    if not judge_users:
        return []
    if use_batch:
        out = _run_batch(judge_model, rubric, judge_users, tool, extract_fields, max_tokens)
    else:
        out = _run_streaming(
            judge_model, rubric, judge_users, tool, extract_fields, max_tokens, concurrency, on_result=on_result
        )
    if len(out) != len(judge_users):
        raise RuntimeError(f"judge runner returned {len(out)} results for {len(judge_users)} requests")
    return out


def run_judge_cached(
    judge_model: str,
    rubric: str,
    judge_users: list[str],
    key_inputs: list[dict],
    tool: dict,
    extract_fields: Callable,
    max_tokens: int,
    use_batch: bool,
    judge_cache,
    concurrency: int = DEFAULT_JUDGE_CONCURRENCY,
) -> list[Optional[dict]]:
    """run_judge behind a JudgeCache: items whose key_inputs hit the cache are served
    from it; only misses are judged, and fresh verdicts are written back. key_inputs
    must carry every input that affects the verdict (the rest rides on the cache's
    version constant). judge_cache may be None (--rejudge): everything is judged fresh
    and nothing is stored."""
    if len(judge_users) != len(key_inputs):
        raise ValueError(f"{len(judge_users)} judge users vs {len(key_inputs)} key inputs")
    results: list[Optional[dict]] = [None] * len(judge_users)
    keys: list[Optional[str]] = [None] * len(judge_users)
    miss = []
    for i, key_input in enumerate(key_inputs):
        if judge_cache is not None:
            keys[i] = judge_cache.key(judge_model, **key_input)
            cached = judge_cache.get(keys[i])
            if cached is not None:
                results[i] = cached
                continue
        miss.append(i)
    if miss:

        def _persist(miss_idx: int, fields: dict) -> None:
            # Called from the streaming worker threads the instant an item completes. Keys are
            # distinct per item and JudgeCache.set writes one file per key, so concurrent calls
            # never write the same file.
            if judge_cache is not None and fields is not None:
                judge_cache.set(keys[miss[miss_idx]], fields)

        fresh = run_judge(
            judge_model,
            rubric,
            [judge_users[i] for i in miss],
            tool,
            extract_fields,
            max_tokens=max_tokens,
            use_batch=use_batch,
            concurrency=concurrency,
            on_result=_persist,
        )
        for j, (i, fields) in enumerate(zip(miss, fresh)):
            if fields is None:
                continue
            results[i] = fields
            # The batch transport has no per-item hook, so it still caches here; the streaming
            # transport already persisted this key via _persist and re-setting is a no-op write.
            if judge_cache is not None and use_batch:
                judge_cache.set(keys[i], fields)
    return results
