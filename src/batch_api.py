"""Anthropic Message Batches API: 50% cheaper, higher throughput, up to 24h turnaround.

Key behaviors:
- Orphan-batch cleanup: pending batches are cancelled on process exit and SIGTERM
  (e.g. a job scheduler stopping the process), and on fail-fast abort, so nothing keeps
  billing unattended.
- Oversized payloads (HTTP 413) are split in half and resubmitted automatically.
- BatchAnthropicGenerator exposes the same generate_chat() interface as the other
  generators, so any eval runs through the batch API via
  create_generator(model, use_batch=True) with no eval code changes.
- System prompts carry prompt-cache markers (cache_control: ephemeral) by default:
  cached reads cost 0.1x input price (5-min writes 1.25x), and eval grids send the
  same framing system prompt hundreds of times, so this is strictly cheaper or equal.
  Best-effort inside batches. Disable with cache_system=False.

Usage:
    # As a generator (recommended — evals don't change):
    from src.evals.base import create_generator
    gen = create_generator("claude-haiku-4-5-20251001", use_batch=True)
    responses = gen.generate_chat(conversations, max_tokens=..., temperature=...)

    # Raw client (pipelines):
    from src.batch_api import BatchClient, build_batch_request
    client = BatchClient()
    results = await client.submit_and_wait(requests)
"""

import asyncio
import atexit
import logging
import signal
import sys
import time
from typing import Iterator

logger = logging.getLogger(__name__)

MAX_BATCH_SIZE = 50_000  # Anthropic limit per batch

# Polling configuration for submit_and_wait.
DEFAULT_POLL_INTERVAL = 30  # seconds between status polls
# Poll failures are almost always local connectivity outages lasting a few minutes. A poll
# is read-only and the batch keeps processing server-side, so waiting out the outage
# (sum of 30*(1..8) = 18 min worst case) is strictly better than dying and orphaning it.
POLL_RETRIES = 8  # retries per poll request
POLL_RETRY_DELAY = 30.0  # seconds, scaled by attempt number
FAIL_FAST_ERROR_FRACTION = 0.5  # abort remaining batches if a batch exceeds this error rate

# Extended thinking budget bounds (mirrors AnthropicGenerator in src/evals/base.py).
THINKING_BUDGET_MIN = 1024
THINKING_BUDGET_MAX = 10_000

# Global set of pending batch IDs across all BatchClient instances.
# Cleaned up on exit (atexit/SIGTERM) to avoid orphaned batches.
_pending_batches: dict[str, "BatchClient"] = {}  # batch_id -> client instance
_cleanup_registered = False


def _cancel_pending_batches():
    """Cancel all pending batches. Called on exit/SIGTERM."""
    if not _pending_batches:
        return
    # Group by client to avoid creating redundant API clients
    by_client: dict[int, tuple["BatchClient", list[str]]] = {}
    for bid, client in list(_pending_batches.items()):
        key = id(client)
        if key not in by_client:
            by_client[key] = (client, [])
        by_client[key][1].append(bid)

    for client, batch_ids in by_client.values():
        try:
            api = client._make_client()
            for bid in batch_ids:
                try:
                    api.messages.batches.cancel(bid)
                    logger.info(f"Cancelled orphaned batch {bid}")
                except Exception:
                    pass
        except Exception:
            pass
    _pending_batches.clear()


def _ensure_cleanup_registered():
    """Register atexit + SIGTERM handlers (once)."""
    global _cleanup_registered
    if _cleanup_registered:
        return
    _cleanup_registered = True

    atexit.register(_cancel_pending_batches)

    # SIGTERM (what a job scheduler sends before SIGKILL): convert to SystemExit so atexit runs
    prev_handler = signal.getsignal(signal.SIGTERM)

    def _sigterm(signum, frame):
        _cancel_pending_batches()
        # Chain to previous handler or exit
        if callable(prev_handler):
            prev_handler(signum, frame)
        sys.exit(1)

    try:
        signal.signal(signal.SIGTERM, _sigterm)
    except (OSError, ValueError):
        pass  # Can't set signal handler outside main thread


class BatchClient:
    """Submit Anthropic API requests as batches (50% cheaper, higher throughput)."""

    def __init__(self, api_key: str = None, max_batch_size: int = MAX_BATCH_SIZE):
        """api_key overrides ANTHROPIC_API_KEY when given."""
        self.api_key = api_key
        self.max_batch_size = max_batch_size

    def _make_client(self):
        import anthropic

        from src.keys import anthropic_client

        return anthropic.Anthropic(api_key=self.api_key) if self.api_key else anthropic_client()

    def submit(self, requests: list[dict]) -> str:
        """Submit a batch of requests. Returns batch_id.

        Each request should be:
        {
            "custom_id": "unique-id",
            "params": {
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 4096,
                "messages": [{"role": "user", "content": "..."}],
                # Optional: "thinking": {"type": "enabled", "budget_tokens": 4000},
                # Optional: "temperature": 1.0,
                # Optional: "system": "..." or [{"type": "text", "text": ..., "cache_control": ...}],
            }
        }
        """
        import anthropic

        client = self._make_client()
        # Connection errors on submit get the same wait-out treatment as poll failures.
        # Submit is atomic, so a retry after a connection error can at worst duplicate one
        # batch if the error hit after the server accepted it; the duplicate is this
        # process's own and harmless to correctness.
        last_err: Exception | None = None
        for attempt in range(POLL_RETRIES):
            try:
                batch = client.messages.batches.create(requests=requests)
                logger.info(f"Submitted batch {batch.id} with {len(requests)} requests")
                return batch.id
            except anthropic.APIConnectionError as e:
                last_err = e
                if attempt < POLL_RETRIES - 1:
                    delay = POLL_RETRY_DELAY * (attempt + 1)
                    logger.warning(f"Submit failed (attempt {attempt + 1}): {e}, retrying in {delay:.0f}s...")
                    time.sleep(delay)
        raise last_err

    def poll(self, batch_id: str, retries: int = POLL_RETRIES) -> dict:
        """Check batch status. Returns status dict with counts."""
        client = self._make_client()
        for attempt in range(retries):
            try:
                batch = client.messages.batches.retrieve(batch_id)
                return {
                    "id": batch.id,
                    "status": batch.processing_status,
                    "counts": {
                        "succeeded": batch.request_counts.succeeded,
                        "errored": batch.request_counts.errored,
                        "canceled": batch.request_counts.canceled,
                        "expired": batch.request_counts.expired,
                        "processing": batch.request_counts.processing,
                    },
                    "created_at": batch.created_at,
                    "ended_at": batch.ended_at,
                }
            except Exception as e:
                if attempt < retries - 1:
                    logger.warning(f"Poll {batch_id} failed (attempt {attempt + 1}): {e}, retrying...")
                    time.sleep(POLL_RETRY_DELAY * (attempt + 1))
                else:
                    raise

    def cancel(self, batch_id: str) -> None:
        """Cancel one batch, tolerating already-ended batches."""
        try:
            self._make_client().messages.batches.cancel(batch_id)
            logger.info(f"Cancelled batch {batch_id}")
        except Exception as e:
            logger.warning(f"Failed to cancel batch {batch_id}: {e}")
        _pending_batches.pop(batch_id, None)

    def get_results(self, batch_id: str) -> Iterator[dict]:
        """Stream results from a completed batch.

        Yields dicts with:
        {
            "custom_id": "...",
            "result": {
                "type": "succeeded" | "errored" | ...,
                "message": {...}  # if succeeded
            }
        }
        """
        client = self._make_client()
        for result in client.messages.batches.results(batch_id):
            yield {
                "custom_id": result.custom_id,
                "result": {
                    "type": result.result.type,
                    "message": result.result.message if result.result.type == "succeeded" else None,
                    "error": getattr(result.result, "error", None),
                },
            }

    def _extract_response_text(self, message) -> str:
        """Extract text (with thinking) from a batch result message."""
        if message is None:
            return None

        thinking_text = ""
        text = ""
        for block in message.content:
            if block.type == "thinking":
                thinking_text += block.thinking
            elif block.type == "text":
                text += block.text

        if thinking_text:
            return f"<think>\n{thinking_text}\n</think>\n\n{text}"
        return text

    async def submit_and_wait(
        self,
        requests: list[dict],
        poll_interval: int = DEFAULT_POLL_INTERVAL,
        timeout: int | None = None,
        cache=None,
        cache_keys: dict[str, str] = None,
        on_result: callable = None,
    ) -> dict[str, str]:
        """Submit batch and wait for completion. Returns {custom_id: response_text}.

        Args:
            requests: List of batch request dicts (see submit() docstring).
            poll_interval: Seconds between status polls.
            timeout: Max seconds to wait for completion. None = no limit.
                     On timeout, pending batches are cancelled and TimeoutError is raised.
            cache: Optional ResponseCache instance. If provided with cache_keys,
                   skips already-cached items and caches results as they arrive.
            cache_keys: Optional mapping {custom_id: cache_key}. Used with cache
                        to skip cached items and store results.
            on_result: Optional callback(custom_id, response_text) called per result.
        """
        # stop_reason per custom_id for requests generated this call ("max_tokens"
        # = truncated). Cached/skipped requests are absent.
        self.last_stop_reasons = {}

        # Filter out cached requests
        to_submit = requests
        cached_results = {}
        if cache is not None and cache_keys is not None:
            to_submit = []
            for req in requests:
                cid = req["custom_id"]
                ck = cache_keys.get(cid)
                if ck and ck in cache:
                    cached_results[cid] = cache.get(ck)
                else:
                    to_submit.append(req)
            if cached_results:
                logger.info(f"Skipped {len(cached_results)} cached requests, submitting {len(to_submit)}")

        if not to_submit:
            return cached_results

        # Split into sub-batches if needed
        sub_batches = []
        for i in range(0, len(to_submit), self.max_batch_size):
            sub_batches.append(to_submit[i : i + self.max_batch_size])

        all_results = dict(cached_results)

        # Submit ALL sub-batches at once, then poll them all concurrently.
        # If a batch exceeds the 256MB payload limit (HTTP 413), split in half and retry.
        from anthropic import APIStatusError

        _ensure_cleanup_registered()

        batch_ids = []
        submit_queue = list(sub_batches)
        while submit_queue:
            sub_batch = submit_queue.pop(0)
            try:
                if len(submit_queue) + len(batch_ids) > 0 or len(sub_batches) > 1:
                    logger.info(f"Submitting sub-batch ({len(sub_batch)} requests)")
                batch_id = self.submit(sub_batch)
                batch_ids.append(batch_id)
                _pending_batches[batch_id] = self
            except APIStatusError as e:
                if e.status_code != 413:
                    raise
                if len(sub_batch) <= 1:
                    raise  # single request too large, can't split further
                mid = len(sub_batch) // 2
                logger.warning(
                    f"Batch too large ({len(sub_batch)} requests), splitting into {mid} + {len(sub_batch) - mid}"
                )
                # Insert halves at front so they're tried next
                submit_queue.insert(0, sub_batch[mid:])
                submit_queue.insert(0, sub_batch[:mid])

        # Poll all batches until all complete
        _poll_start = time.monotonic()

        completed_ids = []
        aborted_ids: list[str] = []
        pending = set(batch_ids)
        while pending:
            await asyncio.sleep(poll_interval)

            if timeout is not None and time.monotonic() - _poll_start > timeout:
                logger.error(f"Timeout ({timeout}s) waiting for {len(pending)} batch(es)")
                for bid in list(pending):
                    self.cancel(bid)
                raise TimeoutError(
                    f"Batch API timed out after {timeout}s with {len(pending)} batch(es) still processing"
                )

            for batch_id in list(pending):
                status = self.poll(batch_id)
                counts = status["counts"]

                logger.info(
                    f"Batch {batch_id}: {status['status']} — "
                    f"{counts['succeeded']} succeeded, {counts['errored']} errored, "
                    f"{counts['processing']} processing"
                )

                if status["status"] == "ended":
                    # discard, not remove: a fail-fast abort earlier in this same poll cycle
                    # may already have cleared `pending` while we iterate its snapshot.
                    pending.discard(batch_id)
                    completed_ids.append(batch_id)
                    _pending_batches.pop(batch_id, None)
                    # Fail fast: if a batch has too many errors, cancel the remaining
                    # batches (not just stop polling — they would keep billing).
                    total = counts["succeeded"] + counts["errored"]
                    if total > 0 and counts["errored"] / total > FAIL_FAST_ERROR_FRACTION:
                        logger.error(
                            f"Batch {batch_id}: {counts['errored']}/{total} errored "
                            f"(>{FAIL_FAST_ERROR_FRACTION:.0%}). Cancelling {len(pending)} remaining batch(es)."
                        )
                        for bid in list(pending):
                            self.cancel(bid)
                        # A cancelled batch still has whatever it already completed, and that
                        # work is already billed. Collect it rather than discarding it, then
                        # fail loudly below so a partial grid can never be mistaken for a
                        # complete one.
                        aborted_ids.extend(pending)
                        pending.clear()

        # Collect from aborted batches too: their succeeded rows are already paid for.
        for batch_id in completed_ids + aborted_ids:
            n_succeeded = 0
            n_errored = 0
            for result in self.get_results(batch_id):
                cid = result["custom_id"]
                if result["result"]["type"] == "succeeded":
                    message = result["result"]["message"]
                    response_text = self._extract_response_text(message)
                    all_results[cid] = response_text
                    self.last_stop_reasons[cid] = getattr(message, "stop_reason", None)
                    n_succeeded += 1

                    # Cache result
                    if cache is not None and cache_keys is not None:
                        ck = cache_keys.get(cid)
                        if ck and response_text is not None:
                            cache.set(ck, response_text)

                    if on_result:
                        on_result(cid, response_text)
                else:
                    all_results[cid] = None
                    n_errored += 1
                    error = result["result"].get("error")
                    logger.warning(f"Batch request {cid} failed: {error}")

            logger.info(f"Batch {batch_id} complete: {n_succeeded} succeeded, {n_errored} errored")

        if aborted_ids:
            missing = sum(1 for v in all_results.values() if v is None)
            raise RuntimeError(
                f"Batch grid aborted: {len(aborted_ids)} sub-batch(es) cancelled after a "
                f"fail-fast trigger ({aborted_ids}). Partial results for the batches that had "
                f"already completed were collected ({missing} custom_ids have no response), but "
                f"this grid is INCOMPLETE and must not be treated as a finished run."
            )
        return all_results


def build_batch_request(
    custom_id: str,
    messages: list[dict],
    model: str = "claude-haiku-4-5-20251001",
    max_tokens: int = 4096,
    temperature: float = 1.0,
    thinking: bool = False,
    thinking_budget: int = 4000,
    system: str = None,
    cache_system: bool = True,
    top_p: float = 1.0,
    prefill_tag: str = None,
) -> dict:
    """Build a single batch request dict (Batches API supports all Messages API params).

    cache_system marks the system prompt with cache_control (5-minute ephemeral):
    identical system prompts across requests read from the prompt cache at 0.1x input
    price (best-effort inside batches). Semantically identical either way.
    top_p is included when != 1.0 (Claude 4+ rejects temperature and top_p together).
    prefill_tag appends an assistant-turn prefill (pre-4.6 models only; incompatible
    with thinking — callers must reject that combination before building).
    """
    if prefill_tag:
        messages = messages + [{"role": "assistant", "content": prefill_tag}]
    params = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if system:
        if cache_system:
            params["system"] = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        else:
            params["system"] = system
    # Lazy import avoids a module-load cycle (base.py -> batch_api.py on some paths).
    from src.evals.base import adaptive_thinking_params, is_adaptive_thinking_model

    if is_adaptive_thinking_model(model):
        # These model ids reject sampling params and take adaptive or disabled thinking only.
        params.update(adaptive_thinking_params(model, thinking))
        if thinking:
            params["max_tokens"] = max(max_tokens, 2048)  # room for adaptive thinking + answer
    elif thinking:
        params["temperature"] = 1.0
        params["thinking"] = {
            "type": "enabled",
            "budget_tokens": thinking_budget,
        }
        # Ensure max_tokens is large enough for thinking + response
        params["max_tokens"] = max(max_tokens, thinking_budget + 2048)
    elif top_p != 1.0:
        # The API rejects temperature and top_p together; omitting temperature == 1.0.
        params["top_p"] = top_p
    else:
        params["temperature"] = temperature

    return {"custom_id": custom_id, "params": params}


class BatchAnthropicGenerator:
    """Anthropic generation through the Message Batches API (50% cheaper).

    Same generate_chat() interface as AnthropicGenerator in src/evals/base.py, so any
    eval can use it unchanged via create_generator(model, use_batch=True). Trade-off:
    results arrive when the whole batch ends (usually minutes, up to 24h), so this is
    for grids and overnight runs, not interactive use.

    Parameter support mirrors AnthropicGenerator: seed is an error (the API has no
    seed parameter); top_p passes through when set; prefill_tag is an assistant-turn
    prefill; top_p/prefill_tag with thinking=True raise.
    """

    def __init__(
        self,
        model_name: str = "claude-haiku-4-5-20251001",
        api_key: str = None,
        cache_system: bool = True,
        poll_interval: int = DEFAULT_POLL_INTERVAL,
        timeout: int | None = None,
    ):
        self.model_name = model_name
        self.model_path = model_name  # For cache key compatibility
        self.supports_thinking = True
        self.cache_system = cache_system
        self.poll_interval = poll_interval
        self.timeout = timeout
        self.client = BatchClient(api_key=api_key)

    def load(self):
        pass

    def generate_chat(
        self,
        conversations: list[list[dict]],
        max_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 1.0,
        seed: int = None,
        thinking: bool = False,
        prefill_tag: str = None,
        on_response: callable = None,
    ) -> list[str]:
        """Generate completions for all conversations via one batch submission.

        Returns responses in input order; failed requests are None (same contract as
        the other generators).
        """
        from src.evals.base import _run_async, reject_unsupported_params

        reject_unsupported_params("BatchAnthropicGenerator", {"seed": (seed, None)})
        if thinking:
            reject_unsupported_params(
                "BatchAnthropicGenerator with thinking=True",
                {"top_p": (top_p, 1.0), "prefill_tag": (prefill_tag, None)},
            )
        if top_p != 1.0 and temperature != 1.0:
            raise ValueError(
                "The Anthropic API rejects temperature and top_p together. Pass temperature=1.0 "
                "explicitly when using top_p — omitting temperature is equivalent to 1.0."
            )

        requests = []
        for idx, messages in enumerate(conversations):
            system_content = None
            api_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    system_content = msg["content"]
                else:
                    api_messages.append(msg)

            # Mirror AnthropicGenerator's thinking budget so batch and non-batch
            # runs of the same eval are comparable.
            thinking_budget = max(THINKING_BUDGET_MIN, min(max_tokens // 2, THINKING_BUDGET_MAX))
            requests.append(
                build_batch_request(
                    custom_id=str(idx),
                    messages=api_messages,
                    model=self.model_name,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    thinking=thinking,
                    thinking_budget=thinking_budget,
                    system=system_content,
                    cache_system=self.cache_system,
                    top_p=top_p,
                    prefill_tag=prefill_tag,
                )
            )

        def _with_prefill(text: str | None) -> str | None:
            # vLLM-convention parity: stored response includes the prefill.
            if text is not None and prefill_tag:
                return f"{prefill_tag}\n{text}"
            return text

        def _on_result(custom_id: str, response_text: str) -> None:
            if on_response and response_text is not None:
                on_response(int(custom_id), _with_prefill(response_text))

        results = _run_async(
            self.client.submit_and_wait(
                requests,
                poll_interval=self.poll_interval,
                timeout=self.timeout,
                on_result=_on_result,
            )
        )
        # Aligned with input conversations; "max_tokens" = truncated.
        batch_stop_reasons = getattr(self.client, "last_stop_reasons", {})
        self.last_stop_reasons = [batch_stop_reasons.get(str(i)) for i in range(len(conversations))]
        return [_with_prefill(results.get(str(i))) for i in range(len(conversations))]

    def cleanup(self):
        pass
