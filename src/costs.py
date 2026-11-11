"""The one pricing table every cost estimate reads, so estimates cannot drift by copy.

Everything here is for ESTIMATES ONLY: a number printed before spending so a run's cost is never a
surprise. The invoice is the truth; reconcile against it, not against this.

Every row carries the date it was checked. A row may also carry `expires`, a provider-announced end
date; no row in the table has one today, so `expired_rows` returns nothing for it and the test that
calls it on the real table checks nothing until a row gains the field. `tests/test_costs.py`
exercises `expired_rows` on a synthetic expired row.
"""

from __future__ import annotations

from datetime import date

# List prices in USD per million tokens as of each row's "checked" date: "in" is prompt, "out" is
# completion. "expires" is a provider-announced end date, if any. Check the provider's pricing page
# before relying on them.
PRICING: dict[str, dict] = {
    # OpenAI: the judges the MASK benchmark specifies. o3-mini's hidden reasoning tokens bill as output.
    "gpt-4o": {"in": 2.50, "out": 10.00, "checked": "2026-10-08"},
    "o3-mini": {"in": 1.10, "out": 4.40, "checked": "2026-10-08"},
    # Anthropic, standard (non-batch) rates.
    "claude-opus-4-6": {"in": 5.0, "out": 25.0, "checked": "2026-08-28"},
    "claude-opus-4-8": {"in": 5.0, "out": 25.0, "checked": "2026-08-28"},
    "claude-sonnet-4-6": {"in": 3.0, "out": 15.0, "checked": "2026-08-28"},
    "claude-sonnet-4-5": {"in": 3.0, "out": 15.0, "checked": "2026-10-13"},
    "claude-haiku-4-5-20251001": {"in": 1.0, "out": 5.0, "checked": "2026-08-28"},
}

# The Anthropic Message Batches API bills half the streaming rate.
BATCH_DISCOUNT = 0.5

# Anthropic prompt-cache read multiplier: a cached system prompt bills at 0.1x input rate.
CACHE_READ_MULT = 0.1

# Chars-per-token for turning estimated text volume into token counts. Deliberately conservative:
# English prose runs about 5 characters per token, so 4 overestimates token counts, and therefore
# cost, by about 25% on prose. An estimate printed before spending should err high; provider
# tokenizers also vary, and JSON-heavy prompts tokenize worse than prose.
CHARS_PER_TOKEN = 4

# Tinker list prices in USD per million tokens, by base model, as of TINKER_CHECKED. Training bills
# forward+backward ("train"); evaluation bills input at the prefill rate and generated tokens at the
# sample rate. Check Tinker's pricing page before relying on them.
TINKER_TRAIN_USD_PER_MTOK = {
    "openai/gpt-oss-20b": 0.396,
    "openai/gpt-oss-120b": 0.737,
    "Qwen/Qwen3-8B": 0.44,
    "Qwen/Qwen3.6-35B-A3B": 1.177,
}
TINKER_PREFILL_USD_PER_MTOK = {
    "openai/gpt-oss-20b": 0.18,
    "openai/gpt-oss-120b": 0.33,
    "Qwen/Qwen3.6-35B-A3B": 0.44,
}
# Prefill tokens that hit Tinker's prompt cache bill at this rate (an 80% discount on the prefill rate).
# In a multi-turn episode whose observations extend each other, every token is prefilled at full
# price once and at this rate on every later turn.
TINKER_CACHED_PREFILL_USD_PER_MTOK = {
    "openai/gpt-oss-20b": 0.036,
    "openai/gpt-oss-120b": 0.066,
}
TINKER_SAMPLE_USD_PER_MTOK = {
    "openai/gpt-oss-20b": 0.45,
    "openai/gpt-oss-120b": 0.84,
    "Qwen/Qwen3.6-35B-A3B": 1.335,
}
TINKER_CHECKED = "2026-08-04"


def usd_per_mtok(model: str) -> dict:
    """The row for a model; unknown models are an error, never a guess."""
    if model not in PRICING:
        raise KeyError(f"no pricing for {model!r}; add its rate to src/costs.py PRICING")
    return PRICING[model]


def estimate_usd(model: str, in_tokens: float, out_tokens: float) -> float:
    """Estimated dollars for a (prompt, completion) token volume at list rates."""
    p = usd_per_mtok(model)
    return (in_tokens * p["in"] + out_tokens * p["out"]) / 1_000_000


def judge_call_usd(judge_model: str, *, use_batch: bool, **tokens: int) -> float:
    """Estimated dollars for one judge call given its token assumption, e.g. in=1500, out=100.
    Batch calls bill at the batch discount."""
    if set(tokens) != {"in", "out"}:
        raise ValueError(f"expected in= and out= token counts, got {sorted(tokens)}")
    return estimate_usd(judge_model, tokens["in"], tokens["out"]) * (BATCH_DISCOUNT if use_batch else 1.0)


def expired_rows(today: date | None = None) -> list[str]:
    """Models whose announced price expiry has passed. Non-empty means the table is wrong."""
    today = today or date.today()
    return [m for m, row in PRICING.items() if "expires" in row and date.fromisoformat(row["expires"]) < today]


CACHE_WRITE_MULT = 1.25
"""Anthropic prompt caching: a cache write bills at 1.25x the input rate, a cache read at CACHE_READ_MULT."""


def tinker_prices(model: str) -> tuple[float, float]:
    """(sample, prefill) USD per million tokens for a Tinker-sampled model; unknown is an error.
    Inspect names carry a provider prefix (`tinker/openai/gpt-oss-20b`), which is dropped."""
    base = model if model in TINKER_SAMPLE_USD_PER_MTOK else (model.split("/", 1)[1] if "/" in model else model)
    if base not in TINKER_SAMPLE_USD_PER_MTOK or base not in TINKER_PREFILL_USD_PER_MTOK:
        raise ValueError(f"no Tinker sample/prefill price for {base!r} in src/costs.py; refusing to estimate")
    return TINKER_SAMPLE_USD_PER_MTOK[base], TINKER_PREFILL_USD_PER_MTOK[base]


def usage_usd(
    model_key: str, input_tokens: int, output_tokens: int, cache_write: int = 0, cache_read: int = 0
) -> float:
    """Dollars for one model's recorded usage. `tinker/<base>` at the Tinker rates; any other
    provider at the price table row for the bare model name, cache tokens at their multipliers."""
    provider, _, name = model_key.partition("/")
    if provider == "tinker":
        sample_rate, prefill_rate = tinker_prices(model_key)
        return (output_tokens * sample_rate + input_tokens * prefill_rate) / 1e6
    row = usd_per_mtok(name)
    inputs = input_tokens + cache_write * CACHE_WRITE_MULT + cache_read * CACHE_READ_MULT
    return (inputs * row["in"] + output_tokens * row["out"]) / 1e6
