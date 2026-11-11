"""Tests for shared trainer machinery in tinker_common."""

import math

import pytest

from src.training.tinker_common import (
    TINKER_TRAIN_USD_PER_MTOK,
    estimate_train_tokens,
    format_cost_estimate,
    json_dumps_strict,
    validate_json_metric,
    verify_reference_client,
)


class _FakeTokenizer:
    def __init__(self, ids):
        self._ids = ids

    def encode(self, _text):
        return self._ids


class _FakeClient:
    def __init__(self, ids):
        self._tok = _FakeTokenizer(ids)

    def get_tokenizer(self):
        return self._tok


def test_estimate_train_tokens_is_mean_times_batch_times_steps():
    # mean 10 tokens/row x batch 4 x 3 steps = 120
    assert estimate_train_tokens([8, 12, 10], batch_size=4, max_steps=3) == 120


def test_estimate_train_tokens_empty_is_zero():
    assert estimate_train_tokens([], batch_size=4, max_steps=3) == 0


def test_format_cost_estimate_known_model_shows_dollars():
    rate = TINKER_TRAIN_USD_PER_MTOK["Qwen/Qwen3-8B"]
    msg = format_cost_estimate("Qwen/Qwen3-8B", 1_000_000)
    assert f"${rate:.2f}/Mtok" in msg
    assert f"${rate * 1_000_000 / 1_000_000:,.2f}" in msg


def test_format_cost_estimate_unknown_model_is_tokens_only_no_dollars():
    msg = format_cost_estimate("some/unlisted-model", 5_000)
    assert "5,000 train tokens" in msg
    assert "no train $/Mtok rate on file" in msg
    assert "= ~$" not in msg  # no computed dollar figure for an unlisted model


def test_validate_json_metric_accepts_finite_rejects_nonfinite_and_nonnumeric():
    validate_json_metric(1.5, context="m")  # no raise
    for bad in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError, match="finite"):
            validate_json_metric(bad, context="m")
    with pytest.raises(ValueError, match="finite numeric"):
        validate_json_metric("not a number", context="m")
    with pytest.raises(ValueError, match="finite numeric"):
        validate_json_metric(True, context="m")


def test_json_dumps_strict_rejects_nan():
    json_dumps_strict({"ok": 1.0})  # no raise
    with pytest.raises(ValueError):
        json_dumps_strict({"bad": math.nan})


async def test_verify_reference_client_accepts_matching_tokenizer():
    await verify_reference_client(_FakeClient([1, 2, 3]), _FakeClient([1, 2, 3]))  # no raise


async def test_verify_reference_client_rejects_mismatched_tokenizer():
    with pytest.raises(ValueError, match="tokenizer does not match"):
        await verify_reference_client(_FakeClient([1, 2, 3]), _FakeClient([9, 9]))


class _Renderer:
    def __init__(self, dated: bool):
        self.use_system_prompt = dated
        self.current_date = None


def test_apply_renderer_date_defaults_to_the_fixed_date_and_is_rejected_for_undated_renderers():
    from src.training.tinker_common import DEFAULT_RENDERER_DATE, apply_renderer_date

    dated = _Renderer(dated=True)
    apply_renderer_date(dated, "2026-01-01")
    assert dated.current_date == "2026-01-01"
    unset = _Renderer(dated=True)
    apply_renderer_date(unset, None)
    assert unset.current_date == DEFAULT_RENDERER_DATE == "2026-08-20"
    undated = _Renderer(dated=False)
    apply_renderer_date(undated, None)
    with pytest.raises(ValueError):
        apply_renderer_date(undated, "2026-08-30")
