"""src.costs: the one price table, its expiry check, and the judge-call estimate."""

from datetime import date

import pytest

from src import costs


def test_no_price_row_is_expired_today():
    assert costs.expired_rows() == [], "a provider-announced price expiry has passed; update PRICING"


def test_expired_rows_detects_a_past_expiry(monkeypatch):
    monkeypatch.setitem(costs.PRICING, "fake/model", {"in": 1.0, "out": 2.0, "checked": "2026-01-01", "expires": "2026-02-01"})
    assert costs.expired_rows(date(2026, 3, 1)) == ["fake/model"]
    assert costs.expired_rows(date(2026, 1, 15)) == []


def test_every_row_has_rates_and_a_checked_date():
    for model, row in costs.PRICING.items():
        assert {"in", "out", "checked"} <= set(row), model
        date.fromisoformat(row["checked"])


def test_judge_call_usd_matches_the_hand_formula_and_halves_for_batch():
    streaming = costs.judge_call_usd("claude-sonnet-4-6", use_batch=False, **{"in": 1500, "out": 100})
    assert streaming == pytest.approx((1500 * 3.0 + 100 * 15.0) / 1e6)
    assert costs.judge_call_usd("claude-sonnet-4-6", use_batch=True, **{"in": 1500, "out": 100}) == pytest.approx(streaming / 2)
    with pytest.raises(KeyError):
        costs.judge_call_usd("nope", use_batch=False, **{"in": 1, "out": 1})
    with pytest.raises(ValueError):
        costs.judge_call_usd("claude-sonnet-4-6", use_batch=False, **{"in": 1})


def test_tinker_tables_are_importable_from_both_homes():
    from src.training import tinker_common

    assert tinker_common.TINKER_SAMPLE_USD_PER_MTOK is costs.TINKER_SAMPLE_USD_PER_MTOK
