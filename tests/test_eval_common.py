"""Tests for the shared eval helpers (common.py) and the judge runner's guards."""

import pytest

from src.evals.base import ResponseCache
from src.evals.common import generate_with_cache, reject_greedy_thinking, strip_thinking
from src.evals.judge_runner import judge_request_params, run_judge_cached


class FakeGenerator:
    """Records what it was asked to generate; returns canned responses/stop reasons."""

    def __init__(self, responses, stop_reasons):
        self._responses = responses
        self._stop_reasons = stop_reasons
        self.calls = []
        self.kwargs = []
        self.last_stop_reasons = []

    def generate_chat(self, conversations, **kwargs):
        self.calls.append(len(conversations))
        self.kwargs.append(kwargs)
        out = self._responses[: len(conversations)]
        self.last_stop_reasons = self._stop_reasons[: len(conversations)]
        on_response = kwargs.get("on_response")
        if on_response:
            for i, r in enumerate(out):
                if r is not None:
                    on_response(i, r)
        return out


def test_generate_with_cache_persists_stop_reason_and_serves_hits(tmp_path):
    cache = ResponseCache(tmp_path / "c")
    gen = FakeGenerator(["answer one", "answer two"], ["stop", "length"])
    convs = [[{"role": "user", "content": "a"}], [{"role": "user", "content": "b"}]]
    keys = ["k1", "k2"]
    responses, stops = generate_with_cache(gen, convs, keys, cache, max_tokens=10, temperature=0.7, thinking=False)
    assert responses == ["answer one", "answer two"]
    assert stops == ["stop", "length"]
    # A second run is served fully from cache, stop reasons intact.
    gen2 = FakeGenerator([], [])
    responses2, stops2 = generate_with_cache(gen2, convs, keys, cache, max_tokens=10, temperature=0.7, thinking=False)
    assert gen2.calls == []
    assert responses2 == responses and stops2 == stops


def test_generate_with_cache_treats_metaless_entries_as_misses(tmp_path):
    # An entry written by the incremental callback of an interrupted run has no
    # stop_reason metadata; it must be regenerated, not trusted as complete.
    cache = ResponseCache(tmp_path / "c")
    cache.set("k1", "truncated text with no meta")
    gen = FakeGenerator(["fresh full answer"], ["stop"])
    responses, stops = generate_with_cache(
        gen, [[{"role": "user", "content": "a"}]], ["k1"], cache, max_tokens=10, temperature=0.7, thinking=False
    )
    assert gen.calls == [1]
    assert responses == ["fresh full answer"]
    assert stops == ["stop"]


def test_generate_with_cache_passes_explicit_top_p(tmp_path):
    cache = ResponseCache(tmp_path / "c")
    gen = FakeGenerator(["answer"], ["stop"])
    generate_with_cache(
        gen,
        [[{"role": "user", "content": "a"}]],
        ["k1"],
        cache,
        max_tokens=10,
        temperature=1.0,
        thinking=True,
        top_p=0.95,
    )
    assert gen.kwargs[0]["top_p"] == 0.95


def test_generate_with_cache_raises_on_stop_reason_misalignment(tmp_path):
    cache = ResponseCache(tmp_path / "c")
    gen = FakeGenerator(["a", "b"], ["stop"])  # one stop reason for two conversations
    with pytest.raises(RuntimeError, match="stop reasons"):
        generate_with_cache(
            gen,
            [[{"role": "user", "content": "a"}], [{"role": "user", "content": "b"}]],
            ["k1", "k2"],
            cache,
            max_tokens=10,
            temperature=0.7,
            thinking=False,
        )


def test_reject_greedy_thinking():
    with pytest.raises(ValueError, match="greedy"):
        reject_greedy_thinking(True, 0.0)
    reject_greedy_thinking(False, 0.0)
    reject_greedy_thinking(True, 0.7)


def test_strip_thinking():
    assert strip_thinking("<think>\nx</think>\n\nanswer") == "answer"
    assert strip_thinking("plain") == "plain"


def test_judge_request_params_adaptive_thinking_omits_temperature():
    params = judge_request_params("claude-opus-4-8")
    assert "temperature" not in params
    assert params == {"thinking": {"type": "disabled"}}


def test_run_judge_cached_rejects_misaligned_inputs():
    with pytest.raises(ValueError, match="key inputs"):
        run_judge_cached("j", "rubric", ["u1", "u2"], [{"a": 1}], {}, lambda m: None, 10, False, None)
