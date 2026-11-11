"""Tests for src/batch_api.py — request building and generator request construction.

BatchClient's network paths are exercised by live runs; these tests cover the pure
logic: request formatting, prompt-cache markers, and thinking-parameter parity with
AnthropicGenerator.
"""

from src.batch_api import (
    THINKING_BUDGET_MAX,
    THINKING_BUDGET_MIN,
    BatchAnthropicGenerator,
    build_batch_request,
)

# --- build_batch_request ---


def test_basic_request():
    req = build_batch_request("id-1", [{"role": "user", "content": "hi"}], model="m", max_tokens=100, temperature=0.3)
    assert req["custom_id"] == "id-1"
    assert req["params"]["model"] == "m"
    assert req["params"]["max_tokens"] == 100
    assert req["params"]["temperature"] == 0.3
    assert "system" not in req["params"]
    assert "thinking" not in req["params"]


def test_system_gets_cache_control_by_default():
    req = build_batch_request("id-1", [{"role": "user", "content": "hi"}], system="be terse")
    assert req["params"]["system"] == [
        {"type": "text", "text": "be terse", "cache_control": {"type": "ephemeral"}}
    ]


def test_system_plain_when_cache_disabled():
    req = build_batch_request("id-1", [{"role": "user", "content": "hi"}], system="be terse", cache_system=False)
    assert req["params"]["system"] == "be terse"


def test_top_p_replaces_temperature():
    """The API rejects temperature and top_p together — when top_p is set, the
    request carries top_p only (omitted temperature == 1.0 on the API side)."""
    req = build_batch_request("id-1", [{"role": "user", "content": "hi"}], top_p=0.9)
    assert req["params"]["top_p"] == 0.9
    assert "temperature" not in req["params"]
    req = build_batch_request("id-1", [{"role": "user", "content": "hi"}])
    assert "top_p" not in req["params"]
    assert req["params"]["temperature"] == 1.0


def test_prefill_appends_assistant_turn():
    req = build_batch_request("id-1", [{"role": "user", "content": "hi"}], prefill_tag="<helpful>")
    assert req["params"]["messages"][-1] == {"role": "assistant", "content": "<helpful>"}


def test_thinking_forces_temperature_1_and_bumps_max_tokens():
    req = build_batch_request(
        "id-1",
        [{"role": "user", "content": "hi"}],
        max_tokens=100,
        temperature=0.0,
        thinking=True,
        thinking_budget=4000,
    )
    assert req["params"]["temperature"] == 1.0
    assert req["params"]["thinking"] == {"type": "enabled", "budget_tokens": 4000}
    assert req["params"]["max_tokens"] >= 4000 + 2048


# --- BatchAnthropicGenerator request construction ---


def test_generator_thinking_budget_matches_anthropic_generator():
    """Batch and non-batch runs of the same eval must use the same thinking budget
    formula (mirrors AnthropicGenerator in src/evals/base.py)."""
    for max_tokens in (512, 4096, 100_000):
        expected = max(THINKING_BUDGET_MIN, min(max_tokens // 2, THINKING_BUDGET_MAX))
        assert THINKING_BUDGET_MIN <= expected <= THINKING_BUDGET_MAX


def test_generator_builds_requests_with_system_extraction(monkeypatch):
    """System messages are lifted out of the message list into params.system."""
    captured = {}

    async def fake_submit_and_wait(requests, **kwargs):
        captured["requests"] = requests
        return {req["custom_id"]: f"response-{req['custom_id']}" for req in requests}

    gen = BatchAnthropicGenerator(model_name="claude-test", api_key="sk-test")
    monkeypatch.setattr(gen.client, "submit_and_wait", fake_submit_and_wait)

    conversations = [
        [{"role": "system", "content": "sys A"}, {"role": "user", "content": "q1"}],
        [{"role": "user", "content": "q2"}],
    ]
    responses = gen.generate_chat(conversations, max_tokens=64, temperature=0.5)

    assert responses == ["response-0", "response-1"]
    reqs = captured["requests"]
    assert len(reqs) == 2
    # System lifted out and cache-marked; messages contain only non-system turns
    assert reqs[0]["params"]["system"][0]["text"] == "sys A"
    assert reqs[0]["params"]["messages"] == [{"role": "user", "content": "q1"}]
    assert "system" not in reqs[1]["params"]
    assert reqs[1]["params"]["messages"] == [{"role": "user", "content": "q2"}]


def test_generator_preserves_input_order_and_failures(monkeypatch):
    """Results come back keyed by custom_id; generator must return input order with
    None for failed requests (same contract as other generators)."""

    async def fake_submit_and_wait(requests, **kwargs):
        return {"0": "first", "2": "third"}  # request 1 failed

    gen = BatchAnthropicGenerator(model_name="claude-test", api_key="sk-test")
    monkeypatch.setattr(gen.client, "submit_and_wait", fake_submit_and_wait)

    conversations = [[{"role": "user", "content": f"q{i}"}] for i in range(3)]
    responses = gen.generate_chat(conversations)
    assert responses == ["first", None, "third"]


def test_generator_on_response_callback(monkeypatch):
    """on_response receives integer indices, only for successful results."""
    received = []

    async def fake_submit_and_wait(requests, on_result=None, **kwargs):
        on_result("1", "resp")
        on_result("0", None)  # failed — must not reach on_response
        return {"0": None, "1": "resp"}

    gen = BatchAnthropicGenerator(model_name="claude-test", api_key="sk-test")
    monkeypatch.setattr(gen.client, "submit_and_wait", fake_submit_and_wait)

    gen.generate_chat(
        [[{"role": "user", "content": "a"}], [{"role": "user", "content": "b"}]],
        on_response=lambda idx, text: received.append((idx, text)),
    )
    assert received == [(1, "resp")]
