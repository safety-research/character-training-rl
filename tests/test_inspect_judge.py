"""The shared Inspect judge call sends a forced strict tool at temperature 0, verified on the HTTP body."""

import asyncio
import json

import anthropic
import httpx
import pytest
from inspect_ai.model import get_model
from inspect_ai.model._model import init_model_roles

import src.keys as keys
from src.evals import inspect_judge
from src.evals.judge_runner import make_categorical_tool

TOOL = make_categorical_tool("classify", ["yes", "no"], "Classify the answer.")


@pytest.fixture(autouse=True)
def placeholder_key(monkeypatch):
    """A placeholder key so no real one enters the suite; no request leaves the process."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "placeholder-not-a-key")


def _judge_with_captured_requests(monkeypatch) -> tuple[object, list[dict]]:
    """An Inspect Claude judge from `inspect_claude_model` whose HTTP layer is a mock that records each
    request body and answers with one tool_use block. No request leaves the process."""
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        return httpx.Response(
            200,
            json={
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": body["model"],
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_test",
                        "name": "classify",
                        "input": {"reasoning": "r", "category": "yes"},
                    },
                ],
                "stop_reason": "tool_use",
                "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
        )

    def client(**kwargs):
        transport = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        return anthropic.AsyncAnthropic(api_key="test-placeholder", http_client=transport, max_retries=0, **kwargs)

    judge = keys.inspect_claude_model("claude-sonnet-4-6")
    judge.api.client = client()
    return judge, bodies


def test_forced_call_sends_a_strict_forced_tool_at_temperature_zero(monkeypatch):
    judge, bodies = _judge_with_captured_requests(monkeypatch)
    init_model_roles({"judge": judge})
    args = asyncio.run(inspect_judge.forced_tool_call("judge", "rubric", "item", TOOL, 64))
    assert args == {"reasoning": "r", "category": "yes"}
    [body] = bodies
    [tool] = body["tools"]
    assert tool["strict"] is True and tool["name"] == "classify"
    assert tool["input_schema"]["additionalProperties"] is False
    assert tool["input_schema"]["properties"]["category"]["enum"] == ["yes", "no"]
    assert body["tool_choice"]["type"] == "tool" and body["tool_choice"]["name"] == "classify"
    assert body["temperature"] == 0.0 and body["max_tokens"] == 64


def test_unmarked_tools_are_sent_unchanged(monkeypatch):
    judge, _ = _judge_with_captured_requests(monkeypatch)
    info = inspect_judge.tool_info(TOOL).model_copy(update={"options": None})
    [param] = judge.api.tool_params_for_tool_info(info, judge.config)
    assert "strict" not in param


def test_a_tool_without_strict_is_refused():
    loose = {**TOOL, "strict": False}
    with pytest.raises(ValueError, match="strict"):
        inspect_judge.tool_info(loose)


def test_a_judge_that_cannot_send_strict_is_refused():
    init_model_roles({"judge": get_model("mockllm/model")})
    with pytest.raises(RuntimeError, match="does not send strict tools"):
        asyncio.run(inspect_judge.forced_tool_call("judge", "rubric", "item", TOOL, 64))


def test_nested_schema_reaches_the_api_whole_and_no_system_turn_is_sent_when_none(monkeypatch):
    from src.evals.em_categorical.eval import JUDGE_TOOL

    judge, bodies = _judge_with_captured_requests(monkeypatch)
    init_model_roles({"judge": judge})
    asyncio.run(inspect_judge.forced_tool_call("judge", None, "item", JUDGE_TOOL, 64))
    [body] = bodies
    [tool] = body["tools"]
    assert tool["strict"] is True
    assert tool["input_schema"] == JUDGE_TOOL["input_schema"]
    assert "system" not in body and [m["role"] for m in body["messages"]] == ["user"]
