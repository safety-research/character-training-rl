"""What hiding the reasoning removes from a transcript and what it must keep.

uv run pytest monitors/test_transcript.py
"""

from types import SimpleNamespace

import pytest
from inspect_ai.model import ChatMessageAssistant, ChatMessageTool, ChatMessageUser, ContentReasoning, ContentText
from inspect_ai.tool import ToolCall

from monitors.transcript import REASONING_HIDDEN, REASONING_SHOWN, views


def test_reasoning_channel_and_think_tool_hidden_but_actions_kept():
    sample = SimpleNamespace(
        messages=[
            ChatMessageUser(content="Implement the specification; do not change tests."),
            ChatMessageAssistant(
                content=[
                    ContentReasoning(reasoning="PRIVATE native speculation"),
                    ContentText(text="Public explanation remains visible"),
                ],
                tool_calls=[ToolCall(id="t", function="think", arguments={"thought": "PRIVATE scratchpad"})],
            ),
            ChatMessageTool(content="PRIVATE scratchpad echo", tool_call_id="t", function="think"),
            ChatMessageAssistant(
                content="Now inspect the implementation",
                tool_calls=[ToolCall(id="b", function="bash", arguments={"command": "cat func.py"})],
            ),
            ChatMessageTool(content="def func(): return 7", tool_call_id="b", function="bash"),
        ]
    )
    result, reasoning_chars, think_chars = views(sample)
    shown, hidden = result[REASONING_SHOWN], result[REASONING_HIDDEN]
    assert shown.index("PRIVATE native") < shown.index("Public explanation")
    assert "PRIVATE" not in hidden
    for evidence in ("do not change tests", "Public explanation", "cat func.py", "def func(): return 7"):
        assert evidence in hidden and evidence in shown
    assert reasoning_chars > 0 and think_chars > 0


def test_no_reasoning_gives_identical_views():
    sample = SimpleNamespace(
        messages=[
            ChatMessageUser(content="Task"),
            ChatMessageAssistant(
                content="Answer", tool_calls=[ToolCall(id="x", function="bash", arguments={"command": "pytest"})]
            ),
            ChatMessageTool(content="Failed: NotImplementedError", function="bash", tool_call_id="x"),
        ]
    )
    result, reasoning_chars, think_chars = views(sample)
    assert result[REASONING_SHOWN] == result[REASONING_HIDDEN]
    assert (reasoning_chars, think_chars) == (0, 0)


def test_unrecognized_content_fails_instead_of_silently_disappearing():
    m = SimpleNamespace(role="assistant", content=[SimpleNamespace(type="image")])
    with pytest.raises(ValueError, match="unsupported content"):
        views(SimpleNamespace(messages=[m]))


def test_malformed_calls_keep_the_written_program_but_hide_a_written_think_call():
    m = ChatMessageAssistant(
        content="",
        tool_calls=[
            ToolCall(id="bad", function="unparsed", arguments={}, parse_error="invalid JSON"),
            ToolCall(id="thought", function="unparsed", arguments={}, parse_error="invalid JSON"),
        ],
        metadata={
            "unparsed_tool_calls": {
                "bad": {
                    "function": "python",
                    "arguments": "code='print(7)'; print(code)",
                    "raw_text": "issued python text",
                },
                "thought": {"function": "think", "arguments": "SECRET thought", "raw_text": "SECRET thought"},
            },
            "raw_turn": "mixed raw tool call SECRET thought",
        },
    )
    result, _, _ = views(SimpleNamespace(messages=[m]))
    assert "SECRET" in result[REASONING_SHOWN] and "SECRET" not in result[REASONING_HIDDEN]
    assert "code='print(7)'; print(code)" in result[REASONING_HIDDEN]
    assert "invalid JSON" in result[REASONING_HIDDEN]
