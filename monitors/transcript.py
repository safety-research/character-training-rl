"""Render one evaluation episode as the text a monitor reads, with the model's reasoning shown or hidden.

Both renderings hold every message of the episode in order, each under a `MESSAGE <index> ROLE <role>` header (with
`TOOL <name>` for a tool result), then its content blocks, its tool calls as JSON and any tool error.

- reasoning shown: everything, the reasoning blocks and the contents of `think` tool calls included.
- reasoning hidden: the same transcript with the reasoning blocks removed, the arguments of each `think` tool call
  replaced by `{"reasoning_channel_omitted": true}`, and the result (and any error) of a `think` call removed. The
  message headers stay, so the two renderings have the same message indices.
"""

from __future__ import annotations

import json

REASONING_SHOWN = "reasoning_shown"
REASONING_HIDDEN = "reasoning_hidden"
VIEWS = (REASONING_SHOWN, REASONING_HIDDEN)


def encode(x) -> bytes:
    """Compact JSON with sorted keys, the one serialization of tool calls and of stored rows."""
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def views(sample) -> tuple[dict[str, str], int, int]:
    """The two renderings of an Inspect sample, keyed by VIEWS, with the number of characters of reasoning-channel
    text and of `think`-tool text that the reasoning-hidden rendering leaves out."""
    shown, hidden = [], []
    think_ids = set()
    for message in sample.messages:
        written = (getattr(message, "metadata", None) or {}).get("unparsed_tool_calls", {})
        for call in getattr(message, "tool_calls", None) or []:
            if call.function == "think" or written.get(call.id, {}).get("function") == "think":
                think_ids.add(call.id)
    native_chars, think_chars = 0, 0
    for i, m in enumerate(sample.messages):
        prefix = f"\nMESSAGE {i} ROLE {m.role}"
        function = getattr(m, "function", None)
        if function:
            prefix += f" TOOL {function}"
        shown.append(prefix + "\n")
        hidden.append(prefix + "\n")
        blocks = (
            [("text", m.content)]
            if isinstance(m.content, str)
            else [
                (b.type, b.text if b.type == "text" else b.reasoning if b.type == "reasoning" else None)
                for b in m.content
            ]
        )
        think_result = getattr(m, "tool_call_id", None) in think_ids
        for kind, text in blocks:
            if text is None:
                raise ValueError(f"unsupported content block: {kind}")
            marked = f"[{kind}]\n{text}\n"
            shown.append(marked)
            if kind == "reasoning":
                native_chars += len(text)
            elif think_result:
                think_chars += len(text)
            else:
                hidden.append(marked)
        metadata = getattr(m, "metadata", None) or {}
        unparsed = metadata.get("unparsed_tool_calls", {})
        calls = getattr(m, "tool_calls", None) or []
        if metadata.get("raw_turn"):
            raw_tool_turn = "RAW_TOOL_TURN\n" + metadata["raw_turn"] + "\n"
            shown.append(raw_tool_turn)
            # raw_turn is the provider's raw serialization of a turn that holds only tool calls, not a reasoning
            # block, so it stays when the reasoning is hidden, unless the turn holds a think call: its other calls
            # are each kept separately below.
            if not any(c.id in think_ids for c in calls):
                hidden.append(raw_tool_turn)
        for c in calls:
            call = {"id": c.id, "function": c.function, "arguments": c.arguments, "parse_error": c.parse_error}
            if c.id in unparsed:
                call["written_call"] = unparsed[c.id]
            shown.append("TOOL_CALL\n" + encode(call).decode() + "\n")
            if c.id in think_ids:
                think_chars += len(encode(c.arguments).decode())
                call = {**call, "arguments": {"reasoning_channel_omitted": True}}
                if "written_call" in call:
                    call["written_call"] = {"function": "think", "reasoning_channel_omitted": True}
            hidden.append("TOOL_CALL\n" + encode(call).decode() + "\n")
        error = getattr(m, "error", None)
        if error:
            line = "TOOL_ERROR " + str(error) + "\n"
            shown.append(line)
            if not think_result:
                hidden.append(line)
    return {REASONING_SHOWN: "".join(shown), REASONING_HIDDEN: "".join(hidden)}, native_chars, think_chars
