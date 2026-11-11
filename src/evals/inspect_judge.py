"""A forced strict tool call to an Inspect model role, for scorers that judge with a schema.

The judge is addressed by role (`get_model(role=...)`), so the task never names a provider and a
test can hand it a mock. The tool is the same dict shape `src.evals.judge_runner` uses, so one
rubric and one schema serve both the Inspect scorer and a rollout that judges through the SDK.
Anything other than exactly one call to the named tool returns None: a judge error, never a zero.

Strictness. Anthropic constrains a tool's input to its schema (grammar-constrained sampling) only
when the tool definition carries `strict: true`. Inspect's Anthropic provider builds each tool from
its name, description and input schema and has no strict field, so `send_strict_tools` wraps one
provider instance to add the flag to every tool marked with `STRICT_TOOL_OPTION` in its options.
`src.keys.inspect_claude_model` applies it to every Claude model it builds, and `forced_tool_call`
refuses a judge whose provider does not declare `SENDS_STRICT_TOOLS`, so a call made through
`forced_tool_call` never goes out unconstrained. One judge does not go through `forced_tool_call`:
the StrongREJECT judge (`src.evals.strongreject.task.judge`, also used by `surf_sr`) builds its own
tool without the strict mark, so it is a forced tool call whose reply is validated against the schema
in code, without the provider's strict mode.

Temperature. Every Inspect judge call sends temperature 0. Some Claude models take no sampling
parameters (`src.evals.judge_runner.judge_request_params` sends them none), so
`require_pinned_judge_temperature` refuses such a judge model; `src.keys.inspect_claude_model` calls
it, so every evaluation that builds its Claude judge there stops before any call.
"""

from __future__ import annotations

from inspect_ai.model import ChatMessageSystem, ChatMessageUser, GenerateConfig, get_model
from inspect_ai.tool import ToolFunction, ToolInfo, ToolParams
from inspect_ai.util._json import json_schema_dump

JUDGE_TEMPERATURE = 0.0

STRICT_TOOL_OPTION = "strict"
"""Key in `ToolInfo.options` that asks the provider to send the tool with `strict: true`."""

SENDS_STRICT_TOOLS = "sends_strict_tools"
"""Attribute a ModelAPI instance sets to True when it sends tools marked strict with `strict: true`."""


def require_pinned_judge_temperature(judge_model: str) -> None:
    """Refuse a judge model that is not a claude-* id (ValueError) or that does not take temperature 0
    (SystemExit). The Inspect judge calls always send temperature 0, and some Claude models reject
    sampling parameters; see `src.evals.judge_runner.judge_request_params`."""
    from src.evals.judge_runner import judge_request_params
    from src.keys import require_standard_claude

    require_standard_claude(judge_model, "judge")
    params = judge_request_params(judge_model)
    if params != {"temperature": JUDGE_TEMPERATURE}:
        raise SystemExit(
            f"judge {judge_model!r} takes {params}, not temperature {JUDGE_TEMPERATURE}; the Inspect judge "
            "call pins temperature 0, so choose a judge model that accepts a temperature"
        )


def tool_info(tool: dict) -> ToolInfo:
    """The Inspect ToolInfo for a strict judge_runner-shaped tool dict, marked strict for the provider.
    The input schema is carried whole (nested objects and arrays included) and must survive Inspect's
    schema model unchanged, so the schema the API constrains to is exactly the one written."""
    if tool.get("strict") is not True:
        raise ValueError(f"judge tool {tool.get('name')!r} must declare strict: True")
    params = ToolParams.model_validate(tool["input_schema"])
    if json_schema_dump(params) != tool["input_schema"]:
        raise ValueError(f"judge tool {tool['name']!r}: Inspect's schema model changes its input_schema")
    return ToolInfo(
        name=tool["name"], description=tool["description"], parameters=params, options={STRICT_TOOL_OPTION: True}
    )


def send_strict_tools(api) -> None:
    """Make this Inspect Anthropic provider instance send `strict: true` on every tool whose options
    carry STRICT_TOOL_OPTION, and declare SENDS_STRICT_TOOLS on it. A strict tool that Inspect would
    send as a native (server-side) tool, or as anything other than one plain tool param, is refused."""
    original = api.tool_params_for_tool_info

    def tool_params_for_tool_info(tool: ToolInfo, config: GenerateConfig):
        params = original(tool, config)
        if not (tool.options or {}).get(STRICT_TOOL_OPTION):
            return params
        if len(params) != 1 or "input_schema" not in params[0]:
            raise RuntimeError(f"strict tool {tool.name!r} did not become one plain tool param: {params!r}")
        return [{**params[0], "strict": True}]

    api.tool_params_for_tool_info = tool_params_for_tool_info
    setattr(api, SENDS_STRICT_TOOLS, True)


async def forced_tool_call(role: str, system: str | None, user: str, tool: dict, max_tokens: int) -> dict | None:
    """The tool's arguments from one forced strict call at temperature 0, or None on non-conformance:
    a reply cut at `max_tokens`, or anything but exactly one call to the tool. `system` None sends the
    user turn alone. Raises when the role's provider cannot send a strict tool."""
    info = tool_info(tool)
    model = get_model(role=role)
    if getattr(model.api, SENDS_STRICT_TOOLS, False) is not True:
        raise RuntimeError(
            f"judge role {role!r} ({model}) does not send strict tools; build it with src.keys.inspect_claude_model"
        )
    messages = [ChatMessageUser(content=user)]
    if system is not None:
        messages.insert(0, ChatMessageSystem(content=system))
    out = await model.generate(
        messages,
        tools=[info],
        tool_choice=ToolFunction(name=info.name),
        config=GenerateConfig(max_tokens=max_tokens, temperature=JUDGE_TEMPERATURE),
    )
    if out.stop_reason == "max_tokens":
        return None
    calls = out.message.tool_calls or []
    if len(calls) != 1 or calls[0].function != info.name:
        return None
    args = calls[0].arguments
    return dict(args) if isinstance(args, dict) else None
