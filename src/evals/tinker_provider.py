"""Inspect model provider for Tinker models: `tinker/<base model>`.

The prompt is rendered by the repo's TinkerGenerator, so an Inspect task samples a checkpoint
exactly as every other path in this repo does (tokenizer, harmony renderer, pinned date,
declared thinking). Inspect owns concurrency and retries; the generator owns one sampling call.

model_args (all required except the last two):
    tokenizer_id          HF id of the base model's tokenizer
    renderer_name         Tinker cookbook renderer, e.g. gpt_oss_low_reasoning
    thinking              bool; declared, never inferred
    max_connections       in-flight sampling requests
    checkpoint            a tinker:// sampler path; the model name is then its base model
    renderer_current_date YYYY-MM-DD pinned into a dated renderer; unset means the fixed default
    reasoning_prefill     text the reasoning channel starts with on every sample (harmony renderers,
                          thinking only, no tools); unset means the model writes all of its reasoning

Generation settings (temperature, top_p, max_tokens) must be given on the eval; None is an error.

System messages. A system message renders as the cookbook renders one: a harmony developer message
wrapped in "# Instructions". One whose metadata sets HARMONY_ROLE to "developer" renders as a harmony
developer message holding its text verbatim, which is how a developer message written directly in
harmony looks (harmony renderers only).

Reasoning prefill. With `reasoning_prefill` set, every prompt ends with the assistant turn's
analysis-channel header followed by the prefill, and the model continues from there. The recorded
completion is the prefill followed by what the model sampled, so the reasoning block in the log
starts with the prefill exactly as the model saw it; usage counts the prefill as input.
Registered under the `inspect_ai` entry point in pyproject, so `--model tinker/...` resolves
without an import.

The completion is recorded with its channels separated (`assistant_message`): in thinking mode a
closed reasoning channel becomes a reasoning block plus the answer text, so the viewer shows the
chain collapsed above the answer and scorers read the answer without re-splitting (`channels`).

Tool calls. A task that hands the model tools takes the harmony tool path: the tool schemas go
into the developer block, the conversation (assistant turns with their calls, tool results) is
rendered in harmony wire format, and the reply is parsed by the renderer into reasoning, text and
calls. Harmony carries no call ids, so the provider mints one per call; Inspect pairs the tool
result to it and the renderer needs only the function name. Reasoning from earlier assistant turns
is kept only after the last user message, which is the harmony rule for a tool loop in progress.
Only harmony (gpt-oss) renderers render tools; any other renderer is refused loudly.

Context window. A conversation whose prompt alone fills the model's context window is answered
with an empty turn whose stop reason is `model_length`, not an error: an agent loop ends there and
the scorer grades the state the run left, as it does at the message limit. A reply that would run
past the window is shortened by Tinker itself and arrives with stop reason `max_tokens`.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any

from inspect_ai.model import (
    ChatCompletionChoice,
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ContentReasoning,
    ContentText,
    GenerateConfig,
    ModelAPI,
    ModelOutput,
    ModelUsage,
    modelapi,
)
from inspect_ai.tool import ToolCall, ToolChoice, ToolInfo

from src.evals.base import (
    TINKER_DETERMINISTIC_STATUS,
    PromptExceedsContextWindow,
    TinkerGenerator,
    tinker_failure_is_transient,
)
from src.thinking import HARMONY_ANALYSIS_OPEN, renderer_uses_think_blocks, split

PROVIDER = "tinker"
REQUIRED_MODEL_ARGS = ("tokenizer_id", "renderer_name", "thinking", "max_connections")
# Tinker's stop reasons to Inspect's.
STOP_REASONS = {"stop": "stop", "length": "max_tokens"}
REQUIRED_SETTINGS = ("temperature", "top_p", "max_tokens")
# The function name Inspect sees for every call the renderer could not parse. It names no real tool,
# so a malformed call never runs and never counts as a submission (basic_agent counts any result
# under the submit tool's name as one); Inspect hands the parse error back as that call's result.
UNPARSED_FUNCTION = "unparsed"
# Assistant-message metadata key holding, per call id, each call the renderer could not parse as the
# model wrote it: {"function": the name its header gives, "arguments": its argument text, "raw_text":
# the renderer's raw text of the call}. The next prompt renders the call with that name and text.
UNPARSED_CALLS = "unparsed_tool_calls"
# Assistant-message metadata key holding the renderer's whole decoded turn when the turn held calls
# and that string was dropped from the content (see turn_message), so the log keeps it.
RAW_TURN = "raw_turn"
# Harmony markers that delimit a call in the model's raw text.
HARMONY_RECIPIENT = "to=functions."
HARMONY_MESSAGE = "<|message|>"
HARMONY_ENDS = ("<|end|>", "<|call|>", "<|return|>")
#: Metadata key on a system message naming the harmony role it renders as; its one allowed value is
#: DEVELOPER (a developer message holding the text verbatim). Without the key a system message renders
#: as the cookbook renders a system message.
HARMONY_ROLE = "harmony_role"
DEVELOPER = "developer"
#: How a task's framing prompt enters the conversation: not at all, as a system message, or as a
#: system message rendered as a harmony developer message.
FRAMING_ROLES = ("none", "system", DEVELOPER)


def framing_messages(framing_prompt: str | None, framing_role: str) -> list[ChatMessageSystem]:
    """The framing prompt as the messages that open a conversation. Role "none" goes with no prompt
    and gives no message; "system" and DEVELOPER go with a prompt and give one system message, the
    second carrying HARMONY_ROLE. Any other pairing is refused."""
    if framing_role not in FRAMING_ROLES:
        raise ValueError(f"framing role {framing_role!r}; expected one of {FRAMING_ROLES}")
    if (framing_prompt is None) != (framing_role == "none"):
        raise ValueError(f"framing role {framing_role!r} with {'no' if framing_prompt is None else 'a'} framing prompt")
    if framing_prompt is None:
        return []
    if not framing_prompt.strip():
        raise ValueError("framing prompt is empty; use role 'none' for no framing")
    metadata = {HARMONY_ROLE: DEVELOPER} if framing_role == DEVELOPER else None
    return [ChatMessageSystem(content=framing_prompt, metadata=metadata)]


def messages_of(input: list[ChatMessage]) -> list[dict]:
    """Inspect chat messages as the role/content dicts the renderer takes. A system message whose
    metadata sets HARMONY_ROLE takes that role; any value other than DEVELOPER is refused."""
    out = []
    for m in input:
        role = m.role
        if isinstance(m, ChatMessageSystem) and HARMONY_ROLE in (m.metadata or {}):
            if m.metadata[HARMONY_ROLE] != DEVELOPER:
                raise ValueError(f"{HARMONY_ROLE} {m.metadata[HARMONY_ROLE]!r}; the only role is {DEVELOPER!r}")
            role = DEVELOPER
        out.append({"role": role, "content": m.text})
    return out


def assistant_message(text: str, *, renderer_name: str, thinking: bool) -> ChatMessageAssistant:
    """The completion as an Inspect message. In thinking mode a CLOSED reasoning channel becomes a
    reasoning block followed by the answer text; `message.text` is then the answer alone. An
    unclosed channel, or a nothinking completion, stays plain text. So a reasoning block on the
    message means exactly that the channel closed."""
    if thinking:
        parts = split(text, renderer_name=renderer_name)
        if parts.closed:
            return ChatMessageAssistant(
                content=[ContentReasoning(reasoning=parts.reasoning or ""), ContentText(text=parts.answer)],
                source="generate",
            )
    return ChatMessageAssistant(content=text, source="generate")


@dataclass(frozen=True)
class Channels:
    reasoning: str | None
    answer: str
    closed: bool


def channels(message: ChatMessageAssistant) -> Channels:
    """Reasoning and answer as `assistant_message` recorded them. Scorers read this instead of
    re-splitting text, so the display and the measurement agree by construction."""
    if isinstance(message.content, str):
        return Channels(reasoning=None, answer=message.content.strip(), closed=False)
    reasoning = [c.reasoning for c in message.content if isinstance(c, ContentReasoning)]
    answer = "\n".join(c.text for c in message.content if isinstance(c, ContentText)).strip()
    return Channels(reasoning="\n".join(reasoning) if reasoning else None, answer=answer, closed=bool(reasoning))


def model_output(
    model: str, text: str, stop_reason: str, *, renderer_name: str, thinking: bool, usage: ModelUsage | None = None
) -> ModelOutput:
    """One completion as a ModelOutput, channels separated. Used by the provider and by replay
    providers, so replayed and live completions have the same shape."""
    return ModelOutput(
        model=model,
        choices=[
            ChatCompletionChoice(
                message=assistant_message(text, renderer_name=renderer_name, thinking=thinking),
                stop_reason=STOP_REASONS[stop_reason],
            )
        ],
        usage=usage,
    )


def context_full_output(model: str, ex: PromptExceedsContextWindow) -> ModelOutput:
    """The turn for a prompt that fills the context window: nothing sampled, stop reason
    `model_length`. No request was sent, so usage is zero; the prompt's length and the window are in
    the output's metadata."""
    return ModelOutput(
        model=model,
        choices=[
            ChatCompletionChoice(
                message=ChatMessageAssistant(content="", source="generate"), stop_reason="model_length"
            )
        ],
        usage=ModelUsage(input_tokens=0, output_tokens=0, total_tokens=0),
        metadata={"prompt_tokens": ex.prompt_tokens, "context_window": ex.context_window},
    )


def rendered_prompt(
    model: str,
    *,
    tokenizer_id: str,
    renderer_name: str,
    renderer_current_date: str | None,
    thinking: bool,
    messages: list[ChatMessage],
    reasoning_prefill: str | None,
) -> str:
    """The exact prompt text the provider would send for `messages`, special tokens included,
    rendered locally with no Tinker call. Dry runs print it so the framing and the prefill are
    seen where the model sees them. `model` is the Inspect model name, `tinker/<base model>`."""
    from src.evals.base import force_thinking_prompt, reasoning_prefill_prompt
    from src.training.tinker_common import apply_renderer_date, load_renderer, load_tokenizer

    prefix = f"{PROVIDER}/"
    if not model.startswith(prefix):
        raise ValueError(f"model {model!r} is not a {PROVIDER} model")
    tokenizer = load_tokenizer(tokenizer_id)
    renderer = load_renderer(model[len(prefix) :], renderer_name, tokenizer)
    apply_renderer_date(renderer, renderer_current_date)
    dicts = messages_of(messages)
    if any(m["role"] == DEVELOPER for m in dicts) and renderer_uses_think_blocks(renderer_name):
        raise ValueError(f"a {DEVELOPER} message needs a harmony renderer, not {renderer_name!r}")
    if reasoning_prefill is not None:
        prompt = reasoning_prefill_prompt(renderer, dicts, renderer_name, reasoning_prefill)
    elif thinking:
        prompt = force_thinking_prompt(renderer, tokenizer, dicts, renderer_name)
    else:
        raise ValueError("rendered_prompt shows thinking prompts only")
    return tokenizer.decode(prompt.to_ints(), skip_special_tokens=False)


# --- the tool path -------------------------------------------------------------------------------


def tool_spec(tool: ToolInfo) -> dict:
    """An Inspect tool as the name/description/parameters spec the renderer formats."""
    return {
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters.model_dump(exclude_none=True),
    }


def split_system(input: list[ChatMessage]) -> tuple[str, list[ChatMessage]]:
    """(system prompt, the rest). System messages lead the conversation and are joined into the
    one developer block; a system message after the first non-system turn is refused, since
    harmony has no slot for it. A system message carrying HARMONY_ROLE is refused: the tool path
    builds its own developer block and cannot honour it."""
    system: list[str] = []
    rest: list[ChatMessage] = []
    for m in input:
        if isinstance(m, ChatMessageSystem):
            if rest:
                raise ValueError("a system message after the conversation began has no harmony slot")
            if HARMONY_ROLE in (m.metadata or {}):
                raise ValueError(f"{HARMONY_ROLE} is not supported on the tool path")
            system.append(m.text)
        else:
            rest.append(m)
    return "\n\n".join(system), rest


def _cookbook_call(call: ToolCall, written: dict | None = None):
    """A call as the renderer renders it; for a call the renderer could not parse, `written` (its
    UNPARSED_CALLS entry) gives the name and argument text the model wrote, so the model sees them."""
    from tinker_cookbook.renderers import ToolCall as CookbookToolCall

    if written is None:
        name, arguments = call.function, json.dumps(call.arguments)
    else:
        name, arguments = written["function"], written["arguments"]
    return CookbookToolCall(function=CookbookToolCall.FunctionBody(name=name, arguments=arguments), id=call.id)


def history_of(rest: list[ChatMessage]) -> list[dict]:
    """The conversation after the system prompt as renderer messages. Assistant turns carry their
    text and calls; reasoning is kept only for turns after the last user message. Tool results
    carry the function name the renderer routes by (for an unparsed call, the name the model wrote);
    an errored call's result is its error text."""
    last_user = max((i for i, m in enumerate(rest) if m.role == "user"), default=-1)
    written_names: dict[str, str] = {}
    out: list[dict] = []
    for i, m in enumerate(rest):
        if isinstance(m, ChatMessageAssistant):
            parts: list[dict] = []
            if isinstance(m.content, str):
                if m.content:
                    parts.append({"type": "text", "text": m.content})
            else:
                for c in m.content:
                    if isinstance(c, ContentReasoning) and i > last_user:
                        parts.append({"type": "thinking", "thinking": c.reasoning})
                    elif isinstance(c, ContentText) and c.text:
                        parts.append({"type": "text", "text": c.text})
            msg: dict = {"role": "assistant", "content": parts}
            if m.tool_calls:
                unparsed = (m.metadata or {}).get(UNPARSED_CALLS, {})
                written_names.update({i: u["function"] for i, u in unparsed.items()})
                msg["tool_calls"] = [_cookbook_call(c, unparsed.get(c.id)) for c in m.tool_calls]
            out.append(msg)
        elif isinstance(m, ChatMessageTool):
            text = m.text
            if m.error is not None and not text:
                text = m.error.message
            name = written_names.get(m.tool_call_id, m.function)
            out.append({"role": "tool", "name": name, "content": text, "tool_call_id": m.tool_call_id})
        else:
            out.append({"role": m.role, "content": m.text})
    return out


def _call_id() -> str:
    return f"call_{uuid.uuid4().hex[:12]}"


def unparsed_call_parts(raw_text: str) -> tuple[str, str]:
    """(function name, argument text) of a call the renderer could not parse, read from the model's
    raw harmony text after its last `to=functions.`: the name follows that marker, the arguments
    follow the first `<|message|>` after it, up to an end marker. Text before the marker (reasoning,
    earlier calls) is never read. Text with no marker gives (UNPARSED_FUNCTION, ""); a header that
    names no function gives UNPARSED_FUNCTION; a header with no message gives empty arguments."""
    if HARMONY_RECIPIENT not in raw_text:
        return UNPARSED_FUNCTION, ""
    call = raw_text.rsplit(HARMONY_RECIPIENT, 1)[1]
    header = call.split("<|", 1)[0].split()
    name = header[0] if header else UNPARSED_FUNCTION
    arguments = ""
    if HARMONY_MESSAGE in call:
        arguments = call.split(HARMONY_MESSAGE, 1)[1]
        for end in HARMONY_ENDS:
            arguments = arguments.split(end, 1)[0]
    return name, arguments


def turn_message(parsed: dict) -> ChatMessageAssistant:
    """The renderer's parsed turn as an Inspect message: reasoning and text blocks in order, then
    the calls (each with a minted id). A call the renderer could not parse is named UNPARSED_FUNCTION
    and carries a `parse_error` that Inspect returns to the model as the call's result; the name and
    argument text the model wrote are kept in metadata (UNPARSED_CALLS), for the next prompt and the log.

    The renderer returns the whole decoded turn as one string when it finds no reasoning or text
    part. When the turn holds calls, that string is only the raw form of those calls, so it leaves
    the content (kept, it would be rendered back as text and the call would appear twice, its harmony
    markers re-encoded as control tokens) and is stored in metadata under RAW_TURN."""
    content: list[ContentReasoning | ContentText] = []
    metadata: dict = {}
    raw = parsed.get("content") or []
    has_calls = bool(parsed.get("tool_calls") or parsed.get("unparsed_tool_calls"))
    if isinstance(raw, str):
        if raw and has_calls:
            metadata[RAW_TURN] = raw
        elif raw:
            content.append(ContentText(text=raw))
    else:
        for part in raw:
            if part["type"] == "thinking":
                content.append(ContentReasoning(reasoning=part["thinking"]))
            elif part["type"] == "text":
                content.append(ContentText(text=part["text"]))
    calls = [
        ToolCall(id=_call_id(), function=c.function.name, arguments=json.loads(c.function.arguments), type="function")
        for c in parsed.get("tool_calls") or []
    ]
    unparsed: dict[str, dict] = {}
    for u in parsed.get("unparsed_tool_calls") or []:
        name, arguments = unparsed_call_parts(u.raw_text)
        call = ToolCall(id=_call_id(), function=UNPARSED_FUNCTION, arguments={}, parse_error=u.error, type="function")
        unparsed[call.id] = {"function": name, "arguments": arguments, "raw_text": u.raw_text}
        calls.append(call)
    if unparsed:
        metadata[UNPARSED_CALLS] = unparsed
    return ChatMessageAssistant(
        content=content or "",
        tool_calls=calls or None,
        source="generate",
        metadata=metadata or None,
    )


class TinkerModelAPI(ModelAPI):
    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_vars: list[str] = [],
        config: GenerateConfig = GenerateConfig(),
        **model_args: Any,
    ) -> None:
        super().__init__(model_name, base_url, api_key, api_key_vars, config)
        missing = [k for k in REQUIRED_MODEL_ARGS if k not in model_args]
        if missing:
            raise ValueError(f"{PROVIDER} provider needs model_args {missing}")
        self.thinking = bool(model_args["thinking"])
        self.renderer_name = str(model_args["renderer_name"])
        self._max_connections = int(model_args["max_connections"])
        self.reasoning_prefill = model_args.get("reasoning_prefill")
        if self.reasoning_prefill is not None:
            if renderer_uses_think_blocks(self.renderer_name) or not self.thinking:
                raise ValueError("reasoning_prefill needs a harmony renderer and thinking=true")
            if not self.reasoning_prefill:
                raise ValueError("reasoning_prefill is empty; leave it unset for no prefill")
        checkpoint = model_args.get("checkpoint")
        self.generator = TinkerGenerator(
            model_path=checkpoint or model_name,
            base_model=model_name if checkpoint else None,
            tokenizer_id=model_args["tokenizer_id"],
            renderer_name=self.renderer_name,
            max_concurrent=self._max_connections,
            renderer_current_date=model_args.get("renderer_current_date"),
        )
        self.generator.load()

    def max_connections(self) -> int:
        return self._max_connections

    def should_retry(self, ex: Exception) -> bool:
        import tinker

        if isinstance(ex, (tinker.RateLimitError, tinker.APIConnectionError, tinker.RequestFailedError)):
            return tinker_failure_is_transient(ex)
        if isinstance(ex, tinker.APIStatusError):
            return getattr(ex, "status_code", None) not in TINKER_DETERMINISTIC_STATUS
        return False

    async def generate(
        self, input: list[ChatMessage], tools: list[ToolInfo], tool_choice: ToolChoice, config: GenerateConfig
    ) -> ModelOutput:
        missing = [f for f in REQUIRED_SETTINGS if getattr(config, f) is None]
        if missing:
            raise ValueError(f"generation settings {missing} must be given explicitly; there are no defaults")
        try:
            if tools:
                return await self._generate_with_tools(input, tools, config)
            return await self._generate_plain(input, config)
        except PromptExceedsContextWindow as ex:
            return context_full_output(self.model_name, ex)

    async def _generate_plain(self, input: list[ChatMessage], config: GenerateConfig) -> ModelOutput:
        messages = messages_of(input)
        if any(m["role"] == DEVELOPER for m in messages) and renderer_uses_think_blocks(self.renderer_name):
            raise ValueError(f"a {DEVELOPER} message needs a harmony renderer, not {self.renderer_name!r}")
        sampled = await self.generator.sample_one(
            messages,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            seed=config.seed,
            thinking=self.thinking,
            reasoning_prefill=self.reasoning_prefill,
        )
        [seq] = sampled.sequences
        n_in, n_out = len(sampled.prompt_tokens), len(seq.tokens)
        # The sampled text starts mid-reasoning when the prompt held a prefill; the channel name and
        # the prefill are put back in front so the completion splits as one the model wrote whole.
        text = seq.text if self.reasoning_prefill is None else HARMONY_ANALYSIS_OPEN + self.reasoning_prefill + seq.text
        return model_output(
            self.model_name,
            text,
            seq.stop_reason,
            renderer_name=self.renderer_name,
            thinking=self.thinking,
            usage=ModelUsage(input_tokens=n_in, output_tokens=n_out, total_tokens=n_in + n_out),
        )

    async def _generate_with_tools(
        self, input: list[ChatMessage], tools: list[ToolInfo], config: GenerateConfig
    ) -> ModelOutput:
        if not self.thinking:
            raise ValueError(f"the {PROVIDER} tool path renders harmony reasoning; declare thinking=true")
        if self.reasoning_prefill is not None:
            raise ValueError("reasoning_prefill is not supported on the tool path")
        system, rest = split_system(input)
        sampled = await self.generator.sample_turn(
            history_of(rest),
            tools=[tool_spec(t) for t in tools],
            system_prompt=system,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            seed=config.seed,
        )
        [seq] = sampled.sequences
        parsed, _termination = self.generator.parse_turn(seq.tokens)
        message = turn_message(parsed)
        # A turn that stopped at a call is a tool-call stop; a turn cut at the cap is max_tokens
        # even if a call was parsed out of it; otherwise the provider's own stop reason.
        if seq.stop_reason == "length":
            stop = "max_tokens"
        elif message.tool_calls:
            stop = "tool_calls"
        else:
            stop = STOP_REASONS[seq.stop_reason]
        n_in, n_out = len(sampled.prompt_tokens), len(seq.tokens)
        return ModelOutput(
            model=self.model_name,
            choices=[ChatCompletionChoice(message=message, stop_reason=stop)],
            usage=ModelUsage(input_tokens=n_in, output_tokens=n_out, total_tokens=n_in + n_out),
        )


@modelapi(name=PROVIDER)
def tinker():
    return TinkerModelAPI
