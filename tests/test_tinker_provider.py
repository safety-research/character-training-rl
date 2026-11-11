"""The Inspect provider maps one TinkerGenerator sample to one ModelOutput, refuses missing
settings, and lets Inspect retry only transient Tinker errors. No Tinker calls: the generator is
replaced by a fake."""

from __future__ import annotations

import asyncio

import pytest
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, GenerateConfig

from src.evals import tinker_provider
from src.evals.base import SampledPrompt, SampledSequence

MODEL_ARGS = dict(tokenizer_id="tok", renderer_name="gpt_oss_low_reasoning", thinking=True, max_connections=7)


class FakeGenerator:
    stop_reason = "stop"

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []

    def load(self):
        pass

    async def sample_one(self, messages, **settings):
        self.calls.append((messages, settings))
        return SampledPrompt(
            prompt_tokens=[1, 2, 3],
            sequences=[
                SampledSequence(
                    tokens=[4, 5],
                    logprobs=[-0.1, -0.2],
                    text="analysis hi assistantfinal ok",
                    stop_reason=self.stop_reason,
                )
            ],
        )


@pytest.fixture
def provider(monkeypatch):
    monkeypatch.setattr(tinker_provider, "TinkerGenerator", FakeGenerator)
    return tinker_provider.TinkerModelAPI("openai/gpt-oss-120b", **MODEL_ARGS)


def test_base_model_and_checkpoint_routing(monkeypatch):
    monkeypatch.setattr(tinker_provider, "TinkerGenerator", FakeGenerator)
    base = tinker_provider.TinkerModelAPI("openai/gpt-oss-120b", **MODEL_ARGS)
    assert base.generator.kwargs["model_path"] == "openai/gpt-oss-120b" and base.generator.kwargs["base_model"] is None
    ckpt = tinker_provider.TinkerModelAPI(
        "openai/gpt-oss-120b", checkpoint="tinker://run:0/sampler_weights/x", **MODEL_ARGS
    )
    assert ckpt.generator.kwargs["model_path"] == "tinker://run:0/sampler_weights/x"
    assert ckpt.generator.kwargs["base_model"] == "openai/gpt-oss-120b"
    assert base.max_connections() == 7


def test_missing_model_args_is_an_error(monkeypatch):
    monkeypatch.setattr(tinker_provider, "TinkerGenerator", FakeGenerator)
    with pytest.raises(ValueError, match="thinking"):
        tinker_provider.TinkerModelAPI("openai/gpt-oss-120b", tokenizer_id="tok", renderer_name="r", max_connections=1)


def test_generate_maps_one_sample(provider):
    config = GenerateConfig(temperature=1.0, top_p=0.95, max_tokens=64)
    out = asyncio.run(
        provider.generate([ChatMessageSystem(content="sys"), ChatMessageUser(content="q")], [], None, config)
    )
    # thinking mode, closed channel: a reasoning block plus the answer; `completion` is the answer
    assert out.completion == "ok"
    assert tinker_provider.channels(out.message) == tinker_provider.Channels(reasoning="hi", answer="ok", closed=True)
    assert out.choices[0].stop_reason == "stop"
    assert (out.usage.input_tokens, out.usage.output_tokens) == (3, 2)
    messages, settings = provider.generator.calls[0]
    assert messages == [{"role": "system", "content": "sys"}, {"role": "user", "content": "q"}]
    assert settings == dict(
        max_tokens=64, temperature=1.0, top_p=0.95, seed=None, thinking=True, reasoning_prefill=None
    )


def test_length_stop_reason_is_max_tokens(provider):
    provider.generator.stop_reason = "length"
    config = GenerateConfig(temperature=1.0, top_p=1.0, max_tokens=8)
    out = asyncio.run(provider.generate([ChatMessageUser(content="q")], [], None, config))
    assert out.choices[0].stop_reason == "max_tokens"


def test_missing_settings_are_an_error(provider):
    with pytest.raises(ValueError, match="top_p"):
        asyncio.run(
            provider.generate([ChatMessageUser(content="q")], [], None, GenerateConfig(temperature=1.0, max_tokens=8))
        )


def test_should_retry_only_transient(provider):
    tinker = pytest.importorskip("tinker")

    class Status(tinker.APIStatusError):
        def __init__(self, code):
            self.status_code = code

        def __str__(self):
            return f"status {self.status_code}"

    assert provider.should_retry(Status(500)) is True
    assert provider.should_retry(Status(402)) is False
    assert provider.should_retry(ValueError("x")) is False
    cat = tinker.types.RequestErrorCategory
    failed = lambda c: tinker.RequestFailedError("Failed to sample after exhausting retries", "r", c)  # noqa: E731
    assert provider.should_retry(failed(cat.Unknown)) is True
    assert provider.should_retry(failed(cat.Server)) is True
    assert provider.should_retry(failed(cat.User)) is False


def test_assistant_message_separates_channels_only_when_closed():
    closed = tinker_provider.assistant_message(
        "analysis think assistantfinal answer", renderer_name="gpt_oss_low_reasoning", thinking=True
    )
    assert tinker_provider.channels(closed) == tinker_provider.Channels("think", "answer", True)
    unclosed = tinker_provider.assistant_message(
        "analysis never closes", renderer_name="gpt_oss_low_reasoning", thinking=True
    )
    assert tinker_provider.channels(unclosed) == tinker_provider.Channels(None, "analysis never closes", False)
    nothinking = tinker_provider.assistant_message(
        "plain answer", renderer_name="gpt_oss_low_reasoning", thinking=False
    )
    assert tinker_provider.channels(nothinking) == tinker_provider.Channels(None, "plain answer", False)
    qwen = tinker_provider.assistant_message("<think>\nhmm\n</think>\nyes", renderer_name="qwen3", thinking=True)
    assert tinker_provider.channels(qwen) == tinker_provider.Channels("hmm", "yes", True)


# --- the tool path -------------------------------------------------------------------------------


class FakeToolGenerator(FakeGenerator):
    """Records the tool turn it was asked for and returns a parsed turn of the caller's choosing."""

    parsed = {"role": "assistant", "content": [{"type": "thinking", "thinking": "look"}], "tool_calls": []}
    stop_reason = "stop"

    async def sample_turn(self, messages, **settings):
        self.calls.append((messages, settings))
        return SampledPrompt(
            prompt_tokens=[1] * 10,
            sequences=[SampledSequence(tokens=[4, 5, 6], logprobs=[0.0] * 3, text="", stop_reason=self.stop_reason)],
        )

    def parse_turn(self, tokens):
        return self.parsed, "stop_sequence"


def _tool():
    from inspect_ai.tool import ToolInfo, ToolParam, ToolParams

    return ToolInfo(
        name="shell",
        description="Run a command.",
        parameters=ToolParams(
            properties={"cmd": ToolParam(type="string", description="the command")}, required=["cmd"]
        ),
    )


def _cookbook_call(name, arguments):
    from tinker_cookbook.renderers import ToolCall as CookbookToolCall

    return CookbookToolCall(function=CookbookToolCall.FunctionBody(name=name, arguments=arguments), id=None)


def test_tool_turn_is_rendered_and_parsed(monkeypatch):
    from inspect_ai.model import ChatMessageAssistant, ChatMessageTool, ContentReasoning, ContentText
    from inspect_ai.tool import ToolCall

    monkeypatch.setattr(tinker_provider, "TinkerGenerator", FakeToolGenerator)
    provider = tinker_provider.TinkerModelAPI("openai/gpt-oss-20b", **MODEL_ARGS)
    provider.generator.parsed = {
        "role": "assistant",
        "content": [{"type": "thinking", "thinking": "read it"}],
        "tool_calls": [_cookbook_call("shell", '{"cmd": "cat a"}')],
    }
    history = [
        ChatMessageSystem(content="You are an agent."),
        ChatMessageUser(content="List /tmp"),
        ChatMessageAssistant(
            content=[ContentReasoning(reasoning="I should list."), ContentText(text="")],
            tool_calls=[ToolCall(id="call_1", function="shell", arguments={"cmd": "ls /tmp"}, type="function")],
        ),
        ChatMessageTool(content="a\nb", tool_call_id="call_1", function="shell"),
    ]
    config = GenerateConfig(temperature=1.0, top_p=1.0, max_tokens=64)
    out = asyncio.run(provider.generate(history, [_tool()], None, config))
    # the reply: reasoning block, one call with a minted id and parsed arguments, tool_calls stop
    [call] = out.message.tool_calls
    assert (call.function, call.arguments) == ("shell", {"cmd": "cat a"}) and call.id.startswith("call_")
    assert tinker_provider.channels(out.message).reasoning == "read it"
    assert out.choices[0].stop_reason == "tool_calls"
    assert (out.usage.input_tokens, out.usage.output_tokens) == (10, 3)
    # the request: system prompt split off, tool spec from the ToolInfo, history in renderer shape
    messages, settings = provider.generator.calls[0]
    assert settings["system_prompt"] == "You are an agent."
    assert settings["tools"] == [
        {
            "name": "shell",
            "description": "Run a command.",
            "parameters": {
                "type": "object",
                "properties": {"cmd": {"type": "string", "description": "the command"}},
                "required": ["cmd"],
                "additionalProperties": False,
            },
        }
    ]
    assert messages[0] == {"role": "user", "content": "List /tmp"}
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == [{"type": "thinking", "thinking": "I should list."}]  # after the last user turn
    assert messages[1]["tool_calls"][0].function.name == "shell"
    assert messages[1]["tool_calls"][0].function.arguments == '{"cmd": "ls /tmp"}'
    assert messages[2] == {"role": "tool", "name": "shell", "content": "a\nb", "tool_call_id": "call_1"}


def test_tool_turn_final_answer_and_cap(monkeypatch):
    monkeypatch.setattr(tinker_provider, "TinkerGenerator", FakeToolGenerator)
    provider = tinker_provider.TinkerModelAPI("openai/gpt-oss-20b", **MODEL_ARGS)
    provider.generator.parsed = {
        "role": "assistant",
        "content": [{"type": "thinking", "thinking": "done"}, {"type": "text", "text": "All good."}],
    }
    config = GenerateConfig(temperature=1.0, top_p=1.0, max_tokens=64)
    out = asyncio.run(provider.generate([ChatMessageUser(content="q")], [_tool()], None, config))
    assert out.message.tool_calls is None and out.completion == "All good."
    assert out.choices[0].stop_reason == "stop"
    provider.generator.stop_reason = "length"
    out = asyncio.run(provider.generate([ChatMessageUser(content="q")], [_tool()], None, config))
    assert out.choices[0].stop_reason == "max_tokens"


def test_unparsed_call_becomes_a_parse_error(monkeypatch):
    from inspect_ai.model import ChatMessageTool
    from inspect_ai.tool import ToolCallError
    from tinker_cookbook.renderers import UnparsedToolCall

    monkeypatch.setattr(tinker_provider, "TinkerGenerator", FakeToolGenerator)
    provider = tinker_provider.TinkerModelAPI("openai/gpt-oss-20b", **MODEL_ARGS)
    provider.generator.parsed = {
        "role": "assistant",
        "content": [],
        "unparsed_tool_calls": [
            UnparsedToolCall(
                raw_text='<|start|>assistant to=functions.shell<|channel|>commentary <|constrain|>json<|message|>{"cmd": "ls" "-a"}<|call|>',
                error="Invalid JSON",
            )
        ],
    }
    config = GenerateConfig(temperature=1.0, top_p=1.0, max_tokens=64)
    out = asyncio.run(provider.generate([ChatMessageUser(content="q")], [_tool()], None, config))
    [call] = out.message.tool_calls
    assert (call.function, call.arguments, call.parse_error) == (tinker_provider.UNPARSED_FUNCTION, {}, "Invalid JSON")
    raw = provider.generator.parsed["unparsed_tool_calls"][0].raw_text
    assert out.message.metadata == {
        tinker_provider.UNPARSED_CALLS: {
            call.id: {"function": "shell", "arguments": '{"cmd": "ls" "-a"}', "raw_text": raw}
        }
    }
    assert out.choices[0].stop_reason == "tool_calls"
    # The next prompt renders the call, and its result, with the name and argument text the model wrote.
    result = ChatMessageTool(
        content="", tool_call_id=call.id, function=call.function, error=ToolCallError("parsing", "Invalid JSON")
    )
    rendered_turn, rendered_result = tinker_provider.history_of([out.message, result])
    [rendered] = rendered_turn["tool_calls"]
    assert (rendered.function.name, rendered.function.arguments) == ("shell", '{"cmd": "ls" "-a"}')
    assert (rendered_result["name"], rendered_result["content"]) == ("shell", "Invalid JSON")


def test_a_malformed_submit_is_not_a_submission():
    """basic_agent counts any tool result under the submit tool's name as a submission; a submit call
    the renderer could not parse must not produce one."""
    from inspect_ai.model import execute_tools
    from inspect_ai.tool import tool
    from tinker_cookbook.renderers import UnparsedToolCall

    @tool
    def submit():
        async def execute(answer: str):
            """Submit.

            Args:
              answer (str): the answer
            """
            return answer

        return execute

    message = tinker_provider.turn_message(
        {
            "role": "assistant",
            "content": [],
            "unparsed_tool_calls": [
                UnparsedToolCall(
                    raw_text='to=functions.submit<|channel|>commentary <|constrain|>json<|message|>{"answer": "done"',
                    error="Invalid JSON",
                )
            ],
        }
    )
    result = asyncio.run(execute_tools([message], [submit()]))
    [reply] = result.messages
    assert reply.function != "submit" and reply.error is not None


def test_unparsed_call_parts():
    parts = tinker_provider.unparsed_call_parts
    assert parts('to=functions.bash<|channel|>commentary <|constrain|>json<|message|>{"a": 1,}<|call|>') == (
        "bash",
        '{"a": 1,}',
    )
    # a header with no message, and text with no header at all
    assert parts("<|start|>assistant to=functions.python<|channel|>commentary") == ("python", "")
    assert parts("<|channel|>analysis<|message|>{x}") == (tinker_provider.UNPARSED_FUNCTION, "")
    # reasoning before a dangling header is never read as its arguments
    dangling = "<|channel|>analysis<|message|>I will list files<|end|> to=functions.bash<|channel|>commentary"
    assert parts(dangling) == ("bash", "")
    # only the last call header of the text is read
    two = 'to=functions.bash<|message|>{"cmd": "ls"}<|call|> to=functions.python<|message|>{"code": 1,'
    assert parts(two) == ("python", '{"code": 1,')


def test_raw_turn_string_is_dropped_when_the_turn_holds_calls():
    raw = '<|channel|>analysis to=functions.shell<|message|>{"cmd": "ls"}'
    with_call = {"role": "assistant", "content": raw, "tool_calls": [_cookbook_call("shell", '{"cmd": "ls"}')]}
    message = tinker_provider.turn_message(with_call)
    assert message.content == "" and [c.function for c in message.tool_calls] == ["shell"]
    assert message.metadata == {tinker_provider.RAW_TURN: raw}  # kept in the log, not in the prompt
    # a turn with no call keeps its text
    assert tinker_provider.turn_message({"role": "assistant", "content": "plain words"}).text == "plain words"


def test_tool_path_refusals(monkeypatch):
    monkeypatch.setattr(tinker_provider, "TinkerGenerator", FakeToolGenerator)
    config = GenerateConfig(temperature=1.0, top_p=1.0, max_tokens=64)
    nothink = tinker_provider.TinkerModelAPI("openai/gpt-oss-20b", **{**MODEL_ARGS, "thinking": False})
    with pytest.raises(ValueError, match="thinking=true"):
        asyncio.run(nothink.generate([ChatMessageUser(content="q")], [_tool()], None, config))
    with pytest.raises(ValueError, match="system message after"):
        tinker_provider.split_system([ChatMessageUser(content="q"), ChatMessageSystem(content="late")])


def test_history_drops_reasoning_before_the_last_user_turn():
    from inspect_ai.model import ChatMessageAssistant, ContentReasoning, ContentText

    rest = [
        ChatMessageUser(content="first"),
        ChatMessageAssistant(content=[ContentReasoning(reasoning="old"), ContentText(text="a")]),
        ChatMessageUser(content="second"),
        ChatMessageAssistant(content=[ContentReasoning(reasoning="new"), ContentText(text="b")]),
    ]
    out = tinker_provider.history_of(rest)
    assert out[1]["content"] == [{"type": "text", "text": "a"}]
    assert out[3]["content"] == [{"type": "thinking", "thinking": "new"}, {"type": "text", "text": "b"}]


class FullWindowGenerator(FakeToolGenerator):
    """A generator whose every prompt fills the context window."""

    async def sample_one(self, messages, **settings):
        from src.evals.base import PromptExceedsContextWindow

        raise PromptExceedsContextWindow(32_789, 32_768)

    async def sample_turn(self, messages, **settings):
        return await self.sample_one(messages, **settings)


@pytest.mark.parametrize("with_tools", [False, True])
def test_full_context_window_is_model_length_not_an_error(monkeypatch, with_tools):
    monkeypatch.setattr(tinker_provider, "TinkerGenerator", FullWindowGenerator)
    api = tinker_provider.TinkerModelAPI("openai/gpt-oss-20b", **MODEL_ARGS)
    out = asyncio.run(
        api.generate(
            [ChatMessageUser(content="hi")],
            [_tool()] if with_tools else [],
            "auto",
            GenerateConfig(temperature=1.0, top_p=1.0, max_tokens=8192),
        )
    )
    assert out.choices[0].stop_reason == "model_length"
    assert out.choices[0].message.text == "" and not out.choices[0].message.tool_calls
    assert (out.usage.input_tokens, out.usage.output_tokens) == (0, 0)  # nothing was sent
    assert out.metadata == {"prompt_tokens": 32_789, "context_window": 32_768}


def test_generator_refuses_a_prompt_that_fills_the_window():
    import tinker

    from src.evals.base import PromptExceedsContextWindow, TinkerGenerator

    class Caps:
        supported_models = [type("M", (), {"model_name": "openai/gpt-oss-20b", "max_context_length": 4})()]

    class Service:
        async def get_server_capabilities_async(self):
            return Caps()

    gen = TinkerGenerator(model_path="openai/gpt-oss-20b", tokenizer_id="tok", renderer_name="gpt_oss_low_reasoning")
    gen._service = Service()
    asyncio.run(gen._check_fits(tinker.ModelInput.from_ints([1, 2, 3])))
    with pytest.raises(PromptExceedsContextWindow) as ex:
        asyncio.run(gen._check_fits(tinker.ModelInput.from_ints([1, 2, 3, 4])))
    assert (ex.value.prompt_tokens, ex.value.context_window) == (4, 4)
    gen.base_model = "unknown/model"
    gen._context_window = None
    with pytest.raises(RuntimeError, match="not among"):
        asyncio.run(gen._check_fits(tinker.ModelInput.from_ints([1])))


def test_concurrent_first_turns_read_the_window_once_and_a_missing_length_is_an_error():
    import tinker

    from src.evals.base import TinkerGenerator

    class Service:
        def __init__(self, length):
            self.length, self.requests = length, 0

        async def get_server_capabilities_async(self):
            self.requests += 1
            await asyncio.sleep(0.01)
            model = type("M", (), {"model_name": "openai/gpt-oss-20b", "max_context_length": self.length})()
            return type("Caps", (), {"supported_models": [model]})()

    async def many(gen):
        await asyncio.gather(*(gen._check_fits(tinker.ModelInput.from_ints([1])) for _ in range(50)))

    gen = TinkerGenerator(model_path="openai/gpt-oss-20b", tokenizer_id="tok", renderer_name="gpt_oss_low_reasoning")
    gen._service = Service(32_768)
    asyncio.run(many(gen))
    assert gen._service.requests == 1 and gen._context_window == 32_768
    gen = TinkerGenerator(model_path="openai/gpt-oss-20b", tokenizer_id="tok", renderer_name="gpt_oss_low_reasoning")
    gen._service = Service(None)
    with pytest.raises(RuntimeError, match="declares no context length"):
        asyncio.run(many(gen))


def test_framing_messages_roles_and_refusals():
    assert tinker_provider.framing_messages(None, "none") == []
    [system] = tinker_provider.framing_messages("rules", "system")
    assert system.text == "rules" and not system.metadata
    [developer] = tinker_provider.framing_messages("rules", "developer")
    assert developer.text == "rules" and developer.metadata == {tinker_provider.HARMONY_ROLE: "developer"}
    with pytest.raises(ValueError, match="no framing prompt"):
        tinker_provider.framing_messages(None, "developer")
    with pytest.raises(ValueError, match="a framing prompt"):
        tinker_provider.framing_messages("rules", "none")
    with pytest.raises(ValueError, match="expected one of"):
        tinker_provider.framing_messages("rules", "user")
    with pytest.raises(ValueError, match="empty"):
        tinker_provider.framing_messages("  \n", "developer")
    with pytest.raises(ValueError, match="tool path"):
        tinker_provider.split_system(tinker_provider.framing_messages("rules", "developer"))


def test_developer_framing_reaches_the_renderer_as_developer(provider):
    config = GenerateConfig(temperature=1.0, top_p=1.0, max_tokens=64)
    framing = tinker_provider.framing_messages("rules", "developer")
    asyncio.run(provider.generate([*framing, ChatMessageUser(content="q")], [], None, config))
    messages, _ = provider.generator.calls[0]
    assert messages == [{"role": "developer", "content": "rules"}, {"role": "user", "content": "q"}]
    bad = ChatMessageSystem(content="rules", metadata={tinker_provider.HARMONY_ROLE: "user"})
    with pytest.raises(ValueError, match="the only role"):
        tinker_provider.messages_of([bad])


def test_developer_framing_needs_a_harmony_renderer(monkeypatch):
    monkeypatch.setattr(tinker_provider, "TinkerGenerator", FakeGenerator)
    qwen = tinker_provider.TinkerModelAPI("Qwen/Qwen3-8B", **{**MODEL_ARGS, "renderer_name": "qwen3"})
    framing = tinker_provider.framing_messages("rules", "developer")
    config = GenerateConfig(temperature=1.0, top_p=1.0, max_tokens=64)
    with pytest.raises(ValueError, match="harmony renderer"):
        asyncio.run(qwen.generate([*framing, ChatMessageUser(content="q")], [], None, config))


class PrefillGenerator(FakeGenerator):
    """Returns only what the model sampled after a prefill: the rest of the reasoning, then the answer."""

    async def sample_one(self, messages, **settings):
        self.calls.append((messages, settings))
        return SampledPrompt(
            prompt_tokens=[1, 2, 3, 4],
            sequences=[
                SampledSequence(tokens=[5], logprobs=[-0.1], text=" It is fine.assistantfinalok", stop_reason="stop")
            ],
        )


def test_reasoning_prefill_is_sent_and_recorded_in_front_of_the_reasoning(monkeypatch):
    monkeypatch.setattr(tinker_provider, "TinkerGenerator", PrefillGenerator)
    api = tinker_provider.TinkerModelAPI("openai/gpt-oss-20b", reasoning_prefill="Let me check.", **MODEL_ARGS)
    config = GenerateConfig(temperature=1.0, top_p=1.0, max_tokens=64)
    out = asyncio.run(api.generate([ChatMessageUser(content="q")], [], None, config))
    assert api.generator.calls[0][1]["reasoning_prefill"] == "Let me check."
    assert tinker_provider.channels(out.message) == tinker_provider.Channels(
        reasoning="Let me check. It is fine.", answer="ok", closed=True
    )
    assert (out.usage.input_tokens, out.usage.output_tokens) == (4, 1)


def test_reasoning_prefill_refusals(monkeypatch):
    monkeypatch.setattr(tinker_provider, "TinkerGenerator", FakeToolGenerator)
    with pytest.raises(ValueError, match="harmony renderer and thinking"):
        tinker_provider.TinkerModelAPI("openai/gpt-oss-20b", reasoning_prefill="x", **{**MODEL_ARGS, "thinking": False})
    with pytest.raises(ValueError, match="harmony renderer and thinking"):
        tinker_provider.TinkerModelAPI(
            "Qwen/Qwen3-8B", reasoning_prefill="x", **{**MODEL_ARGS, "renderer_name": "qwen3"}
        )
    with pytest.raises(ValueError, match="empty"):
        tinker_provider.TinkerModelAPI("openai/gpt-oss-20b", reasoning_prefill="", **MODEL_ARGS)
    api = tinker_provider.TinkerModelAPI("openai/gpt-oss-20b", reasoning_prefill="x", **MODEL_ARGS)
    config = GenerateConfig(temperature=1.0, top_p=1.0, max_tokens=64)
    with pytest.raises(ValueError, match="tool path"):
        asyncio.run(api.generate([ChatMessageUser(content="q")], [_tool()], None, config))


def test_reasoning_prefill_prompt_opens_the_analysis_channel():
    from src.evals.base import HARMONY_ANALYSIS_HEADER, reasoning_prefill_prompt

    class Renderer:
        def build_generation_prompt(self, messages, prefill=None):
            return prefill

    assert reasoning_prefill_prompt(Renderer(), [], "gpt_oss_medium_reasoning", "Let me check.") == (
        HARMONY_ANALYSIS_HEADER + "Let me check."
    )
    with pytest.raises(ValueError, match="think blocks"):
        reasoning_prefill_prompt(Renderer(), [], "qwen3", "Let me check.")
