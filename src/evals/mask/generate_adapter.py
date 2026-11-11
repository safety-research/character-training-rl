"""Runs MASK's own generate_responses.py with the Tinker model in place of the OpenAI client.

generate_responses.py builds its client with `AsyncOpenAI(...)` and calls
`client.chat.completions.create(model=, messages=, max_tokens=, temperature=)`, reading
`completion.choices[0].message.content`. Before the script runs, `openai.AsyncOpenAI` is replaced by
`TinkerClient`, which answers that call with the Inspect Tinker model: the messages go through the
model's renderer, and the content returned is the answer channel. It is None when the model produced
no answer or its reasoning channel never closed (cut off at max_tokens), which the script records as
an error, as it does for a failed API call. Every completion,
reasoning channel included, is appended to completions.jsonl the moment it returns.

    python -m src.evals.mask.generate_adapter <config.json> <script.py> [script args...]

run with the work directory (holding csv_data/) as the current directory.
"""

from __future__ import annotations

import json
import runpy
import sys
import time
from dataclasses import dataclass
from pathlib import Path

#: Where each completion is written as it arrives, relative to the work directory.
COMPLETIONS_FILE = "completions.jsonl"
#: Running token totals for the measured cost, rewritten after every completion.
USAGE_FILE = "generation_usage.json"


def to_inspect_messages(messages: list[dict]):
    """generate_responses.py's message dicts as Inspect chat messages (system, user, assistant)."""
    from inspect_ai.model import ChatMessageAssistant, ChatMessageSystem, ChatMessageUser

    kinds = {"system": ChatMessageSystem, "user": ChatMessageUser, "assistant": ChatMessageAssistant}
    out = []
    for m in messages:
        if m["role"] not in kinds:
            raise ValueError(f"unexpected role {m['role']!r} from generate_responses.py")
        out.append(kinds[m["role"]](content=m["content"]))
    return out


@dataclass
class _Message:
    content: str | None


@dataclass
class _Choice:
    message: _Message


@dataclass
class _Completion:
    choices: list[_Choice]


class _Completions:
    def __init__(self, client: "TinkerClient"):
        self._client = client

    async def create(self, *, model: str, messages: list[dict], max_tokens: int, temperature: float):
        return await self._client.complete(model, messages, max_tokens, temperature)


class _Chat:
    def __init__(self, client: "TinkerClient"):
        self.completions = _Completions(client)


class TinkerClient:
    """The slice of AsyncOpenAI that generate_responses.py uses, served by the Tinker model."""

    config: dict = {}

    def __init__(self, api_key=None, **_):
        from inspect_ai.model import GenerateConfig

        from src.evals.common import build_generator

        c = self.config
        args = type("Args", (), c["model_args"])()
        self._model = build_generator(args, c["max_connections"])
        self._gen = GenerateConfig(top_p=c["top_p"])
        self._expected = {"max_tokens": c["max_tokens"], "temperature": c["temperature"]}
        self._out = Path(COMPLETIONS_FILE)
        # The script builds one client per archetype file, and a resumed run starts a new process:
        # totals carry over from the file so the measured cost covers every completion.
        usage_file = Path(USAGE_FILE)
        self._usage = (
            json.loads(usage_file.read_text())
            if usage_file.exists()
            else {"input_tokens": 0, "output_tokens": 0, "completions": 0}
        )
        self.chat = _Chat(self)

    async def complete(self, model: str, messages: list[dict], max_tokens: int, temperature: float):
        from inspect_ai.model import GenerateConfig

        from src.evals.tinker_provider import channels

        if {"max_tokens": max_tokens, "temperature": temperature} != self._expected:
            raise RuntimeError(
                f"generate_responses.py asked for {max_tokens=}, {temperature=}; run declared {self._expected}"
            )
        cfg = self._gen.merge(GenerateConfig(max_tokens=max_tokens, temperature=temperature))
        output = await self._model.generate(to_inspect_messages(messages), config=cfg)
        ch = channels(output.message)
        usage = output.usage
        with self._out.open("a") as f:
            f.write(
                json.dumps(
                    {
                        "time": time.time(),
                        "messages": messages,
                        "reasoning": ch.reasoning,
                        "answer": ch.answer,
                        "reasoning_closed": ch.closed,
                        "stop_reason": output.choices[0].stop_reason,
                        "input_tokens": usage.input_tokens if usage else None,
                        "output_tokens": usage.output_tokens if usage else None,
                    }
                )
                + "\n"
            )
        if usage:
            self._usage["input_tokens"] += usage.input_tokens
            self._usage["output_tokens"] += usage.output_tokens
        self._usage["completions"] += 1
        Path(USAGE_FILE).write_text(json.dumps(self._usage))
        # Thinking mode: an unclosed reasoning channel (cut off at max_tokens) comes back as plain text,
        # so `answer` would be the reasoning; an API server returns no content there, and so do we.
        usable = ch.closed or not self.config["model_args"]["thinking"]
        return _Completion([_Choice(_Message(ch.answer if usable and ch.answer else None))])


def main(argv: list[str]) -> None:
    import openai

    config_path, script, *script_args = argv
    TinkerClient.config = json.loads(Path(config_path).read_text())
    openai.AsyncOpenAI = TinkerClient
    sys.argv = [script, *script_args]
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main(sys.argv[1:])
