"""MASK's judging with a Claude judge (forced tool call, temperature 0, through run_judge) in place of
gpt-4o and o3-mini, with MASK's evaluate.py and its judge prompts unchanged.

evaluate.py asks its judge one prompt at a time through an OpenAI client; run_judge takes a list. So:
  1. record: evaluate.py runs on a copy of the responses with a stand-in client that records every judge
     prompt (gpt-4o letter prompts and o3-mini numerical-range prompts) and answers with a placeholder;
  2. judge: the recorded prompts go to run_judge with a forced tool whose fields mirror what each prompt
     asks for -- a brief reasoning and an answer letter A-D for the letter prompts, MASK's NumericalRange
     fields for the numerical ones; every verdict is appended to VERDICTS_FILE as it lands (a rerun
     resumes from it and asks only the prompts that have no verdict yet, judge errors included);
  3. replay: evaluate.py runs in the work directory with a stand-in client that returns each stored
     verdict ("<reasoning>\\n\\nAnswer: <letter>", or the range). A judge error becomes the "ERROR: ..." text
     evaluate.py records for a failed API call, so MASK's metric and our failure count read it as one.
Identical prompts (the same response judged twice) are judged once and get the same verdict.
"""

from __future__ import annotations

import hashlib
import json
import os
import runpy
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

#: Where Claude's verdicts are written as they arrive, relative to the work directory.
VERDICTS_FILE = "claude_verdicts.jsonl"
#: The record pass's scratch copy of the work directory.
RECORD_DIR = "claude_record"
#: "Briefly reason, then answer" fits easily; a truncated reply is a judge error.
JUDGE_MAX_TOKENS = 1500
LETTERS = ["A", "B", "C", "D"]
NUMERIC_FIELDS = ("lower_bound", "upper_bound", "relative_lower_bound", "relative_upper_bound")
LETTER_TOOL = {
    "name": "submit_judgment",
    "description": "Submit your brief reasoning and your final answer letter.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {"reasoning": {"type": "string"}, "answer": {"type": "string", "enum": LETTERS}},
        "required": ["reasoning", "answer"],
        "additionalProperties": False,
    },
}
NUMERIC_TOOL = {
    "name": "submit_range",
    "description": "Submit the numerical range: each bound a number or null.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {f: {"type": ["number", "null"]} for f in NUMERIC_FIELDS},
        "required": list(NUMERIC_FIELDS),
        "additionalProperties": False,
    },
}
#: What the replay returns for a prompt Claude gave no verdict on (evaluate.py's failed-call text).
ERROR_TEXT = "ERROR: the Claude judge returned no verdict"


def prompt_key(system: str, prompt: str) -> str:
    return hashlib.sha256(f"{system}\x00{prompt}".encode()).hexdigest()


def tool_extractor(name: str, fields: tuple[str, ...]):
    def extract(message):
        blocks = [b for b in message.content if getattr(b, "type", None) == "tool_use" and b.name == name]
        if len(blocks) != 1 or set(blocks[0].input) != set(fields):
            return None
        return dict(blocks[0].input)

    return extract


class StandInClient:
    """The slice of openai.AsyncOpenAI evaluate.py uses. `verdicts` None records; otherwise replays."""

    def __init__(self, recorded: dict, verdicts: dict | None):
        self.recorded, self.verdicts = recorded, verdicts
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.beta = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=self._parse)))

    def _note(self, kind: str, messages: list[dict]) -> str:
        system, prompt = messages[0]["content"], messages[1]["content"]
        k = prompt_key(system, prompt)
        self.recorded[k] = {"kind": kind, "system": system, "prompt": prompt}
        return k

    async def _create(self, *, messages, **_):
        k = self._note("letter", messages)
        if self.verdicts is None:
            content = "Answer: A"
        else:
            v = self.verdicts.get(k)
            # the answer line alone: evaluate.py takes the FIRST "Answer:" in the text, and a reasoning that
            # mentions one would be parsed as the verdict. The reasoning stays in VERDICTS_FILE.
            content = ERROR_TEXT if v is None else f"Answer: {v['answer']}"
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    async def _parse(self, *, messages, **_):
        k = self._note("numeric", messages)
        v = {f: None for f in NUMERIC_FIELDS} if self.verdicts is None else self.verdicts.get(k)
        if v is None:
            raise RuntimeError(ERROR_TEXT)  # evaluate.py retries, then records its ERROR text
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=SimpleNamespace(**v)))])


def run_evaluate(workdir: Path, responses: list[str], client: StandInClient, concurrency: int) -> None:
    """MASK's evaluate.py, in the work directory, once per responses file, with `client` as its OpenAI client."""
    import asyncio

    import dotenv
    import openai

    async def no_wait(*_a, **_k):
        return None

    saved = (openai.AsyncOpenAI, asyncio.sleep, dotenv.load_dotenv, Path.cwd(), list(sys.path), list(sys.argv))
    openai.AsyncOpenAI = lambda *a, **k: client
    # The stand-in answers instantly, so evaluate.py's retry backoff (2+4+...+32 s) only delays a missing
    # verdict past its row's 60 s timeout, which would turn the row's other verdicts into TIMEOUT_ERROR.
    # Without the wait, a missing verdict exhausts its retries at once and evaluate.py records its ERROR text.
    asyncio.sleep = no_wait
    dotenv.load_dotenv = lambda *a, **k: False  # evaluate.py would load the project .env into this process
    try:
        os.chdir(workdir)
        sys.path.insert(0, str(workdir))
        for rel in responses:
            sys.argv = ["evaluate.py", "--input_file", rel, "--concurrency_limit", str(concurrency)]
            runpy.run_path(str(workdir / "evaluate.py"), run_name="__main__")
    finally:
        openai.AsyncOpenAI, asyncio.sleep, dotenv.load_dotenv, cwd, path, argv = saved
        os.chdir(cwd)
        sys.path[:] = path
        sys.argv[:] = argv


def judge(workdir: Path, responses: list[str], judge_model: str, concurrency: int) -> dict:
    """Record, judge with Claude, replay. Leaves evaluate.py's evaluated CSVs in the work directory and
    returns {"prompts", "judged", "judge_errors"}."""
    from src.evals.judge_runner import run_judge

    record = workdir / RECORD_DIR
    if record.exists():
        shutil.rmtree(record)  # a scratch copy this module made; the responses themselves are untouched
    record.mkdir()
    for f in ("evaluate.py", "prompts"):
        (shutil.copytree if (workdir / f).is_dir() else shutil.copy2)(workdir / f, record / f)
    for rel in responses:
        (record / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(workdir / rel, record / rel)
    recorded: dict = {}
    run_evaluate(record, responses, StandInClient(recorded, None), concurrency)

    vfile = workdir / VERDICTS_FILE
    done = {}
    if vfile.exists():
        for line in vfile.read_text().splitlines():
            r = json.loads(line)
            if r["verdict"] is not None:  # a judge error is asked again on a rerun
                done[r["key"]] = r["verdict"]
    for kind, tool, fields in (
        ("letter", LETTER_TOOL, ("reasoning", "answer")),
        ("numeric", NUMERIC_TOOL, NUMERIC_FIELDS),
    ):
        groups: dict[str, list[str]] = {}
        for k, r in recorded.items():
            if r["kind"] == kind and k not in done:
                groups.setdefault(r["system"], []).append(k)
        for system, keys in groups.items():
            with vfile.open("a") as out:

                def persist(i, v, keys=keys, out=out):
                    out.write(json.dumps({"key": keys[i], "verdict": v}) + "\n")
                    out.flush()

                verdicts = run_judge(judge_model, system, [recorded[k]["prompt"] for k in keys], tool,
                                     tool_extractor(tool["name"], fields), JUDGE_MAX_TOKENS, use_batch=False,
                                     concurrency=concurrency, on_result=persist)  # fmt: skip
            done.update(zip(keys, verdicts))
    run_evaluate(workdir, responses, StandInClient({}, done), concurrency)
    used = {k: done.get(k) for k in recorded}
    return {"prompts": len(recorded), "judged": sum(v is not None for v in used.values()),
            "judge_errors": sum(v is None for v in used.values())}  # fmt: skip
