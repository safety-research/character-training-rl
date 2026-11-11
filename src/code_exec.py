"""Run untrusted model code against assertions, and the two parsers that feed it.

Every stage that decides pass/fail by executing a candidate goes through `run_code`: assertion
audits, base-rate measurement, RL rewards, held-out evaluation. Nothing here knows about
rewards, hacking surfaces, or prompts — those belong to the experiment that defines them.

The sandbox is a required argument, never a default, because the choice decides whether
untrusted code can reach the host filesystem and network.
"""

from __future__ import annotations

import ast
import json
import os
import secrets
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field

from src.thinking import answer_of

# Wall-clock ceiling for one sandboxed execution, seconds. Matches the upstream LeetCode
# filtering timeout, so a task that solved in time there still solves here.
EXEC_TIMEOUT_S = 10
# Address-space ceiling per execution, bytes.
EXEC_MEMORY_LIMIT_BYTES = 4 * 1024 * 1024 * 1024

SANDBOX_SUBPROCESS = "subprocess"
SANDBOX_DOCKER = "docker"
SANDBOXES = (SANDBOX_SUBPROCESS, SANDBOX_DOCKER)
DOCKER_IMAGE = "python:3.13-slim"


def call_source(assert_src: str) -> str | None:
    """Normalised source of the call on the left of `assert <call> == <expected>`."""
    try:
        tree = ast.parse(assert_src.strip())
    except SyntaxError:
        return None
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assert):
        return None
    test = tree.body[0].test
    if not isinstance(test, ast.Compare) or len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
        return None
    return ast.unparse(test.left)


def parse_code(response: str) -> str:
    """Extract the last fenced python block from the answer, or the whole answer if unfenced.

    Reasoning is stripped first: a fenced block inside the think block is a draft, not the
    submission, and grading it would score work the model went on to discard.

    This is the simple rule. Two things it gets wrong outside `<think>`-style models:

    1. It finds the reasoning boundary via `src.thinking`, i.e. `</think>`. Harmony renderers
       (gpt_oss) have no such tag, so the strip silently does nothing and drafts written inside
       the reasoning become candidates for "the last block".
    2. Taking the last block alone, or joining every block, both mis-extract discursive answers:
       joining blocks glues explanatory prose onto the solution, which penalises a model that
       explains its answer far more than one that does not. The stricter rule is to prefer the
       last block that both parses and defines the entry point.

    So: fine for a `</think>` model emitting one code block. Otherwise, pass the reasoning
    boundary and the entry point explicitly and use the stricter rule.
    """
    answer = answer_of(response)
    fence = "```"
    if fence not in answer:
        return answer.strip()
    blocks = []
    parts = answer.split(fence)
    for i in range(1, len(parts), 2):
        block = parts[i]
        if block.startswith("python"):
            block = block[len("python") :]
        blocks.append(block.strip())
    return blocks[-1] if blocks else answer.strip()


# --- sandboxed execution -------------------------------------------------------------

_RUNNER = textwrap.dedent(
    """
    print("<<<STARTED>>>", flush=True)
    import json, resource, sys, io, contextlib


    def _run():
        limit = int(sys.argv[1])
        marker = sys.argv[2]
        del sys.argv[2:]
        for which, value in ((resource.RLIMIT_AS, limit), (resource.RLIMIT_NPROC, 256)):
            try:
                soft, hard = resource.getrlimit(which)
                if hard != resource.RLIM_INFINITY:
                    value = min(value, hard)
                resource.setrlimit(which, (value, hard))
            except (ValueError, OSError):
                pass

        payload = json.loads(sys.stdin.read())
        ns = {}
        results = []
        ok = True
        compiled = True
        try:
            exec(compile(payload["setup"], "<setup>", "exec"), ns)
            exec(compile(payload["code"], "<code>", "exec"), ns)
        except BaseException:
            compiled = False
            ok = False
        if compiled:
            for stmt in payload["tests"]:
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        exec(compile(stmt, "<test>", "exec"), ns)
                    results.append(True)
                except BaseException:
                    results.append(False)
                    ok = False
        print(marker + json.dumps({"compiled": compiled, "ok": ok, "results": results}))


    _run()
    """
)
# The result line starts with a marker drawn fresh for each run and handed to the runner as its last
# argument. The runner keeps it in a local variable and removes it from sys.argv before any candidate
# code runs, and it is in no file, so a candidate that prints a result-shaped line (at import, or from
# an exit hook) does not know the marker and its line is not read. This is not a hard boundary: the
# candidate runs in the runner's process and could still dig the marker out of the interpreter's
# frames or the process command line; the sandbox choice, not the marker, is the isolation.
RESULT_MARKER_BYTES = 16
# The runner prints this before touching candidate code, so a missing RESULT can be told apart
# from a runner that never ran at all. Candidate code CAN kill the process (os._exit) and so
# suppress RESULT; that is a property of the candidate, not a harness fault, and conflating the
# two either crashes a whole run or silently scores a real harness break as a legitimate run.
START_MARKER = "<<<STARTED>>>"


@dataclass
class RunResult:
    compiled: bool = False
    timeout: bool = False
    results: list[bool] = field(default_factory=list)
    # The candidate killed the process before a verdict was printed. Never counts as a pass:
    # otherwise `os._exit(0)` inside run_tests would read as "raised nothing" and be rewarded.
    no_verdict: bool = False

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(self.results) / len(self.results)

    @property
    def all_pass(self) -> bool:
        if self.no_verdict:
            return False
        return bool(self.results) and all(self.results)


def run_code(setup: str, code: str, tests: list[str], *, sandbox: str) -> RunResult:
    """Execute `code` against `tests` and report which assertions held.

    `sandbox` is required; the choice determines whether untrusted model code can touch the
    host filesystem or network. The verdict is read from the runner's own result line, which carries
    a marker drawn at random for this run; a result line the candidate prints itself is ignored.
    """
    if sandbox not in SANDBOXES:
        raise ValueError(f"unknown sandbox {sandbox!r}; expected one of {SANDBOXES}")
    if not code.strip():
        return RunResult(compiled=False)
    payload = json.dumps({"setup": setup, "code": code, "tests": tests})
    marker = f"<<<RESULT {secrets.token_hex(RESULT_MARKER_BYTES)}>>>"

    if sandbox == SANDBOX_SUBPROCESS:
        argv = [os.path.realpath(sys.executable), "-c", _RUNNER, str(EXEC_MEMORY_LIMIT_BYTES), marker]
    else:
        argv = [
            "docker",
            "run",
            "--rm",
            "-i",
            "--network",
            "none",
            "--read-only",
            "--memory",
            "4g",
            "--pids-limit",
            "256",
            DOCKER_IMAGE,
            "python",
            "-c",
            _RUNNER,
            str(EXEC_MEMORY_LIMIT_BYTES),
            marker,
        ]

    try:
        proc = subprocess.run(argv, input=payload, capture_output=True, text=True, timeout=EXEC_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return RunResult(compiled=True, timeout=True)

    if marker not in proc.stdout:
        if START_MARKER in proc.stdout:
            # The runner started and the candidate killed the process before the result was
            # printed (os._exit, signal, C-level abort). A property of the candidate: record it
            # as a run that produced no verdict rather than crashing the caller or, worse, letting
            # a process-kill count as "run_tests raised nothing" and earn the correctness reward.
            return RunResult(compiled=True, no_verdict=True)
        raise RuntimeError(f"sandbox runner never started (rc={proc.returncode}).\nstderr: {proc.stderr[:2000]}")
    if proc.stdout.count(marker) != 1:
        # The runner prints its marker once. A second copy means the candidate found the marker and
        # printed a result of its own; which line is the runner's cannot be told, so there is no verdict.
        return RunResult(compiled=True, no_verdict=True)
    # Only the first line after the marker is ours. Candidate stdout buffered until interpreter
    # exit, or written by an atexit hook or a thread, lands AFTER our line.
    tail = proc.stdout.split(marker, 1)[1]
    parsed = json.loads(tail.splitlines()[0].strip())
    return RunResult(compiled=parsed["compiled"], results=parsed["results"])
