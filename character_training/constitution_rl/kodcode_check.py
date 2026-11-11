"""KodCode correctness check: the submission is the last fenced block of the final channel that compiles
and defines the entry function (failing that, every compiling block joined); it runs in
`src.code_exec.run_code` as module `solution`, the test file is executed, then each test function is
called. Identical to the checker that selected the context-distilled coding rows
(s16_generate_coding), so RL correctness means what correctness meant there.
"""

from __future__ import annotations

import ast
import re
from typing import Any

from src.code_exec import run_code

FENCED_BLOCK = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL)
# Run as the first test statement: expose everything the candidate defined as module `solution`,
# so the test file's `from solution import f` resolves to the candidate's code.
MAKE_SOLUTION_MODULE = (
    "import sys as _s, types as _t\n"
    "_m = _t.ModuleType('solution')\n"
    "_m.__dict__.update({k: v for k, v in dict(globals()).items() if not k.startswith('__')})\n"
    "_s.modules['solution'] = _m"
)


def _compiles(src: str) -> bool:
    try:
        compile(src, "<candidate>", "exec")
        return True
    except (SyntaxError, ValueError):
        return False


def _defines(src: str, name: str) -> bool:
    tree = ast.parse(src)
    return any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name for n in tree.body)


def extract_submission(final: str, entry: str) -> str | None:
    """The last compiling fenced block of the final channel that defines `entry` at top level;
    failing that, every compiling block joined; None when nothing compiles."""
    blocks = [b.strip() for b in FENCED_BLOCK.findall(final) if b.strip()]
    compiling = [b for b in blocks if _compiles(b)]
    for b in reversed(compiling):
        if _defines(b, entry):
            return b
    return "\n\n".join(compiling) if compiling else None


def test_statements(test_src: str) -> list[str]:
    """What run_code executes after the candidate: make `solution`, run the test file, call each test."""
    names = [n.name for n in ast.parse(test_src).body if isinstance(n, ast.FunctionDef) and n.name.startswith("test")]
    return [MAKE_SOLUTION_MODULE, test_src] + [f"{n}()" for n in names]


def _site_setup() -> str:
    """run_code's subprocess runs the base interpreter, outside this project's virtualenv; KodCode tests
    import pytest (pytest.raises) and some solutions use numpy. Put the virtualenv's packages on the
    path. Problems needing a package the env lacks fail the grader self-test and are dropped. Matplotlib is
    forced onto its non-interactive backend so plotting code never opens a window on the host, and each run
    starts in a fresh temporary directory so files the code writes never land in the repo."""
    import sysconfig

    return (
        f"import os, sys, tempfile\nos.environ['MPLBACKEND'] = 'Agg'\n"
        f"os.chdir(tempfile.mkdtemp(prefix='kodcode_run_'))\n"
        f"sys.path.insert(0, {sysconfig.get_paths()['purelib']!r})"
    )


SITE_SETUP = _site_setup()


def check(final: str, row: dict[str, Any], sandbox: str) -> dict[str, Any]:
    code = extract_submission(final, row["entry_point"])
    if code is None:
        return {
            "correct": False,
            "code_found": False,
            "compiled": False,
            "timeout": False,
            "tests_passed": 0,
            "tests_total": 0,
        }
    r = run_code(SITE_SETUP, code, test_statements(row["test"]), sandbox=sandbox)
    n_tests = len(test_statements(row["test"])) - 2
    return {
        "correct": r.all_pass,
        "code_found": True,
        "compiled": r.compiled,
        "timeout": r.timeout,
        "tests_passed": sum(r.results[2:]),
        "tests_total": n_tests,
    }
