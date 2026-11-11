"""Hand an evaluation's command line a full set of knobs, written as a mapping of flag name to value.

Every evaluation CLI here takes each knob as a required flag, plus three that say where and how to
run rather than what: the log directory, the previous log to continue from, and the dry-run switch.
These helpers supply those three, so a test states only the knobs.
"""

from __future__ import annotations

import argparse
from typing import Callable

RUN_FLAGS = ("log-dir", "retry-log", "dry-run")


def flags(build_parser: Callable[[], argparse.ArgumentParser]) -> set[str]:
    """Every knob the CLI accepts, dashes as underscores, without the three run flags and --help."""
    out = set()
    for action in build_parser()._actions:
        long = [o[2:] for o in action.option_strings if o.startswith("--")]
        if long and not isinstance(action, argparse._HelpAction) and long[0] not in RUN_FLAGS:
            out.add(long[0].replace("-", "_"))
    return out


def argv(args: dict, *, log_dir: str = "/nonexistent/validate") -> list[str]:
    """The dry-run command line for these knobs: a fresh run (no previous log) into `log_dir`."""
    out: list[str] = []
    for key, value in args.items():
        text = ("true" if value else "false") if isinstance(value, bool) else str(value)
        out += [f"--{key.replace('_', '-')}", text]
    return [*out, "--log-dir", log_dir, "--retry-log", "none", "--dry-run"]


def validate(parse_args: Callable[[list[str]], object], args: dict) -> None:
    """Let the CLI's own parser check these knobs; a refusal (argparse exits) is a ValueError."""
    try:
        parse_args(argv(args))
    except SystemExit as e:
        raise ValueError(f"the CLI rejected these args (exit {e.code}); see the message above") from None
