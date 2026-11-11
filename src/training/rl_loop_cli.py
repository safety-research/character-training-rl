"""Flags for the online RL round loop. The flag set is the record of the run.

Every knob is required: there are no defaults anywhere, so the command line alone says what ran.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from src.training.tinker_common import (
    _bool,
    _nonneg_float,
    _nonneg_int,
    _positive_float,
    _positive_int,
)

#: Comparison operators a gate may use, mapped to the test that FAILS the round. Spelled `gt`/`lt`
#: rather than `>`/`<` because a bare `>` in an argument is a shell redirect, and these flags get
#: copied off printed command lines.
GATE_OPS = {"gt": lambda v, t: v > t, "lt": lambda v, t: v < t}

#: A gate fires only after this many consecutive failing rounds; one bad round is batch noise.
GATE_CONSECUTIVE = 2

#: `--init-sampler`/`--init-state` value meaning "start from the base model, no prior adapter".
BASE_START = "base"


@dataclass(frozen=True)
class Gate:
    """A halt condition read from the rollout's own stats file by key."""

    key: str
    op: str
    threshold: float

    def fails(self, stats: dict) -> bool:
        if self.key not in stats:
            raise KeyError(f"gate key {self.key!r} is not in the rollout stats: {sorted(stats)}")
        value = stats[self.key]
        if value is None:
            raise ValueError(f"gate key {self.key!r} is None; a gate cannot read a missing measurement")
        return GATE_OPS[self.op](float(value), self.threshold)


def _gates(value: str) -> tuple[Gate, ...]:
    """Comma-separated `key:op:threshold`, or `none`. Matching `--wandb-tags`: a repeatable flag
    is written as one comma-separated string. A repeated
    gate is refused: it would count twice per round and halt after one failure."""
    if value == "none":
        return ()
    gates = tuple(_one_gate(part.strip()) for part in value.split(",") if part.strip())
    if len(set(gates)) != len(gates):
        raise argparse.ArgumentTypeError(f"a gate is repeated in {value!r}")
    return gates


def _one_gate(value: str) -> Gate:
    """`key:op:threshold`, e.g. `drop_rate:gt:0.25` halts when the drop rate exceeds 0.25."""
    parts = value.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"gate must be key:op:threshold, got {value!r}")
    key, op, threshold = parts
    if op not in GATE_OPS:
        raise argparse.ArgumentTypeError(f"gate op must be one of {sorted(GATE_OPS)}, got {op!r}")
    try:
        return Gate(key, op, float(threshold))
    except ValueError:
        raise argparse.ArgumentTypeError(f"gate threshold must be a number, got {threshold!r}") from None


def _passthroughs(value: str) -> tuple[tuple[str, str], ...]:
    """Comma-separated `flag=value`, or `none`. A value cannot contain a comma, and a flag always
    takes a value (`flag=` passes the empty string); environments needing more take a file path."""
    if value == "none":
        return ()
    return tuple(_one_passthrough(part.strip()) for part in value.split(",") if part.strip())


def _one_passthrough(value: str) -> tuple[str, str]:
    """`flag=value` for an environment-specific knob, handed to the rollout script as `--flag`.

    Written without the leading dashes on purpose: argparse refuses an option value that begins
    with `--`, so the dashed form cannot be passed at all.
    """
    if value.startswith("-") or "=" not in value:
        raise argparse.ArgumentTypeError(f"rollout-arg must look like flag=value with no leading dashes, got {value!r}")
    flag, _, rest = value.partition("=")
    if not flag:
        raise argparse.ArgumentTypeError(f"rollout-arg needs a flag name, got {value!r}")
    return f"--{flag}", rest


def _script(value: str) -> str:
    """An existing file, checked against the current directory at parse time so a typo fails
    when the command line is parsed rather than in round 0. The loop makes the path absolute
    before it builds any command."""
    if value != BASE_START and not Path(value).is_file():
        raise argparse.ArgumentTypeError(f"script {value!r} is not a file")
    return value


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rl_loop", allow_abbrev=False, description=__doc__)
    p.add_argument(
        "--rollout-script",
        required=True,
        type=_script,
        help="script that samples and scores one round; writes GRPO rows and <out>.stats.json",
    )
    p.add_argument(
        "--rollout-arg",
        required=True,
        type=_passthroughs,
        help="comma-separated flag=value (no leading dashes), each passed to the rollout script as --flag, or 'none'",
    )
    p.add_argument("--base-model", required=True)
    p.add_argument("--renderer-name", required=True)
    p.add_argument(
        "--init-sampler", required=True, help=f"tinker:// sampler to start from, or {BASE_START!r}; lets RL follow SFT"
    )
    p.add_argument("--init-state", required=True, help=f"tinker:// optimizer state to start from, or {BASE_START!r}")
    p.add_argument(
        "--resume",
        required=True,
        type=_bool,
        help="continue an existing rl_history.json. False refuses to start when one "
        "exists, rather than silently restarting and overwriting the record",
    )
    p.add_argument("--rounds", required=True, type=_positive_int)
    p.add_argument("--n-prompts", required=True, type=_positive_int)
    p.add_argument("--group-size", required=True, type=_positive_int)
    p.add_argument("--temperature", required=True, type=_positive_float)
    p.add_argument("--top-p", required=True, type=_positive_float)
    p.add_argument("--max-tokens", required=True, type=_positive_int)
    p.add_argument("--max-concurrent", required=True, type=_positive_int)
    p.add_argument("--seed", required=True, type=_nonneg_int, help="trainer seed; the rollout gets seed + round")
    p.add_argument(
        "--prompt-tokens",
        required=True,
        type=_positive_int,
        help="tokens per rendered prompt, for the dry-run cost estimate only; nothing generated depends on it",
    )
    p.add_argument(
        "--sampled-tokens",
        required=True,
        type=_positive_int,
        help="expected sampled tokens per completion, for the cost estimate only; "
        "at most --max-tokens, which stays the cap and prices nothing",
    )
    p.add_argument(
        "--heldout-every",
        required=True,
        type=_nonneg_int,
        help="run --heldout-script every N rounds and always on the last, a round skipped for uniform reward "
        "included; 0 disables. "
        "Checkpoints are saved every round regardless, so a jump found at one evaluation can "
        "be bracketed by evaluating the rounds either side later, without retraining",
    )
    p.add_argument(
        "--heldout-script",
        required=True,
        type=_script,
        help=f"script evaluating a checkpoint on a fixed set; writes rows.jsonl (every sampled response) and summary.json. "
        f"{BASE_START!r} when --heldout-every is 0",
    )
    p.add_argument(
        "--heldout-completions",
        required=True,
        type=_nonneg_int,
        help="completions one held-out evaluation samples, so it enters the cost ceiling; 0 when --heldout-every is 0",
    )
    p.add_argument(
        "--heldout-arg",
        required=True,
        type=_passthroughs,
        help="comma-separated flag=value passed to the held-out script, or 'none'",
    )
    p.add_argument(
        "--gate",
        required=True,
        type=_gates,
        help=f"comma-separated key:op:threshold read from the rollout stats, or 'none'; "
        f"halts after {GATE_CONSECUTIVE} consecutive failing rounds",
    )
    # --- the training step, forwarded to src.training.tinker_rl grpo -----------------------
    p.add_argument("--learning-rate", required=True, type=_positive_float)
    p.add_argument("--lora-rank", required=True, type=_positive_int)
    p.add_argument(
        "--train-batch-size",
        required=True,
        type=_nonneg_int,
        help="rows per optimizer step; 0 means one step over the whole surviving batch",
    )
    p.add_argument("--grpo-loss-fn", required=True, choices=("importance_sampling", "ppo", "cispo"))
    p.add_argument("--grpo-clip-low-threshold", required=True, type=_nonneg_float)
    p.add_argument("--grpo-clip-high-threshold", required=True, type=_nonneg_float)
    p.add_argument("--train-mlp", required=True, type=_bool)
    p.add_argument("--train-attn", required=True, type=_bool)
    p.add_argument(
        "--train-unembed",
        required=True,
        type=_bool,
        help="checked against the init checkpoint when starting from one: the loaded architecture decides, the flag only records",
    )
    p.add_argument("--beta1", required=True, type=_positive_float)
    p.add_argument("--beta2", required=True, type=_positive_float)
    p.add_argument("--eps", required=True, type=_positive_float)
    p.add_argument("--weight-decay", required=True, type=_nonneg_float)
    p.add_argument("--grad-clip-norm", required=True, type=_nonneg_float)
    p.add_argument("--dry-run", required=True, type=_bool)
    p.add_argument("--out", required=True)
    return p


def parse_args(argv: list[str]) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.grpo_loss_fn == "importance_sampling":
        if args.grpo_clip_low_threshold or args.grpo_clip_high_threshold:
            raise SystemExit(
                "importance_sampling takes no clip thresholds: an unclipped ratio is only "
                "correct when generation and update share a policy."
            )
        if args.train_batch_size != 0:
            raise SystemExit(
                "importance_sampling assumes an on-policy update, but --train-batch-size "
                f"{args.train_batch_size} produces several stale minibatch steps per "
                "generation, where one token with a large ratio can dominate. Use 0, or cispo."
            )
    elif not args.grpo_clip_high_threshold:
        raise SystemExit(f"{args.grpo_loss_fn} needs a non-zero --grpo-clip-high-threshold")
    if (args.init_sampler == BASE_START) != (args.init_state == BASE_START):
        raise SystemExit(f"--init-sampler and --init-state must both be {BASE_START!r} or neither")
    if args.heldout_every and args.heldout_script == BASE_START:
        raise SystemExit("--heldout-every is non-zero but --heldout-script was not given")
    if args.sampled_tokens > args.max_tokens:
        raise SystemExit(f"--sampled-tokens {args.sampled_tokens} exceeds the cap --max-tokens {args.max_tokens}")
    if bool(args.heldout_every) != bool(args.heldout_completions):
        raise SystemExit(
            "--heldout-completions must be positive exactly when --heldout-every is; it prices the evaluations"
        )
    return args
