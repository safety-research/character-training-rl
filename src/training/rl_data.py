"""Data contracts for grouped reward-based language-model training."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.jsonl import read_jsonl_numbered
from src.training.arrays import finite_floats_to_list, token_ids_to_list

IMPORTANCE_SAMPLING_LOSS = "importance_sampling"
GRPO_ROLLOUT_KEYS = (
    "group_id",
    "prompt_tokens",
    "completion_tokens",
    "completion_logprobs",
    "reward",
    "split",
)


@dataclass(frozen=True)
class RLTrainingArrays:
    """Plain arrays for a prompt/completion sample with scalar advantage."""

    input_tokens: list[int]
    target_tokens: list[int]
    logprobs: list[float]
    advantages: list[float]
    prompt_tokens: list[int]
    completion_tokens: list[int]


@dataclass(frozen=True)
class GRPORollout:
    """One sampled completion with a reward, grouped for advantage computation."""

    group_id: str
    prompt_tokens: list[int]
    completion_tokens: list[int]
    completion_logprobs: list[float]
    reward: float
    split: str


def _require_exact_keys(path: Path, line_no: int, row: dict[str, Any], keys: tuple[str, ...]) -> None:
    missing = [key for key in keys if key not in row]
    extra = [key for key in row if key not in keys]
    if missing or extra:
        raise ValueError(f"{path}:{line_no}: keys must be exactly {keys}; missing={missing}, extra={extra}")


def _nonempty_str(value: Any, *, path: Path, line_no: int, key: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path}:{line_no}: {key} must be a non-empty string")
    return value


def _finite_float(value: Any, *, path: Path, line_no: int, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{path}:{line_no}: {key} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{path}:{line_no}: {key} must be finite")
    return value


def load_grpo_rollouts_jsonl(path: str | Path, *, split: str) -> list[GRPORollout]:
    """Load strict precomputed rollout rows for grouped policy-gradient updates."""
    path = Path(path)
    if not isinstance(split, str) or not split.strip():
        raise ValueError("split must be a non-empty string")
    rollouts = []
    for line_no, row in read_jsonl_numbered(path):
        _require_exact_keys(path, line_no, row, GRPO_ROLLOUT_KEYS)
        row_split = _nonempty_str(row["split"], path=path, line_no=line_no, key="split")
        prompt_tokens = token_ids_to_list(row["prompt_tokens"], context=f"{path}:{line_no}: prompt_tokens")
        completion_tokens = token_ids_to_list(row["completion_tokens"], context=f"{path}:{line_no}: completion_tokens")
        completion_logprobs = finite_floats_to_list(
            row["completion_logprobs"],
            context=f"{path}:{line_no}: completion_logprobs",
        )
        if len(completion_tokens) != len(completion_logprobs):
            raise ValueError(
                f"{path}:{line_no}: completion_tokens has {len(completion_tokens)} tokens but "
                f"completion_logprobs has {len(completion_logprobs)} values"
            )
        rollout = GRPORollout(
            group_id=_nonempty_str(row["group_id"], path=path, line_no=line_no, key="group_id"),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            completion_logprobs=completion_logprobs,
            reward=_finite_float(row["reward"], path=path, line_no=line_no, key="reward"),
            split=row_split,
        )
        if rollout.split == split:
            rollouts.append(rollout)
    if not rollouts:
        raise ValueError(f"{path}: no rollout rows for split {split!r}")
    return rollouts


def compute_dr_grpo_advantages(rewards: list[float]) -> list[float]:
    """Compute per-group advantages as reward minus group mean."""
    rewards = finite_floats_to_list(rewards, context="rewards", allow_empty=True)
    if len(rewards) < 2:
        raise ValueError("rewards must contain at least two samples")
    mean_reward = sum(rewards) / len(rewards)
    if len(set(rewards)) <= 1:
        return [0.0] * len(rewards)
    return [reward - mean_reward for reward in rewards]


def build_grpo_training_arrays(rollouts: list[GRPORollout]) -> list[RLTrainingArrays]:
    """Convert grouped rollout rows to RL arrays with DR-GRPO advantages."""
    if not rollouts:
        raise ValueError("no rollouts")
    groups: dict[str, list[GRPORollout]] = defaultdict(list)
    for rollout in rollouts:
        groups[rollout.group_id].append(rollout)

    arrays = []
    for group_id in sorted(groups):
        group = groups[group_id]
        prompt_tokens = group[0].prompt_tokens
        mismatched = [i for i, rollout in enumerate(group) if rollout.prompt_tokens != prompt_tokens]
        if mismatched:
            raise ValueError(f"group_id {group_id!r} contains multiple prompt token sequences")
        rewards = [rollout.reward for rollout in group]
        advantages = compute_dr_grpo_advantages(rewards)
        for rollout, advantage in zip(group, advantages, strict=True):
            arrays.append(
                build_rl_training_arrays(
                    prompt_tokens=rollout.prompt_tokens,
                    completion_tokens=rollout.completion_tokens,
                    completion_logprobs=rollout.completion_logprobs,
                    advantage=advantage,
                )
            )
    return arrays


def build_rl_training_arrays(
    *,
    prompt_tokens,
    completion_tokens,
    completion_logprobs,
    advantage: float,
) -> RLTrainingArrays:
    """Build aligned next-token arrays for Tinker importance-sampling RL."""
    prompt_tokens = token_ids_to_list(prompt_tokens, context="prompt_tokens")
    completion_tokens = token_ids_to_list(completion_tokens, context="completion_tokens")
    completion_logprobs = finite_floats_to_list(completion_logprobs, context="completion_logprobs")
    if len(completion_tokens) != len(completion_logprobs):
        raise ValueError(
            f"completion_tokens has {len(completion_tokens)} tokens but completion_logprobs has "
            f"{len(completion_logprobs)} values"
        )
    if isinstance(advantage, bool) or not isinstance(advantage, int | float):
        raise ValueError(f"advantage must be numeric, got {advantage!r}")
    advantage = float(advantage)
    if not math.isfinite(advantage):
        raise ValueError(f"advantage must be finite, got {advantage!r}")

    full_tokens = prompt_tokens + completion_tokens
    if len(full_tokens) < 2:
        raise ValueError("prompt plus completion must contain at least two tokens")

    logprobs_full = [0.0] * len(prompt_tokens) + completion_logprobs
    advantages_full = [0.0] * len(prompt_tokens) + [advantage] * len(completion_tokens)

    input_tokens = full_tokens[:-1]
    target_tokens = full_tokens[1:]
    logprobs = logprobs_full[1:]
    advantages = advantages_full[1:]
    if not (len(input_tokens) == len(target_tokens) == len(logprobs) == len(advantages)):
        raise ValueError("RL arrays are misaligned")

    return RLTrainingArrays(
        input_tokens=input_tokens,
        target_tokens=target_tokens,
        logprobs=logprobs,
        advantages=advantages,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


def rl_arrays_to_tinker_datum(arrays: RLTrainingArrays):
    """Convert RL arrays to a Tinker Datum for `importance_sampling`."""
    from tinker import types

    arrays = validate_rl_training_arrays(arrays)
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=arrays.input_tokens),
        loss_fn_inputs={
            "target_tokens": types.TensorData(
                data=arrays.target_tokens,
                dtype="int64",
                shape=[len(arrays.target_tokens)],
            ),
            "logprobs": types.TensorData(
                data=arrays.logprobs,
                dtype="float32",
                shape=[len(arrays.logprobs)],
            ),
            "advantages": types.TensorData(
                data=arrays.advantages,
                dtype="float32",
                shape=[len(arrays.advantages)],
            ),
        },
    )


def validate_rl_training_arrays(arrays: RLTrainingArrays) -> RLTrainingArrays:
    """Validate and normalize one reward-training row."""
    if not isinstance(arrays, RLTrainingArrays):
        raise TypeError(f"expected RLTrainingArrays, got {type(arrays).__name__}")
    input_tokens = token_ids_to_list(arrays.input_tokens, context="input_tokens")
    target_tokens = token_ids_to_list(arrays.target_tokens, context="target_tokens")
    logprobs = finite_floats_to_list(arrays.logprobs, context="logprobs")
    advantages = finite_floats_to_list(arrays.advantages, context="advantages")
    prompt_tokens = token_ids_to_list(arrays.prompt_tokens, context="prompt_tokens")
    completion_tokens = token_ids_to_list(arrays.completion_tokens, context="completion_tokens")
    if not (len(input_tokens) == len(target_tokens) == len(logprobs) == len(advantages)):
        raise ValueError(
            "RL arrays length mismatch: "
            f"{len(input_tokens)} input_tokens, {len(target_tokens)} target_tokens, "
            f"{len(logprobs)} logprobs, {len(advantages)} advantages"
        )
    if input_tokens != (prompt_tokens + completion_tokens)[:-1]:
        raise ValueError("input_tokens do not match prompt_tokens + completion_tokens")
    if target_tokens != (prompt_tokens + completion_tokens)[1:]:
        raise ValueError("target_tokens do not match prompt_tokens + completion_tokens")
    n_prompt_targets = len(prompt_tokens) - 1
    if any(value != 0.0 for value in logprobs[:n_prompt_targets]):
        raise ValueError("prompt target logprobs must be zero")
    if any(value != 0.0 for value in advantages[:n_prompt_targets]):
        raise ValueError("prompt target advantages must be zero")
    completion_logprobs = logprobs[n_prompt_targets:]
    completion_advantages = advantages[n_prompt_targets:]
    if len(completion_logprobs) != len(completion_tokens):
        raise ValueError("completion logprobs are not aligned with completion tokens")
    if len(completion_advantages) != len(completion_tokens):
        raise ValueError("completion advantages are not aligned with completion tokens")
    if not completion_advantages:
        raise ValueError("completion advantages must be non-empty")
    if len(set(completion_advantages)) != 1:
        raise ValueError("completion advantages must share one scalar advantage")
    return RLTrainingArrays(
        input_tokens=input_tokens,
        target_tokens=target_tokens,
        logprobs=logprobs,
        advantages=advantages,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
