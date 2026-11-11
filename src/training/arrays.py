"""Trainer-independent next-token arrays shared by training data builders."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_PRINT_SAMPLES = 2
SAMPLE_PREVIEW_TOKENS = 60
SAMPLE_PREVIEW_CHARS = 300


@dataclass(frozen=True)
class TrainingArrays:
    """Plain next-token training arrays, independent of any trainer SDK."""

    input_tokens: list[int]
    target_tokens: list[int]
    weights: list[float]
    rendered_text: str


def values_to_list(value) -> list:
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    if _is_tinker_tensor_data(value):
        return _tinker_tensor_data_to_list(value)
    if hasattr(value, "tolist"):
        result = value.tolist()
        if not isinstance(result, list):
            raise TypeError(f"{type(value).__name__}.tolist() returned {type(result).__name__}, expected list")
        return result
    if hasattr(value, "to_torch"):
        return values_to_list(value.to_torch())
    raise TypeError(f"unsupported tensor/list container: {type(value).__name__}")


def _is_tinker_tensor_data(value: Any) -> bool:
    try:
        from tinker import types
    except ImportError:
        return False
    return isinstance(value, types.TensorData)


def _tinker_tensor_data_to_list(value) -> list:
    data = values_to_list(value.data)
    shape = value.shape
    if not isinstance(shape, list | tuple) or not all(isinstance(dim, int) for dim in shape):
        raise TypeError(f"TensorData.shape must be a list of ints, got {shape!r}")
    if list(shape) != [len(data)]:
        raise ValueError(f"TensorData shape {shape!r} does not match data length {len(data)}")
    return data


def _check_tinker_dtype(value, *, context: str, expected: set[str]) -> None:
    if _is_tinker_tensor_data(value) and str(value.dtype) not in expected:
        expected_text = ", ".join(sorted(expected))
        raise ValueError(f"{context} TensorData dtype must be one of {expected_text}, got {value.dtype!r}")


def token_ids_to_list(value, *, context: str, allow_empty: bool = False) -> list[int]:
    _check_tinker_dtype(value, context=context, expected={"int64"})
    tokens = values_to_list(value)
    if not allow_empty and not tokens:
        raise ValueError(f"{context} has no tokens")
    for i, token in enumerate(tokens):
        if isinstance(token, bool) or not isinstance(token, int):
            raise ValueError(f"{context}[{i}] must be an int token id, got {token!r}")
        if token < 0:
            raise ValueError(f"{context}[{i}] must be nonnegative, got {token!r}")
    return tokens


def weights_to_list(value, *, context: str, allow_empty: bool = False) -> list[float]:
    _check_tinker_dtype(value, context=context, expected={"float32", "float64"})
    weights = finite_floats_to_list(value, context=context, allow_empty=allow_empty)
    for i, weight in enumerate(weights):
        if weight < 0:
            raise ValueError(f"{context}[{i}] must be nonnegative, got {weight!r}")
    return weights


def finite_floats_to_list(value, *, context: str, allow_empty: bool = False) -> list[float]:
    _check_tinker_dtype(value, context=context, expected={"float32", "float64"})
    weights = values_to_list(value)
    if not allow_empty and not weights:
        raise ValueError(f"{context} has no values")
    checked = []
    for i, weight in enumerate(weights):
        if isinstance(weight, bool) or not isinstance(weight, int | float):
            raise ValueError(f"{context}[{i}] must be numeric, got {weight!r}")
        weight = float(weight)
        if not math.isfinite(weight):
            raise ValueError(f"{context}[{i}] must be finite, got {weight!r}")
        checked.append(weight)
    return checked


def validate_training_arrays(arrays: TrainingArrays) -> TrainingArrays:
    """Validate and normalize one next-token training row."""
    if not isinstance(arrays, TrainingArrays):
        raise TypeError(f"expected TrainingArrays, got {type(arrays).__name__}")

    input_tokens = token_ids_to_list(arrays.input_tokens, context="input_tokens")
    target_tokens = token_ids_to_list(arrays.target_tokens, context="target_tokens")
    weights = weights_to_list(arrays.weights, context="weights")
    if len(input_tokens) != len(target_tokens) or len(target_tokens) != len(weights):
        raise ValueError(
            "training arrays length mismatch: "
            f"{len(input_tokens)} input_tokens, {len(target_tokens)} target_tokens, {len(weights)} weights"
        )
    if all(weight <= 0 for weight in weights):
        raise ValueError("training arrays have no positive loss weights")
    if not isinstance(arrays.rendered_text, str):
        raise ValueError(f"rendered_text must be a string, got {type(arrays.rendered_text).__name__}")
    return TrainingArrays(
        input_tokens=input_tokens,
        target_tokens=target_tokens,
        weights=weights,
        rendered_text=arrays.rendered_text,
    )


def arrays_to_tinker_datum(arrays: TrainingArrays):
    """Convert plain training arrays to a Tinker cross-entropy Datum."""
    from tinker import types

    arrays = validate_training_arrays(arrays)
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=arrays.input_tokens),
        loss_fn_inputs={
            "target_tokens": types.TensorData(
                data=arrays.target_tokens,
                dtype="int64",
                shape=[len(arrays.target_tokens)],
            ),
            "weights": types.TensorData(
                data=arrays.weights,
                dtype="float32",
                shape=[len(arrays.weights)],
            ),
        },
    )


def load_training_arrays_jsonl(path: str | Path) -> list[TrainingArrays]:
    """Load strict, already-rendered next-token arrays from JSONL.

    This is the trainer-neutral interchange for mixtures whose rows need different
    renderers or loss masks (for example raw GPT-OSS documents plus Harmony replay).
    Tokenization stays upstream; the shared trainer still owns optimization, logging,
    checkpointing, resume, and cost accounting.
    """
    from src.jsonl import read_jsonl_numbered

    path = Path(path)
    rows: list[TrainingArrays] = []
    for line_number, value in read_jsonl_numbered(path):
        missing = {"input_tokens", "target_tokens", "weights"} - set(value)
        if missing:
            raise ValueError(f"{path}:{line_number}: missing fields {sorted(missing)}")
        try:
            row = validate_training_arrays(
                TrainingArrays(
                    input_tokens=value["input_tokens"],
                    target_tokens=value["target_tokens"],
                    weights=value["weights"],
                    rendered_text=value.get("rendered_text", ""),
                )
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"{path}:{line_number}: {error}") from error
        rows.append(row)
    return rows
