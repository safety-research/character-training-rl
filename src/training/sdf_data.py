"""Synthetic-document fine-tuning data conversion.

This module turns strict document JSONL rows into next-token training arrays.
It performs no model calls and launches no training jobs; callers provide the
tokenizer and decide later whether these arrays are used by Tinker or another
trainer.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.training.arrays import (
    DEFAULT_PRINT_SAMPLES,
    SAMPLE_PREVIEW_CHARS,
    SAMPLE_PREVIEW_TOKENS,
    TrainingArrays,
    token_ids_to_list,
    validate_training_arrays,
)


@dataclass(frozen=True)
class SDFDataConfig:
    """Tokenization choices for one synthetic-document corpus build."""

    max_seq_length: int
    append_eos: bool
    # Optional exact document-boundary token. This is distinct from the tokenizer's
    # chat-response EOS on models such as GPT-OSS, where eos_token_id is <|return|>
    # but raw pretraining documents must end with <|endoftext|>.
    terminal_token: str | None = None


def _validate_config(config: SDFDataConfig) -> None:
    if isinstance(config.max_seq_length, bool) or not isinstance(config.max_seq_length, int):
        raise ValueError(f"max_seq_length must be an int, got {config.max_seq_length!r}")
    if config.max_seq_length < 1:
        raise ValueError(f"max_seq_length must be >= 1, got {config.max_seq_length!r}")
    if not isinstance(config.append_eos, bool):
        raise ValueError(f"append_eos must be a bool, got {config.append_eos!r}")
    if config.terminal_token is not None:
        if not isinstance(config.terminal_token, str) or not config.terminal_token:
            raise ValueError(f"terminal_token must be a non-empty string or None, got {config.terminal_token!r}")
        if config.append_eos:
            raise ValueError("append_eos and terminal_token are mutually exclusive")


def load_document_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load strict synthetic-document fine-tuning rows from JSONL.

    Every non-empty row must be a JSON object with a non-empty string `text`
    field. Other metadata is preserved on the row but ignored by tokenization.
    """
    path = Path(path)
    documents = []
    with path.open() as f:
        for line_no, raw in enumerate(f, start=1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {e}") from e
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: row must be a JSON object")
            _document_text(row, context=f"{path}:{line_no}")
            documents.append(row)
    if not documents:
        raise ValueError(f"{path}: no synthetic-document fine-tuning rows")
    return documents


def _document_text(document: dict[str, Any], *, context: str = "row") -> str:
    text = document.get("text")
    if not isinstance(text, str):
        raise ValueError(f"{context}: document must have a string 'text' field")
    if not text.strip():
        raise ValueError(f"{context}: document text is empty")
    return text


def _encode_text(tokenizer, text: str, *, add_special_tokens: bool) -> list[int]:
    if hasattr(tokenizer, "encode"):
        return token_ids_to_list(
            tokenizer.encode(text, add_special_tokens=add_special_tokens),
            context="tokenizer.encode input_ids",
        )
    if callable(tokenizer):
        encoded = tokenizer(
            text,
            truncation=False,
            padding=False,
            add_special_tokens=add_special_tokens,
            return_tensors=None,
        )
        if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
            raise ValueError("tokenizer call did not return a mapping with 'input_ids'")
        return token_ids_to_list(encoded["input_ids"], context="tokenizer call input_ids")
    raise ValueError("tokenizer must expose encode(...) or be callable")


def _decode_tokens(tokenizer, token_ids: list[int]) -> str:
    if not hasattr(tokenizer, "decode"):
        raise ValueError("tokenizer has no decode; cannot inspect rendered document text")
    return tokenizer.decode(token_ids)


def _eos_token_id(tokenizer) -> int:
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if not isinstance(eos_id, int):
        raise ValueError("append_eos=True requires tokenizer.eos_token_id to be an int")
    return eos_id


def _terminal_token_id(tokenizer, token: str) -> int:
    encoded = _encode_text(tokenizer, token, add_special_tokens=False)
    if len(encoded) != 1:
        raise ValueError(f"terminal_token must encode to exactly one token, got {len(encoded)} for {token!r}")
    return encoded[0]


def document_to_training_arrays(
    document: dict[str, Any],
    tokenizer,
    config: SDFDataConfig,
) -> TrainingArrays:
    """Convert one synthetic-document fine-tuning row to all-token next-token arrays."""
    _validate_config(config)
    text = _document_text(document)
    token_ids = _encode_text(tokenizer, text, add_special_tokens=False)
    if config.append_eos:
        eos_id = _eos_token_id(tokenizer)
        if not token_ids or token_ids[-1] != eos_id:
            token_ids.append(eos_id)
    elif config.terminal_token is not None:
        terminal_id = _terminal_token_id(tokenizer, config.terminal_token)
        if not token_ids or token_ids[-1] != terminal_id:
            token_ids.append(terminal_id)

    if len(token_ids) < 2:
        raise ValueError(f"document too short after tokenization: {len(token_ids)} tokens")
    if len(token_ids) > config.max_seq_length:
        raise ValueError(
            f"document is {len(token_ids)} tokens, over max_seq_length={config.max_seq_length}. "
            "Synthetic-document fine-tuning rows are never truncated; raise max_seq_length or split/drop this row."
        )

    return validate_training_arrays(
        TrainingArrays(
            input_tokens=token_ids[:-1],
            target_tokens=token_ids[1:],
            weights=[1.0] * (len(token_ids) - 1),
            rendered_text=_decode_tokens(tokenizer, token_ids),
        )
    )


def _print_samples(arrays: list[TrainingArrays], tokenizer, n: int) -> None:
    shown = min(n, len(arrays))
    print(f"\n{'=' * 60}\nSYNTHETIC-DOCUMENT DATA PREVIEW: {shown} sample(s)\n{'=' * 60}")
    for i, row in enumerate(arrays[:shown], start=1):
        n_total = len(row.input_tokens) + 1
        preview_ids = row.target_tokens[:SAMPLE_PREVIEW_TOKENS]
        preview = _decode_tokens(tokenizer, preview_ids)
        print(f"\n--- sample {i} ---")
        print(f"  tokens: {n_total} total, {len(row.weights)} trained, 0 masked")
        print(f"  first trained text: {preview[:SAMPLE_PREVIEW_CHARS]!r}")
    print(f"\n{'=' * 60}\n")


def build_training_arrays(
    documents: list[dict[str, Any]],
    tokenizer,
    config: SDFDataConfig,
    *,
    print_samples: int = DEFAULT_PRINT_SAMPLES,
) -> list[TrainingArrays]:
    """Convert synthetic-document fine-tuning rows to plain next-token arrays."""
    _validate_config(config)
    if not documents:
        raise ValueError("no synthetic-document fine-tuning rows")
    rows = []
    for i, document in enumerate(documents):
        try:
            rows.append(document_to_training_arrays(document, tokenizer, config))
        except ValueError as e:
            raise ValueError(f"row {i}: {e}") from e

    if print_samples > 0 and rows:
        _print_samples(rows, tokenizer, print_samples)
    return rows
