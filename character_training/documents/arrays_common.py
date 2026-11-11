"""Strict data construction and receipt helpers for the document-stage training arrays."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from src.costs import TINKER_TRAIN_USD_PER_MTOK
from src.training.arrays import TrainingArrays
from src.training.sdf_data import SDFDataConfig
from src.training.sdf_data import build_training_arrays as build_sdf_arrays

END_OF_TEXT_ID = 199999
RETURN_ID = 200002


@dataclass(frozen=True)
class PlannedRow:
    source: str
    source_index: int
    row_id: str
    input_tokens: int
    trained_tokens: int


def strict_bool(value: str) -> bool:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    raise ValueError(f"expected true or false, got {value!r}")


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    if not rows:
        raise ValueError(f"{path} has no JSONL rows")
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    path.write_text(payload)


def load_tokenizer(model: str):
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    return get_tokenizer(model)


def _row_id(row: dict[str, Any], index: int, prefix: str) -> str:
    value = row.get("id", row.get("manifest_id", f"{prefix}-{index:06d}"))
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{prefix} row {index}: id must be a non-empty string")
    return value


def build_document_arrays(
    rows: list[dict[str, Any]], tokenizer, max_seq_length: int
) -> tuple[list[TrainingArrays], list[str]]:
    ids: list[str] = []
    forbidden = set(tokenizer.all_special_ids)
    for index, row in enumerate(rows):
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"document row {index}: text must be a non-empty string")
        row_id = _row_id(row, index, "document")
        body = tokenizer.encode(text, add_special_tokens=False)
        leaked = sorted(set(body) & forbidden)
        if leaked:
            raise ValueError(f"document {row_id}: body contains special token IDs {leaked}")
        ids.append(row_id)
    arrays = build_sdf_arrays(
        rows,
        tokenizer,
        SDFDataConfig(
            max_seq_length=max_seq_length,
            append_eos=False,
            terminal_token="<|endoftext|>",
        ),
        print_samples=0,
    )
    for row_id, arr in zip(ids, arrays, strict=True):
        if arr.target_tokens[-1] != END_OF_TEXT_ID or RETURN_ID in arr.target_tokens:
            raise AssertionError(f"document {row_id}: invalid terminal-token construction")
    if len(ids) != len(set(ids)):
        raise ValueError("document row IDs are not unique")
    return arrays, ids


def trained_tokens(arrays: TrainingArrays) -> int:
    return sum(1 for weight in arrays.weights if weight > 0)


def arrays_digest(arrays: Iterable[TrainingArrays]) -> str:
    digest = hashlib.sha256()
    for row in arrays:
        digest.update(
            canonical_json(
                {
                    "input_tokens": row.input_tokens,
                    "target_tokens": row.target_tokens,
                    "weights": row.weights,
                }
            ).encode()
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _select_unique(
    source: str,
    arrays: list[TrainingArrays],
    ids: list[str],
    token_quota: int,
    seed: int,
) -> list[tuple[PlannedRow, TrainingArrays]]:
    if token_quota <= 0:
        return []
    indices = list(range(len(arrays)))
    random.Random(f"{seed}:{source}").shuffle(indices)
    selected: list[tuple[PlannedRow, TrainingArrays]] = []
    total = 0
    for index in indices:
        row = arrays[index]
        n_trained = trained_tokens(row)
        planned = PlannedRow(source, index, ids[index], len(row.input_tokens), n_trained)
        selected.append((planned, row))
        total += n_trained
        if total >= token_quota:
            return selected
    raise ValueError(f"{source} has {total:,} unique trained tokens, below quota {token_quota:,}")


def make_document_plan(
    *,
    document_arrays: list[TrainingArrays],
    document_ids: list[str],
    target_trained_tokens: int,
    seed: int,
    batch_size: int,
) -> tuple[list[PlannedRow], list[TrainingArrays]]:
    """Select one unique raw-document pass and complete its final fixed-size batch."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    documents = _select_unique("document", document_arrays, document_ids, target_trained_tokens, seed)
    selected_ids = {item[0].row_id for item in documents}
    indices = list(range(len(document_arrays)))
    random.Random(f"{seed}:document").shuffle(indices)
    while len(documents) % batch_size:
        available = next((i for i in indices if document_ids[i] not in selected_ids), None)
        if available is None:
            raise ValueError("not enough unique documents to complete the final fixed-size batch")
        row = document_arrays[available]
        documents.append(
            (
                PlannedRow(
                    "document",
                    available,
                    document_ids[available],
                    len(row.input_tokens),
                    trained_tokens(row),
                ),
                row,
            )
        )
        selected_ids.add(document_ids[available])
    return [item[0] for item in documents], [item[1] for item in documents]


def document_plan_summary(rows: list[PlannedRow], batch_size: int, model: str) -> dict[str, Any]:
    """Counts of the planned rows and the training cost of their trained tokens at `model`'s price in
    src.costs."""
    input_tokens = sum(row.input_tokens for row in rows)
    trained = sum(row.trained_tokens for row in rows)
    return {
        "rows": len(rows),
        "batches": len(rows) // batch_size,
        "input_tokens": input_tokens,
        "trained_tokens": trained,
        "document_trained_tokens": trained,
        "estimated_train_usd": trained / 1_000_000 * TINKER_TRAIN_USD_PER_MTOK[model],
    }


def checkpoint_plan(
    arrays: list[TrainingArrays],
    *,
    batch_size: int,
    seed: int,
    target_tokens: list[int],
) -> list[dict[str, int]]:
    """Map token targets to the first exact optimizer-step crossing.

    This reproduces the shared trainer's seed shuffle and fixed-size one-pass
    batching. The resulting optimizer steps, realized token counts, and
    overshoots are frozen into the preflight receipt before any Tinker call.
    """
    if len(arrays) % batch_size:
        raise ValueError("checkpoint planning requires a complete fixed-size final batch")
    if not target_tokens or target_tokens != sorted(set(target_tokens)) or any(t <= 0 for t in target_tokens):
        raise ValueError("checkpoint token targets must be unique, positive, and ascending")
    order = list(range(len(arrays)))
    random.Random(seed).shuffle(order)
    targets = iter(target_tokens)
    current_target = next(targets, None)
    result: list[dict[str, int]] = []
    cumulative = 0
    for step in range(1, len(arrays) // batch_size + 1):
        start = (step - 1) * batch_size
        for position in order[start : start + batch_size]:
            cumulative += trained_tokens(arrays[position])
        while current_target is not None and cumulative >= current_target:
            result.append(
                {
                    "target_trained_tokens": current_target,
                    "step": step,
                    "realized_trained_tokens": cumulative,
                    "overshoot": cumulative - current_target,
                }
            )
            current_target = next(targets, None)
    if current_target is not None:
        raise ValueError(f"training plan reaches {cumulative:,} tokens, below checkpoint target {current_target:,}")
    return result


def rows_as_dicts(rows: list[PlannedRow]) -> list[dict[str, Any]]:
    return [asdict(row) for row in rows]


def write_training_arrays_jsonl(path: Path, planned_rows: list[PlannedRow], arrays: list[TrainingArrays]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for planned, row in zip(planned_rows, arrays, strict=True):
            handle.write(
                canonical_json(
                    {
                        "id": planned.row_id,
                        "source": planned.source,
                        "input_tokens": row.input_tokens,
                        "target_tokens": row.target_tokens,
                        "weights": row.weights,
                    }
                )
                + "\n"
            )
