"""Helpers for building a training mixture: converting kept constitution-question answers to
training rows, the trainer's row order, and render statistics.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.costs import TINKER_TRAIN_USD_PER_MTOK
from src.training.sft_data import SFTDataConfig, example_to_training_arrays

# A fixed label used in cache keys and data rows; the name has no meaning beyond identifying this version.
COMPONENT_TREATMENT = "treatment"


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    rows = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row must be a JSON object")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_strings(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]], overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite true to replace it")
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n")
        handle.flush()
    temporary.replace(path)


def require_nonempty_string(row: dict[str, Any], key: str, context: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}: {key} must be a non-empty string")
    return value


def treatment_to_sft(
    row: dict[str, Any],
    *,
    arm: str,
    required_renderer: str,
    required_current_date: str,
) -> dict[str, Any]:
    context = f"{arm}:{row.get('question_id', '<missing>')}"
    if row.get("constitution") != arm:
        raise ValueError(f"{context}: constitution={row.get('constitution')!r}, expected {arm!r}")
    if row.get("renderer") != required_renderer:
        raise ValueError(f"{context}: renderer={row.get('renderer')!r}, expected {required_renderer!r}")
    if row.get("current_date") != required_current_date:
        raise ValueError(f"{context}: current_date={row.get('current_date')!r}, expected {required_current_date!r}")
    if row.get("structurally_valid") is not True or row.get("stop_reason") != "stop":
        raise ValueError(f"{context}: treatment row is not a clean stopped Harmony response")

    question_id = require_nonempty_string(row, "question_id", context)
    question = require_nonempty_string(row, "question", context)
    analysis = require_nonempty_string(row, "analysis", context)
    final = require_nonempty_string(row, "final", context)
    trial_id = require_nonempty_string(row, "trial_id", context)
    return {
        "id": f"constqa:{question_id}:{arm}",
        "source": f"constqa_{arm}",
        "mix_component": COMPONENT_TREATMENT,
        "match_key": f"treatment:{question_id}",
        "question_id": question_id,
        "generation_trial_id": trial_id,
        "generation_renderer": required_renderer,
        "generation_current_date": required_current_date,
        "messages": [
            {"role": "user", "content": question},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": analysis},
                    {"type": "text", "text": final},
                ],
            },
        ],
        "thinking": "keep",
    }


def prepare_treatment(
    rows: list[dict[str, Any]],
    *,
    arm: str,
    required_count: int,
    required_renderer: str,
    required_current_date: str,
) -> list[dict[str, Any]]:
    if len(rows) != required_count:
        raise ValueError(f"{arm}: got {len(rows)} treatment rows, expected {required_count}")
    converted = [
        treatment_to_sft(
            row,
            arm=arm,
            required_renderer=required_renderer,
            required_current_date=required_current_date,
        )
        for row in rows
    ]
    question_ids = [row["question_id"] for row in converted]
    if len(set(question_ids)) != len(question_ids):
        duplicates = [key for key, count in Counter(question_ids).items() if count > 1]
        raise ValueError(f"{arm}: duplicate question IDs: {duplicates[:5]}")
    return converted


def training_order(n_rows: int, *, seed: int, batch_size: int, max_steps: int) -> list[int]:
    order = list(range(n_rows))
    rng = random.Random(seed)
    rng.shuffle(order)
    used = []
    index = 0
    for _ in range(max_steps * batch_size):
        if index >= len(order):
            rng.shuffle(order)
            index = 0
        used.append(order[index])
        index += 1
    return used


def percentile(values: list[int], quantile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def render_stats(
    rows: list[dict[str, Any]],
    *,
    label: str,
    model: str,
    renderer,
    config: SFTDataConfig,
    training_indices: list[int],
) -> dict[str, Any]:
    total_lengths = []
    trained_lengths = []
    by_component: dict[str, dict[str, Any]] = defaultdict(lambda: {"rows": 0, "total_tokens": 0, "trained_tokens": 0})
    for index, row in enumerate(rows):
        try:
            arrays = example_to_training_arrays(row, renderer, config)
        except ValueError as error:
            raise ValueError(f"render validation row {index} ({row.get('match_key')}): {error}") from error
        total = len(arrays.input_tokens) + 1
        trained = sum(weight > 0 for weight in arrays.weights)
        total_lengths.append(total)
        trained_lengths.append(trained)
        component = row["mix_component"]
        by_component[component]["rows"] += 1
        by_component[component]["total_tokens"] += total
        by_component[component]["trained_tokens"] += trained
        if (index + 1) % 1000 == 0 or index + 1 == len(rows):
            print(f"Rendered {label}: {index + 1:,}/{len(rows):,}", flush=True)

    exact_train_tokens = sum(total_lengths[index] for index in training_indices)
    repeated = training_indices[len(rows) :]
    return {
        "rows": len(rows),
        "total_tokens": sum(total_lengths),
        "trained_tokens": sum(trained_lengths),
        "mean_total_tokens": sum(total_lengths) / len(total_lengths),
        "mean_trained_tokens": sum(trained_lengths) / len(trained_lengths),
        "p50_total_tokens": percentile(total_lengths, 0.50),
        "p90_total_tokens": percentile(total_lengths, 0.90),
        "p99_total_tokens": percentile(total_lengths, 0.99),
        "max_total_tokens": max(total_lengths),
        "by_component": dict(by_component),
        "exact_training_row_uses": len(training_indices),
        "exact_train_tokens": exact_train_tokens,
        "estimated_train_usd": exact_train_tokens * TINKER_TRAIN_USD_PER_MTOK[model] / 1_000_000,
        "repeated_after_first_epoch": [
            {
                "mix_index": index,
                "match_key": rows[index]["match_key"],
                "component": rows[index]["mix_component"],
                "total_tokens": total_lengths[index],
            }
            for index in repeated
        ],
    }
