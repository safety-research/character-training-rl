"""Strict JSONL reading and append-and-flush writing, in one place.

The thing that matters is whether a malformed or missing file is an error or an empty list.
Silently returning `[]` for a file that failed to write is how a run reports "0 rows" instead of
failing, so the choice is explicit here.

Two read contracts, both explicit, because resume loops legitimately need the second:
`read_jsonl` raises on a missing or empty file; `read_jsonl_missing_ok` returns `[]`.

The writer appends and flushes per row, so an interrupted generation leaves a usable dataset on
disk rather than nothing.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Iterator


def _reject_duplicate_keys(path: Path, line_no: int):
    def hook(pairs):
        row: dict[str, Any] = {}
        for key, value in pairs:
            if key in row:
                raise ValueError(f"{path}:{line_no}: duplicate JSON key {key!r}")
            row[key] = value
        return row

    return hook


def iter_jsonl_numbered(path: str | Path) -> Iterator[tuple[int, dict]]:
    """Stream (line_number, row), validating each. Raises on blank lines, bad JSON, duplicate keys,
    or non-object rows. The line number is for callers that validate fields and want to say where."""
    path = Path(path)
    with path.open() as fh:
        for line_no, raw in enumerate(fh, start=1):
            if not raw.strip():
                raise ValueError(f"{path}:{line_no}: blank lines are not allowed")
            try:
                row = json.loads(raw, object_pairs_hook=_reject_duplicate_keys(path, line_no))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: row must be a JSON object")
            yield line_no, row


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    """Stream rows, validating each. Raises on blank lines, bad JSON, or non-object rows."""
    for _line_no, row in iter_jsonl_numbered(path):
        yield row


def read_jsonl_numbered(path: str | Path) -> list[tuple[int, dict]]:
    """Every (line_number, row). Raises if the file is missing or holds no rows."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path}: no such file")
    rows = list(iter_jsonl_numbered(path))
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def read_jsonl(path: str | Path) -> list[dict]:
    """Every row. Raises if the file is missing or holds no rows."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path}: no such file")
    rows = list(iter_jsonl(path))
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def read_jsonl_missing_ok(path: str | Path) -> list[dict]:
    """Every row, or `[]` when the file does not exist yet. For resume loops only."""
    path = Path(path)
    if not path.exists():
        return []
    return list(iter_jsonl(path))


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> Path:
    """Replace `path` with these rows, atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    tmp.replace(path)
    return path


def append_jsonl(path: str | Path, row: dict) -> None:
    """One row, flushed to the OS before returning. Survives a kill on the next line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(row) + "\n")
        fh.flush()


class JsonlWriter:
    """Append rows to an open handle, flushing each one.

    Cheaper than `append_jsonl` in a loop (one open instead of N) with the same guarantee: at
    every instant the file on disk is a complete, readable dataset of what has finished.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = None
        self.n = 0

    def __enter__(self) -> "JsonlWriter":
        self._fh = self.path.open("a")
        return self

    def write(self, row: dict) -> None:
        self._fh.write(json.dumps(row) + "\n")
        self._fh.flush()
        self.n += 1

    def __exit__(self, *exc) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def require_unique(rows: list[dict], key: str) -> None:
    """Refuse a corpus with duplicate ids, naming the first collision."""
    seen: dict[Any, int] = {}
    for i, row in enumerate(rows):
        if key not in row:
            raise KeyError(f"row {i} has no {key!r}")
        value = row[key]
        if value in seen:
            raise ValueError(f"duplicate {key}={value!r} at rows {seen[value]} and {i}")
        seen[value] = i


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_id(*parts: Any, length: int = 12) -> str:
    """A short deterministic id for a content tuple. Used for cache keys and row ids."""
    payload = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:length]
