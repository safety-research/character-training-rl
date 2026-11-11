"""src.jsonl: strictness on read, durability on write."""

import json

import pytest

from src.jsonl import (
    JsonlWriter,
    append_jsonl,
    read_jsonl,
    read_jsonl_missing_ok,
    require_unique,
    sha256_file,
    stable_id,
    write_jsonl,
)


def test_roundtrip(tmp_path):
    rows = [{"a": 1}, {"a": 2}]
    path = write_jsonl(tmp_path / "x.jsonl", rows)
    assert read_jsonl(path) == rows


def test_missing_file_raises_but_missing_ok_returns_empty(tmp_path):
    missing = tmp_path / "nope.jsonl"
    with pytest.raises(FileNotFoundError):
        read_jsonl(missing)
    assert read_jsonl_missing_ok(missing) == []


def test_empty_file_is_an_error_not_zero_rows(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    with pytest.raises(ValueError, match="no rows"):
        read_jsonl(path)


@pytest.mark.parametrize(
    "content,match",
    [
        ('{"a": 1}\n\n{"a": 2}\n', "blank lines"),
        ('{"a": 1}\nnot json\n', "invalid JSON"),
        ('{"a": 1}\n[1, 2]\n', "must be a JSON object"),
        ('{"a": 1, "a": 2}\n', "duplicate JSON key"),
    ],
)
def test_malformed_rows_fail_loudly_with_a_line_number(tmp_path, content, match):
    path = tmp_path / "bad.jsonl"
    path.write_text(content)
    with pytest.raises(ValueError, match=match) as exc:
        read_jsonl(path)
    assert "bad.jsonl:" in str(exc.value)


def test_append_flushes_so_an_interrupted_run_leaves_a_readable_dataset(tmp_path):
    path = tmp_path / "stream.jsonl"
    for i in range(3):
        append_jsonl(path, {"i": i})
        # Read it back through a separate handle after every row: what is on disk right now
        # must already be a complete dataset of what has finished.
        assert read_jsonl(path) == [{"i": j} for j in range(i + 1)]


def test_writer_appends_and_flushes_each_row(tmp_path):
    path = tmp_path / "w.jsonl"
    with JsonlWriter(path) as w:
        w.write({"i": 0})
        assert read_jsonl(path) == [{"i": 0}]
        w.write({"i": 1})
        assert read_jsonl(path) == [{"i": 0}, {"i": 1}]
        assert w.n == 2


def test_write_jsonl_is_atomic_and_leaves_no_temp_file(tmp_path):
    path = write_jsonl(tmp_path / "a.jsonl", [{"x": 1}])
    assert list(tmp_path.iterdir()) == [path]


def test_require_unique_names_the_collision():
    require_unique([{"id": "a"}, {"id": "b"}], "id")
    with pytest.raises(ValueError, match="duplicate id='a' at rows 0 and 2"):
        require_unique([{"id": "a"}, {"id": "b"}, {"id": "a"}], "id")
    with pytest.raises(KeyError):
        require_unique([{"other": 1}], "id")


def test_stable_id_is_deterministic_and_order_sensitive():
    assert stable_id("a", 1) == stable_id("a", 1)
    assert stable_id("a", 1) != stable_id(1, "a")
    assert len(stable_id("a", length=8)) == 8


def test_sha256_file_matches_hashlib(tmp_path):
    import hashlib

    path = tmp_path / "f.bin"
    payload = json.dumps({"k": "v"}).encode()
    path.write_bytes(payload)
    assert sha256_file(path) == hashlib.sha256(payload).hexdigest()


def test_numbered_reader_reports_line_numbers_and_shares_the_strict_rules(tmp_path):
    from src.jsonl import iter_jsonl, read_jsonl_numbered

    p = tmp_path / "rows.jsonl"
    p.write_text('{"a": 1}\n{"a": 2}\n')
    assert read_jsonl_numbered(p) == [(1, {"a": 1}), (2, {"a": 2})]
    assert list(iter_jsonl(p)) == [{"a": 1}, {"a": 2}]
    p.write_text('{"a": 1}\n\n{"a": 2}\n')
    with pytest.raises(ValueError, match=":2: blank lines"):
        read_jsonl_numbered(p)
