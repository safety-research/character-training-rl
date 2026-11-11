"""Tests for shared next-token training array validation."""

import pytest

pytest.importorskip(
    "torch", reason="needs the training extra: uv sync --extra dev --extra training"
)

from src.training.arrays import (
    TrainingArrays,
    load_training_arrays_jsonl,
    validate_training_arrays,
    values_to_list,
)


def test_validate_training_arrays_accepts_normal_row():
    arrays = TrainingArrays(
        input_tokens=[1, 2],
        target_tokens=[2, 3],
        weights=[0.0, 1.0],
        rendered_text="abc",
    )

    assert validate_training_arrays(arrays) == arrays


def test_validate_training_arrays_rejects_length_mismatch():
    arrays = TrainingArrays(
        input_tokens=[1, 2],
        target_tokens=[2],
        weights=[1.0],
        rendered_text="abc",
    )

    with pytest.raises(ValueError, match="length mismatch"):
        validate_training_arrays(arrays)


def test_validate_training_arrays_rejects_nonfinite_weights():
    arrays = TrainingArrays(
        input_tokens=[1],
        target_tokens=[2],
        weights=[float("nan")],
        rendered_text="ab",
    )

    with pytest.raises(ValueError, match="finite"):
        validate_training_arrays(arrays)


def test_validate_training_arrays_rejects_negative_tokens():
    arrays = TrainingArrays(
        input_tokens=[-1],
        target_tokens=[2],
        weights=[1.0],
        rendered_text="ab",
    )

    with pytest.raises(ValueError, match="nonnegative"):
        validate_training_arrays(arrays)


def test_validate_training_arrays_rejects_negative_weights():
    arrays = TrainingArrays(
        input_tokens=[1],
        target_tokens=[2],
        weights=[-1.0],
        rendered_text="ab",
    )

    with pytest.raises(ValueError, match="nonnegative"):
        validate_training_arrays(arrays)


def test_validate_training_arrays_rejects_all_zero_weights():
    arrays = TrainingArrays(
        input_tokens=[1, 2],
        target_tokens=[2, 3],
        weights=[0.0, 0.0],
        rendered_text="abc",
    )

    with pytest.raises(ValueError, match="no positive"):
        validate_training_arrays(arrays)


def test_values_to_list_rejects_unknown_iterable():
    class UnknownIterable:
        def __iter__(self):
            yield 1

    with pytest.raises(TypeError, match="unsupported"):
        values_to_list(UnknownIterable())


def test_values_to_list_rejects_non_flat_tinker_tensor_data():
    from tinker import types

    value = types.TensorData(data=[1, 2], dtype="int64", shape=[1, 2])

    with pytest.raises(ValueError, match="shape"):
        values_to_list(value)


def test_load_training_arrays_jsonl_round_trip(tmp_path):
    path = tmp_path / "arrays.jsonl"
    path.write_text(
        '{"input_tokens":[1,2],"target_tokens":[2,3],"weights":[0,1],"rendered_text":"x"}\n'
    )

    rows = load_training_arrays_jsonl(path)

    assert rows == [TrainingArrays([1, 2], [2, 3], [0.0, 1.0], "x")]


def test_load_training_arrays_jsonl_reports_line(tmp_path):
    path = tmp_path / "arrays.jsonl"
    path.write_text('{"input_tokens":[1]}\n')

    with pytest.raises(ValueError, match=r"arrays.jsonl:1"):
        load_training_arrays_jsonl(path)
