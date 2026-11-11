"""Tests for synthetic-document fine-tuning data conversion."""

import json
from collections import UserDict

import pytest

from src.training.sdf_data import (
    SDFDataConfig,
    build_training_arrays,
    document_to_training_arrays,
    load_document_jsonl,
)


class FakeTokenizer:
    eos_token_id = 3

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, token_ids):
        chars = []
        for token_id in token_ids:
            chars.append("<eos>" if token_id == self.eos_token_id else chr(token_id))
        return "".join(chars)


class CallableTokenizer:
    def __call__(self, text, truncation, padding, add_special_tokens, return_tensors):
        assert truncation is False
        assert padding is False
        assert add_special_tokens is False
        assert return_tensors is None
        return UserDict({"input_ids": [ord(c) for c in text]})

    def decode(self, token_ids):
        return "".join(chr(i) for i in token_ids)


def test_load_document_jsonl_requires_text(tmp_path):
    path = tmp_path / "sdf.jsonl"
    path.write_text(json.dumps({"content": "missing text"}) + "\n")

    with pytest.raises(ValueError, match="text"):
        load_document_jsonl(path)


def test_document_to_training_arrays_trains_every_next_token():
    tokenizer = FakeTokenizer()
    config = SDFDataConfig(max_seq_length=16, append_eos=False)

    arrays = document_to_training_arrays({"text": "abcd"}, tokenizer, config)

    assert arrays.input_tokens == [ord("a"), ord("b"), ord("c")]
    assert arrays.target_tokens == [ord("b"), ord("c"), ord("d")]
    assert arrays.weights == [1.0, 1.0, 1.0]
    assert arrays.rendered_text == "abcd"


def test_callable_tokenizer_mapping_is_supported():
    tokenizer = CallableTokenizer()
    config = SDFDataConfig(max_seq_length=16, append_eos=False)

    arrays = document_to_training_arrays({"text": "abcd"}, tokenizer, config)

    assert arrays.target_tokens == [ord("b"), ord("c"), ord("d")]


def test_callable_tokenizer_rejects_nested_input_ids():
    class NestedTokenizer:
        eos_token_id = 3

        def __call__(self, *args, **kwargs):
            return {"input_ids": [[1, 2]]}

        def decode(self, token_ids):
            return str(token_ids)

    config = SDFDataConfig(max_seq_length=16, append_eos=False)

    with pytest.raises(ValueError, match="int token id"):
        document_to_training_arrays({"text": "abcd"}, NestedTokenizer(), config)


def test_append_eos_adds_stop_token_once():
    tokenizer = FakeTokenizer()
    config = SDFDataConfig(max_seq_length=16, append_eos=True)

    arrays = document_to_training_arrays({"text": "ab"}, tokenizer, config)

    assert arrays.target_tokens[-1] == tokenizer.eos_token_id
    assert arrays.rendered_text.endswith("<eos>")


def test_append_eos_rejects_tokenizer_without_eos_id():
    class NoEOSTokenizer(FakeTokenizer):
        eos_token_id = None

    config = SDFDataConfig(max_seq_length=16, append_eos=True)

    with pytest.raises(ValueError, match="eos_token_id"):
        document_to_training_arrays({"text": "ab"}, NoEOSTokenizer(), config)


def test_named_terminal_token_is_distinct_from_eos():
    class GPTOSSLikeTokenizer(FakeTokenizer):
        eos_token_id = 200002

        def encode(self, text, add_special_tokens=False):
            if text == "<|endoftext|>":
                return [199999]
            return super().encode(text, add_special_tokens=add_special_tokens)

        def decode(self, token_ids):
            return "".join(
                "<|endoftext|>" if token_id == 199999 else "<|return|>" if token_id == 200002 else chr(token_id)
                for token_id in token_ids
            )

    arrays = document_to_training_arrays(
        {"text": "ab"},
        GPTOSSLikeTokenizer(),
        SDFDataConfig(max_seq_length=16, append_eos=False, terminal_token="<|endoftext|>"),
    )

    assert arrays.target_tokens[-1] == 199999
    assert 200002 not in arrays.target_tokens
    assert arrays.rendered_text.endswith("<|endoftext|>")


def test_named_terminal_token_and_append_eos_are_mutually_exclusive():
    config = SDFDataConfig(max_seq_length=16, append_eos=True, terminal_token="<end>")

    with pytest.raises(ValueError, match="mutually exclusive"):
        document_to_training_arrays({"text": "ab"}, FakeTokenizer(), config)


def test_config_rejects_non_bool_append_eos():
    tokenizer = FakeTokenizer()
    config = SDFDataConfig(max_seq_length=16, append_eos="false")

    with pytest.raises(ValueError, match="append_eos"):
        document_to_training_arrays({"text": "ab"}, tokenizer, config)


def test_max_seq_length_never_truncates():
    tokenizer = FakeTokenizer()
    config = SDFDataConfig(max_seq_length=3, append_eos=True)

    with pytest.raises(ValueError, match="never truncated"):
        document_to_training_arrays({"text": "abcd"}, tokenizer, config)


def test_build_training_arrays_identifies_bad_row_index():
    tokenizer = FakeTokenizer()
    config = SDFDataConfig(max_seq_length=16, append_eos=False)

    with pytest.raises(ValueError, match="row 1"):
        build_training_arrays([{"text": "ok"}, {"text": ""}], tokenizer, config, print_samples=0)


def test_build_training_arrays_rejects_empty_documents():
    tokenizer = FakeTokenizer()
    config = SDFDataConfig(max_seq_length=16, append_eos=False)

    with pytest.raises(ValueError, match="no synthetic-document"):
        build_training_arrays([], tokenizer, config, print_samples=0)
