"""Tests for shared SFT data loading and masking."""

import json
import math

import pytest

pytest.importorskip(
    "torch", reason="needs the training extra: uv sync --extra dev --extra training"
)

from src.training.sft_data import (
    SFTDataConfig,
    build_training_arrays,
    example_to_training_arrays,
    load_chat_jsonl,
)


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, token_ids):
        return "".join(chr(i) for i in token_ids)


class FakeModelInput:
    def __init__(self, token_ids):
        self._token_ids = token_ids
        self.length = len(token_ids)

    def to_ints(self):
        return self._token_ids


class FakeRenderer:
    tokenizer = FakeTokenizer()

    def build_supervised_example(self, messages, train_on_what):
        mode = getattr(train_on_what, "value", str(train_on_what))
        tokens = []
        weights = []

        for i, message in enumerate(messages):
            is_last = i == len(messages) - 1
            is_target_assistant = mode == "last_assistant_message" and is_last and message["role"] == "assistant"
            train_all = mode == "all_tokens"
            train_prompt = mode == "all_user_and_system_messages" and message["role"] in ("system", "user")

            header = f"<|im_start|>{message['role']}\n"
            output = f"{message['content']}<|im_end|>\n"
            header_tokens = self.tokenizer.encode(header)
            output_tokens = self.tokenizer.encode(output)

            tokens.extend(header_tokens)
            weights.extend([1.0 if train_all or train_prompt else 0.0] * len(header_tokens))
            tokens.extend(output_tokens)
            weights.extend([1.0 if train_all or train_prompt or is_target_assistant else 0.0] * len(output_tokens))

        return FakeModelInput(tokens), weights


class HarmonyFakeRenderer(FakeRenderer):
    """Small character-token renderer that mirrors GPT-OSS channel structure."""

    def build_supervised_example(self, messages, train_on_what):
        mode = getattr(train_on_what, "value", str(train_on_what))
        tokens = []
        weights = []

        for i, message in enumerate(messages):
            is_last = i == len(messages) - 1
            is_target_assistant = (
                mode == "last_assistant_message"
                and is_last
                and message["role"] == "assistant"
            )
            train_all = mode == "all_tokens"
            train_prompt = (
                mode == "all_user_and_system_messages"
                and message["role"] in ("system", "user")
            )

            header = f"<|start|>{message['role']}"
            content = message["content"]
            if message["role"] == "assistant":
                parts = content if isinstance(content, list) else [{"type": "text", "text": content}]
                thinking = "".join(
                    part["thinking"] for part in parts if part["type"] == "thinking"
                )
                final = "".join(part["text"] for part in parts if part["type"] == "text")
                output = ""
                if thinking:
                    output += (
                        f"<|channel|>analysis<|message|>{thinking}"
                        "<|end|><|start|>assistant"
                    )
                output += f"<|channel|>final<|message|>{final}<|return|>"
            else:
                output = f"<|message|>{content}<|end|>"

            header_tokens = self.tokenizer.encode(header)
            output_tokens = self.tokenizer.encode(output)
            tokens.extend(header_tokens)
            weights.extend([1.0 if train_all or train_prompt else 0.0] * len(header_tokens))
            tokens.extend(output_tokens)
            weights.extend(
                [1.0 if train_all or train_prompt or is_target_assistant else 0.0]
                * len(output_tokens)
            )

        return FakeModelInput(tokens), weights


class BadWeightRenderer(FakeRenderer):
    def build_supervised_example(self, messages, train_on_what):
        model_input, weights = super().build_supervised_example(messages, train_on_what)
        weights[-1] = math.nan
        return model_input, weights


class UnmaskedRenderer(FakeRenderer):
    def build_supervised_example(self, messages, train_on_what):
        model_input, weights = super().build_supervised_example(messages, train_on_what)
        return model_input, [1.0] * len(weights)


def chat_row(assistant_content):
    return {
        "messages": [
            {"role": "system", "content": "System rule"},
            {"role": "user", "content": "User question"},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def trained_text(arrays, tokenizer):
    token_ids = [tok for tok, weight in zip(arrays.target_tokens, arrays.weights) if weight > 0]
    return tokenizer.decode(token_ids)


def test_load_chat_jsonl_requires_messages(tmp_path):
    path = tmp_path / "sft.jsonl"
    path.write_text(json.dumps({"prompt": "missing messages"}) + "\n")

    with pytest.raises(ValueError, match="messages"):
        load_chat_jsonl(path)


def test_messages_reject_extra_keys():
    renderer = FakeRenderer()
    config = SFTDataConfig(max_seq_length=4096, mask_prompt=True, thinking="keep")
    row = chat_row("Assistant answer")
    row["messages"][-1]["tool_calls"] = []

    with pytest.raises(ValueError, match="unsupported key"):
        example_to_training_arrays(row, renderer, config)


def test_prompt_masking_trains_assistant_not_prompt():
    renderer = FakeRenderer()
    config = SFTDataConfig(max_seq_length=4096, mask_prompt=True, thinking="keep")

    arrays = example_to_training_arrays(chat_row("Assistant answer"), renderer, config)
    text = trained_text(arrays, renderer.tokenizer)

    assert "System rule" not in text
    assert "User question" not in text
    assert "Assistant answer" in text


def test_gptoss_structured_reasoning_and_final_are_trained():
    renderer = HarmonyFakeRenderer()
    config = SFTDataConfig(max_seq_length=4096, mask_prompt=True, thinking="keep")
    row = chat_row(
        [
            {"type": "thinking", "thinking": "private reasoning"},
            {"type": "text", "text": "Final answer"},
        ]
    )

    arrays = example_to_training_arrays(row, renderer, config)
    text = trained_text(arrays, renderer.tokenizer)

    assert "<|channel|>analysis<|message|>private reasoning" in text
    assert "<|channel|>final<|message|>Final answer<|return|>" in text
    assert "User question" not in text


def test_gptoss_final_only_string_trains_a_final_channel():
    renderer = HarmonyFakeRenderer()
    config = SFTDataConfig(max_seq_length=4096, mask_prompt=True, thinking="keep")

    arrays = example_to_training_arrays(chat_row("Final answer only"), renderer, config)
    text = trained_text(arrays, renderer.tokenizer)

    assert "<|channel|>analysis" not in arrays.rendered_text
    assert "<|channel|>final<|message|>Final answer only<|return|>" in text


def test_gptoss_dataset_can_mix_rows_with_and_without_private_reasoning():
    renderer = HarmonyFakeRenderer()
    config = SFTDataConfig(max_seq_length=4096, mask_prompt=True, thinking="keep")
    rows = [
        chat_row(
            [
                {"type": "thinking", "thinking": "private reasoning"},
                {"type": "text", "text": "Reasoned answer"},
            ]
        ),
        chat_row("Direct answer"),
    ]

    arrays = build_training_arrays(rows, renderer, config, print_samples=0)

    assert "<|channel|>analysis" in arrays[0].rendered_text
    assert "<|channel|>analysis" not in arrays[1].rendered_text


def test_gptoss_openai_reasoning_content_is_normalized():
    renderer = HarmonyFakeRenderer()
    config = SFTDataConfig(max_seq_length=4096, mask_prompt=True, thinking="keep")
    row = chat_row("Final answer")
    row["messages"][-1]["reasoning_content"] = "private reasoning"

    arrays = example_to_training_arrays(row, renderer, config)
    text = trained_text(arrays, renderer.tokenizer)

    assert "<|channel|>analysis<|message|>private reasoning" in text
    assert "<|channel|>final<|message|>Final answer" in text


def test_gptoss_mask_keeps_analysis_tokens_but_trains_only_final():
    renderer = HarmonyFakeRenderer()
    config = SFTDataConfig(max_seq_length=4096, mask_prompt=True, thinking="mask")
    row = chat_row(
        [
            {"type": "thinking", "thinking": "private reasoning"},
            {"type": "text", "text": "Final answer"},
        ]
    )

    arrays = example_to_training_arrays(row, renderer, config)
    text = trained_text(arrays, renderer.tokenizer)

    assert "private reasoning" in arrays.rendered_text
    assert "private reasoning" not in text
    assert "<|channel|>analysis" not in text
    assert "<|channel|>final<|message|>Final answer<|return|>" in text


def test_gptoss_strip_deletes_analysis_channel():
    renderer = HarmonyFakeRenderer()
    config = SFTDataConfig(max_seq_length=4096, mask_prompt=True, thinking="strip")
    row = chat_row(
        [
            {"type": "thinking", "thinking": "private reasoning"},
            {"type": "text", "text": "Final answer"},
        ]
    )

    arrays = example_to_training_arrays(row, renderer, config)

    assert "<|channel|>analysis" not in arrays.rendered_text
    assert "Final answer" in trained_text(arrays, renderer.tokenizer)


@pytest.mark.parametrize(
    "content, error",
    [
        ([{"type": "thinking", "thinking": "reasoning"}], "text part"),
        (
            [
                {"type": "text", "text": "answer"},
                {"type": "thinking", "thinking": "late reasoning"},
            ],
            "must precede",
        ),
        ([{"type": "image", "url": "x"}], "unsupported type"),
    ],
)
def test_gptoss_structured_content_is_strict(content, error):
    renderer = HarmonyFakeRenderer()
    config = SFTDataConfig(max_seq_length=4096, mask_prompt=True, thinking="keep")

    with pytest.raises(ValueError, match=error):
        example_to_training_arrays(chat_row(content), renderer, config)


ASSISTANT_WITH_THINKING = "<think>\nhidden reasoning\n</think>\nFinal answer"


def test_mask_masks_reasoning_but_keeps_it_in_the_sequence():
    # mask must NOT delete: the think tokens stay in the sequence at weight 0.
    renderer = FakeRenderer()
    config = SFTDataConfig(max_seq_length=4096, mask_prompt=True, thinking="mask")

    arrays = example_to_training_arrays(chat_row(ASSISTANT_WITH_THINKING), renderer, config)
    text = trained_text(arrays, renderer.tokenizer)

    assert "hidden reasoning" not in text  # reasoning not supervised
    assert "<think>" not in text and "</think>" not in text  # both tags masked (no keep-open)
    assert "Final answer" in text  # answer trained
    assert "hidden reasoning" in arrays.rendered_text  # but still present in the token sequence
    assert "<think>" in arrays.rendered_text


def test_mask_empty_think_block_keeps_both_tags_but_trains_neither() -> None:
    renderer = FakeRenderer()
    config = SFTDataConfig(max_seq_length=4096, mask_prompt=True, thinking="mask")

    arrays = example_to_training_arrays(
        chat_row("<think>\n\n</think>\n\nFinal answer"), renderer, config
    )
    text = trained_text(arrays, renderer.tokenizer)

    assert "<think>" in arrays.rendered_text and "</think>" in arrays.rendered_text
    assert "<think>" not in text and "</think>" not in text
    assert "Final answer" in text


def test_keep_trains_trace_and_final_answer():
    renderer = FakeRenderer()
    config = SFTDataConfig(max_seq_length=4096, mask_prompt=True, thinking="keep")

    arrays = example_to_training_arrays(chat_row(ASSISTANT_WITH_THINKING), renderer, config)
    text = trained_text(arrays, renderer.tokenizer)

    assert "hidden reasoning" in text
    assert "<think>" in text and "</think>" in text
    assert "Final answer" in text


def test_mask_composes_with_prompt_unmasked():
    renderer = FakeRenderer()
    config = SFTDataConfig(max_seq_length=4096, mask_prompt=False, thinking="mask")

    arrays = example_to_training_arrays(chat_row(ASSISTANT_WITH_THINKING), renderer, config)
    text = trained_text(arrays, renderer.tokenizer)

    assert "System rule" in text  # prompt trained (mask_prompt=False)
    assert "User question" in text
    assert "hidden reasoning" not in text  # reasoning still masked
    assert "Final answer" in text


def test_mask_no_think_block_is_noop():
    renderer = FakeRenderer()
    config = SFTDataConfig(max_seq_length=4096, mask_prompt=True, thinking="mask")

    arrays = example_to_training_arrays(chat_row("Final answer only"), renderer, config)
    assert "Final answer only" in trained_text(arrays, renderer.tokenizer)


def test_mask_alias_thinking_tag():
    renderer = FakeRenderer()
    config = SFTDataConfig(max_seq_length=4096, mask_prompt=True, thinking="mask")

    arrays = example_to_training_arrays(chat_row("<thinking>hidden</thinking>\nFinal answer"), renderer, config)
    text = trained_text(arrays, renderer.tokenizer)

    assert "hidden" not in text
    assert "Final answer" in text
    assert "hidden" in arrays.rendered_text


def test_mask_rejects_incomplete_block():
    renderer = FakeRenderer()
    config = SFTDataConfig(max_seq_length=4096, mask_prompt=True, thinking="mask")

    with pytest.raises(ValueError, match="incomplete"):
        example_to_training_arrays(chat_row("<think>unterminated reasoning"), renderer, config)


def test_mask_rejects_repeated_block():
    renderer = FakeRenderer()
    config = SFTDataConfig(max_seq_length=4096, mask_prompt=True, thinking="mask")

    with pytest.raises(ValueError, match="nested or repeated"):
        example_to_training_arrays(chat_row("<think>a</think>\nFinal\n<think>b</think>"), renderer, config)


def test_strip_removes_block_for_non_reasoning_target():
    renderer = FakeRenderer()
    config = SFTDataConfig(max_seq_length=4096, mask_prompt=True, thinking="strip")

    arrays = example_to_training_arrays(chat_row("<think>hidden reasoning</think>\nFinal answer"), renderer, config)
    text = trained_text(arrays, renderer.tokenizer)

    assert "Final answer" in text
    assert "hidden reasoning" not in arrays.rendered_text  # actually removed, not just masked
    assert "<think>" not in arrays.rendered_text


def test_strip_rejects_empty_answer():
    renderer = FakeRenderer()
    config = SFTDataConfig(max_seq_length=4096, mask_prompt=True, thinking="strip")

    with pytest.raises(ValueError, match="empty after removing"):
        example_to_training_arrays(chat_row("<think>only reasoning</think>"), renderer, config)


def test_strip_empty_removes_empty_placeholder():
    renderer = FakeRenderer()
    config = SFTDataConfig(max_seq_length=4096, mask_prompt=True, thinking="strip_empty")

    arrays = example_to_training_arrays(chat_row("<think>\n\n</think>\nFinal answer"), renderer, config)
    assert "<think>" not in arrays.rendered_text
    assert "Final answer" in trained_text(arrays, renderer.tokenizer)


def test_strip_empty_keeps_real_reasoning():
    renderer = FakeRenderer()
    config = SFTDataConfig(max_seq_length=4096, mask_prompt=True, thinking="strip_empty")

    arrays = example_to_training_arrays(chat_row("<think>real reasoning</think>\nFinal answer"), renderer, config)
    text = trained_text(arrays, renderer.tokenizer)

    assert "<think>" in arrays.rendered_text  # left in place
    assert "real reasoning" in text  # trained, like keep
    assert "Final answer" in text


def test_per_row_thinking_overrides_config():
    renderer = FakeRenderer()

    masked_default = SFTDataConfig(max_seq_length=4096, mask_prompt=True, thinking="mask")
    row_keep = chat_row(ASSISTANT_WITH_THINKING)
    row_keep["thinking"] = "keep"
    assert "hidden reasoning" in trained_text(
        example_to_training_arrays(row_keep, renderer, masked_default), renderer.tokenizer
    )

    keep_default = SFTDataConfig(max_seq_length=4096, mask_prompt=True, thinking="keep")
    row_mask = chat_row(ASSISTANT_WITH_THINKING)
    row_mask["thinking"] = "mask"
    assert "hidden reasoning" not in trained_text(
        example_to_training_arrays(row_mask, renderer, keep_default), renderer.tokenizer
    )


def test_invalid_row_thinking_override_raises():
    renderer = FakeRenderer()
    config = SFTDataConfig(max_seq_length=4096, mask_prompt=True, thinking="keep")
    row = chat_row(ASSISTANT_WITH_THINKING)
    row["thinking"] = "bogus"

    with pytest.raises(ValueError, match="thinking' override"):
        example_to_training_arrays(row, renderer, config)


def test_config_rejects_unknown_thinking_mode():
    renderer = FakeRenderer()
    config = SFTDataConfig(max_seq_length=4096, mask_prompt=True, thinking="delete")

    with pytest.raises(ValueError, match="unknown thinking mode"):
        example_to_training_arrays(chat_row(ASSISTANT_WITH_THINKING), renderer, config)


def test_prompt_unmasked_trains_prompt_and_assistant():
    renderer = FakeRenderer()
    config = SFTDataConfig(max_seq_length=4096, mask_prompt=False, thinking="keep")

    arrays = example_to_training_arrays(chat_row("Assistant answer"), renderer, config)
    text = trained_text(arrays, renderer.tokenizer)

    assert "System rule" in text
    assert "User question" in text
    assert "Assistant answer" in text


def test_prompt_unmasked_requires_every_token_trained():
    class AssistantOnlyRenderer(FakeRenderer):
        def build_supervised_example(self, messages, train_on_what):
            return super().build_supervised_example(messages, train_on_what="last_assistant_message")

    renderer = AssistantOnlyRenderer()
    config = SFTDataConfig(max_seq_length=4096, mask_prompt=False, thinking="keep")

    with pytest.raises(ValueError, match="mask_prompt=False"):
        example_to_training_arrays(chat_row("Assistant answer"), renderer, config)


def test_build_training_arrays_identifies_bad_row_index():
    renderer = FakeRenderer()
    config = SFTDataConfig(max_seq_length=4096, mask_prompt=True, thinking="keep")

    with pytest.raises(ValueError, match="row 1"):
        build_training_arrays([chat_row("ok"), {"messages": []}], renderer, config, print_samples=0)


def test_build_training_arrays_rejects_empty_examples():
    renderer = FakeRenderer()
    config = SFTDataConfig(max_seq_length=4096, mask_prompt=True, thinking="keep")

    with pytest.raises(ValueError, match="no chat-SFT rows"):
        build_training_arrays([], renderer, config, print_samples=0)


def test_config_rejects_non_bool_mask_prompt():
    renderer = FakeRenderer()
    config = SFTDataConfig(max_seq_length=4096, mask_prompt="false", thinking="keep")

    with pytest.raises(ValueError, match="mask_prompt"):
        example_to_training_arrays(chat_row("Assistant answer"), renderer, config)


def test_max_seq_length_never_truncates():
    renderer = FakeRenderer()
    config = SFTDataConfig(max_seq_length=10, mask_prompt=True, thinking="keep")

    with pytest.raises(ValueError, match="never truncated"):
        example_to_training_arrays(chat_row("long answer"), renderer, config)


def test_renderer_weights_must_be_finite():
    renderer = BadWeightRenderer()
    config = SFTDataConfig(max_seq_length=4096, mask_prompt=True, thinking="keep")

    with pytest.raises(ValueError, match="finite"):
        example_to_training_arrays(chat_row("Assistant answer"), renderer, config)


def test_prompt_masking_requires_leading_zero_weights():
    renderer = UnmaskedRenderer()
    config = SFTDataConfig(max_seq_length=4096, mask_prompt=True, thinking="keep")

    with pytest.raises(ValueError, match="mask_prompt=True"):
        example_to_training_arrays(chat_row("Assistant answer"), renderer, config)
