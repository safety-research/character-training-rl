"""Tests for strict preference-training data contracts."""

import pytest

pytest.importorskip("torch", reason="needs the training extra: uv sync --extra dev --extra training")

from src.training.preference_data import build_dpo_training_pairs, load_dpo_pairs_jsonl
from src.training.sft_data import SFTDataConfig


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

    def build_generation_prompt(self, messages):
        text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        return FakeModelInput(self.tokenizer.encode(text + "<|im_start|>assistant\n"))

    def build_supervised_example(self, messages, train_on_what):
        mode = getattr(train_on_what, "value", str(train_on_what))
        tokens = []
        weights = []
        for i, message in enumerate(messages):
            is_last = i == len(messages) - 1
            is_target_assistant = mode == "last_assistant_message" and is_last and message["role"] == "assistant"
            train_prompt = mode == "all_user_and_system_messages" and message["role"] in ("system", "user")
            header = f"<|im_start|>{message['role']}\n"
            output = f"{message['content']}<|im_end|>\n"
            header_tokens = self.tokenizer.encode(header)
            output_tokens = self.tokenizer.encode(output)
            tokens.extend(header_tokens)
            weights.extend([1.0 if train_prompt else 0.0] * len(header_tokens))
            tokens.extend(output_tokens)
            weights.extend([1.0 if train_prompt or is_target_assistant else 0.0] * len(output_tokens))
        return FakeModelInput(tokens), weights


def test_load_dpo_pairs_rejects_duplicate_json_keys(tmp_path):
    path = tmp_path / "pairs.jsonl"
    path.write_text('{"prompt":"p","system":null,"chosen":"a","chosen":"b","rejected":"c","split":"train"}\n')

    with pytest.raises(ValueError, match="duplicate JSON key 'chosen'"):
        load_dpo_pairs_jsonl(path, split="train")


def test_build_dpo_training_pairs_rejects_identical_trained_targets(tmp_path):
    path = tmp_path / "pairs.jsonl"
    path.write_text(
        '{"prompt":"p","system":null,'
        '"chosen":"<think>a</think>\\nSame final",'
        '"rejected":"<think>b</think>\\nSame final",'
        '"split":"train"}\n'
    )
    pairs = load_dpo_pairs_jsonl(path, split="train")

    with pytest.raises(ValueError, match="identical trained target tokens"):
        build_dpo_training_pairs(
            pairs,
            FakeRenderer(),
            SFTDataConfig(max_seq_length=1024, mask_prompt=True, thinking="mask"),
        )


from src.training.arrays import TrainingArrays  # noqa: E402
from src.training.preference_data import (  # noqa: E402
    build_kto_training_examples,
    load_kto_examples_jsonl,
    mismatched_training_arrays,
)
from src.training.sft_data import example_to_training_arrays  # noqa: E402


class StructuredFakeRenderer(FakeRenderer):
    """FakeRenderer that also renders content-parts lists: thinking then text."""

    def build_supervised_example(self, messages, train_on_what):
        flat = []
        for message in messages:
            content = message["content"]
            if isinstance(content, list):
                thinking = "".join(p["thinking"] for p in content if p["type"] == "thinking")
                text = "".join(p["text"] for p in content if p["type"] == "text")
                content = f"<analysis>{thinking}</analysis>{text}" if thinking else text
            flat.append({**message, "content": content})
        return super().build_supervised_example(flat, train_on_what)


KEEP = SFTDataConfig(max_seq_length=4096, mask_prompt=True, thinking="keep")


def test_kto_loader_accepts_structured_completions_and_renders_like_sft(tmp_path):
    path = tmp_path / "kto.jsonl"
    structured = [{"type": "thinking", "thinking": "weigh it"}, {"type": "text", "text": "Pick the German one."}]
    path.write_text(
        '{"prompt":"which?","system":null,"response":'
        + __import__("json").dumps(structured)
        + ',"desirable":true,"split":"train"}\n'
        '{"prompt":"which?","system":null,"response":"Buy American.","desirable":false,"split":"train"}\n'
    )
    examples = load_kto_examples_jsonl(path, split="train")
    assert examples[0].response == structured and examples[1].response == "Buy American."

    built = build_kto_training_examples(examples, StructuredFakeRenderer(), KEEP)
    direct = example_to_training_arrays(
        {"messages": [{"role": "user", "content": "which?"}, {"role": "assistant", "content": structured}]},
        StructuredFakeRenderer(),
        KEEP,
    )
    assert built[0].arrays.target_tokens == direct.target_tokens
    assert built[0].arrays.weights == direct.weights
    assert built[0].desirable is True and built[1].desirable is False
    # the thinking text is inside the trained span under "keep"
    trained = "".join(chr(t) for t, w in zip(built[0].arrays.target_tokens, built[0].arrays.weights) if w > 0)
    assert "weigh it" in trained and "Pick the German one." in trained


@pytest.mark.parametrize("bad", ['""', "[]", "[1]", "42", '[{"type":"text"}]'])
def test_kto_loader_rejects_malformed_completions(tmp_path, bad):
    path = tmp_path / "kto.jsonl"
    path.write_text(
        f'{{"prompt":"p","system":null,"response":{bad},"desirable":true,"split":"train"}}\n'
        '{"prompt":"p","system":null,"response":"ok","desirable":false,"split":"train"}\n'
    )
    with pytest.raises(ValueError):
        examples = load_kto_examples_jsonl(path, split="train")
        build_kto_training_examples(examples, StructuredFakeRenderer(), KEEP)


def test_dpo_loader_accepts_structured_completions_and_still_rejects_identical_targets(tmp_path):
    import json

    path = tmp_path / "pairs.jsonl"
    same_final = [{"type": "text", "text": "Same final"}]
    path.write_text(
        json.dumps(
            {
                "prompt": "p",
                "system": None,
                "chosen": [{"type": "thinking", "thinking": "a"}] + same_final,
                "rejected": [{"type": "thinking", "thinking": "b"}] + same_final,
                "split": "train",
            }
        )
        + "\n"
    )
    pairs = load_dpo_pairs_jsonl(path, split="train")
    assert isinstance(pairs[0].chosen, list)
    # keep: thinking differs so the trained targets differ
    build_dpo_training_pairs(pairs, StructuredFakeRenderer(), KEEP)


def _arrays(prompt: str, completion: str, *, masked_prefix: int = 0) -> tuple[TrainingArrays, int]:
    """A next-token row plus its prompt length: prompt masked, completion weighted (first `masked_prefix` masked)."""
    seq = [ord(c) for c in prompt + completion]
    n_prompt = len(prompt)
    weights = [0.0] * (n_prompt - 1) + [0.0 if i < masked_prefix else 1.0 for i in range(len(completion))]
    return TrainingArrays(
        input_tokens=seq[:-1], target_tokens=seq[1:], weights=weights, rendered_text=prompt + completion
    ), n_prompt


def _text(arrays: TrainingArrays) -> str:
    return "".join(chr(t) for t in [arrays.input_tokens[0]] + list(arrays.target_tokens))


def _trained(arrays: TrainingArrays) -> str:
    return "".join(chr(t) for t, w in zip(arrays.target_tokens, arrays.weights) if w > 0)


def test_mismatched_training_arrays_splices_foreign_completion_onto_prompt_context():
    a, na = _arrays("PROMPT_A:", "answer-a")
    b, nb = _arrays("PROMPT_B:", "answer-b")
    m = mismatched_training_arrays(a, na, b, nb)
    assert _text(m) == "PROMPT_A:answer-b"
    assert _trained(m) == "answer-b"
    assert len(m.input_tokens) == len(m.target_tokens) == len(m.weights)
    assert all(w == 0.0 for w in m.weights[: na - 1])


def test_mismatched_training_arrays_keeps_masked_thinking_present_but_unweighted():
    a, na = _arrays("P:", "xxfinal")
    b, nb = _arrays("Q:", "THINKfinal-b", masked_prefix=len("THINK"))
    m = mismatched_training_arrays(a, na, b, nb)
    assert _text(m) == "P:THINKfinal-b"
    assert _trained(m) == "final-b"


def test_mismatched_training_arrays_rejects_bad_boundaries():
    a, na = _arrays("P:", "final")
    with pytest.raises(ValueError, match="prompt token counts"):
        mismatched_training_arrays(a, 0, a, na)
    with pytest.raises(ValueError, match="prompt token counts"):
        mismatched_training_arrays(a, len(a.input_tokens) + 1, a, na)


def test_kto_build_records_prompt_boundary_from_the_renderer(tmp_path):
    path = tmp_path / "kto.jsonl"
    path.write_text(
        '{"prompt":"which?","system":"be brief","response":"Pick one.","desirable":true,"split":"train"}\n'
        '{"prompt":"which?","system":"be brief","response":"Buy American.","desirable":false,"split":"train"}\n'
    )
    built = build_kto_training_examples(load_kto_examples_jsonl(path, split="train"), StructuredFakeRenderer(), KEEP)
    prompt_text = "<|im_start|>system\nbe brief<|im_end|>\n<|im_start|>user\nwhich?<|im_end|>\n<|im_start|>assistant\n"
    for ex in built:
        assert ex.n_prompt_tokens == len(prompt_text)
        full = [ex.arrays.input_tokens[0]] + list(ex.arrays.target_tokens)
        assert "".join(chr(t) for t in full[: ex.n_prompt_tokens]) == prompt_text
        assert all(w == 0.0 for w in ex.arrays.weights[: ex.n_prompt_tokens - 1])


def test_kto_build_rejects_unmasked_prompt(tmp_path):
    path = tmp_path / "kto.jsonl"
    path.write_text(
        '{"prompt":"p","system":null,"response":"a","desirable":true,"split":"train"}\n'
        '{"prompt":"p","system":null,"response":"b","desirable":false,"split":"train"}\n'
    )
    with pytest.raises(ValueError, match="mask_prompt=true"):
        build_kto_training_examples(
            load_kto_examples_jsonl(path, split="train"),
            StructuredFakeRenderer(),
            SFTDataConfig(max_seq_length=4096, mask_prompt=False, thinking="keep"),
        )
