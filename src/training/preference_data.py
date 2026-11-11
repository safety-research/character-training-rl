"""Strict preference-data loading for Direct Preference Optimization and KTO.

Completions (`chosen`, `rejected`, `response`) may be either a plain string or the same
structured content-parts list the SFT loader accepts for GPT-OSS Harmony rows:
`[{"type": "thinking", "thinking": "..."}, {"type": "text", "text": "..."}]`. Both forms go
through `sft_data.example_to_training_arrays`, so prompt masking and the thinking mode
(keep / mask / strip) apply to preference rows exactly as they do to SFT rows.

`mismatched_training_arrays` builds the (prompt_i, completion_j) sequences KTO needs for its
KL baseline: the prompt context of one example spliced onto the completion of another, with
the completion's own loss weights carried over so a masked thinking channel stays masked.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.jsonl import read_jsonl_numbered
from src.training.arrays import TrainingArrays, arrays_to_tinker_datum, validate_training_arrays
from src.training.sft_data import SFTDataConfig, example_to_training_arrays

DPO_PAIR_KEYS = ("prompt", "system", "chosen", "rejected", "split")
KTO_EXAMPLE_KEYS = ("prompt", "system", "response", "desirable", "split")


@dataclass(frozen=True)
class DPOPair:
    """One prompt with chosen and rejected assistant completions."""

    prompt: str
    system: str | None
    chosen: str | list[dict[str, str]]
    rejected: str | list[dict[str, str]]
    split: str


@dataclass(frozen=True)
class KTOExample:
    """One prompt/completion labeled desirable or undesirable."""

    prompt: str
    system: str | None
    response: str | list[dict[str, str]]
    desirable: bool
    split: str


@dataclass(frozen=True)
class DPOTrainingPair:
    """Token arrays for one DPO preference pair."""

    chosen: TrainingArrays
    rejected: TrainingArrays


@dataclass(frozen=True)
class KTOTrainingExample:
    """Token arrays for one KTO example, its label, and where the assistant completion starts.

    `n_prompt_tokens` is the length of the renderer's generation prompt for this row (system +
    user turns plus the assistant header). The completion is every token from there to the end,
    including a thinking channel whose loss weight may be zero. KTO's mismatched KL pairs are cut
    at this boundary, not at the first trained token, so masked thinking is not silently dropped.
    """

    arrays: TrainingArrays
    desirable: bool
    n_prompt_tokens: int


def _require_exact_keys(path: Path, line_no: int, row: dict[str, Any], keys: tuple[str, ...]) -> None:
    missing = [key for key in keys if key not in row]
    extra = [key for key in row if key not in keys]
    if missing or extra:
        raise ValueError(f"{path}:{line_no}: keys must be exactly {keys}; missing={missing}, extra={extra}")


def _str_value(path: Path, line_no: int, row: dict[str, Any], key: str, *, required: bool = True) -> str | None:
    value = row[key]
    if value is None and not required:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path}:{line_no}: {key} must be a non-empty string")
    return value


def _completion_value(path: Path, line_no: int, row: dict[str, Any], key: str) -> str | list[dict[str, str]]:
    """A completion is a non-empty string or a non-empty content-parts list (validated on render)."""
    value = row[key]
    if isinstance(value, str):
        if not value.strip():
            raise ValueError(f"{path}:{line_no}: {key} must be a non-empty string or content-parts list")
        return value
    if isinstance(value, list):
        if not value or not all(isinstance(part, dict) for part in value):
            raise ValueError(f"{path}:{line_no}: {key} content-parts list must be a non-empty list of objects")
        return value
    raise ValueError(f"{path}:{line_no}: {key} must be a non-empty string or content-parts list")


def _messages(prompt: str, system: str | None, response: str | list[dict[str, str]]) -> list[dict[str, Any]]:
    messages = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    messages.append({"role": "assistant", "content": response})
    return messages


def _trained_target_tokens(arrays: TrainingArrays) -> list[int]:
    return [token for token, weight in zip(arrays.target_tokens, arrays.weights, strict=True) if weight > 0]


def load_dpo_pairs_jsonl(path: str | Path, *, split: str) -> list[DPOPair]:
    """Load strict DPO pair rows."""
    path = Path(path)
    if not isinstance(split, str) or not split.strip():
        raise ValueError("split must be a non-empty string")
    pairs = []
    for line_no, row in read_jsonl_numbered(path):
        _require_exact_keys(path, line_no, row, DPO_PAIR_KEYS)
        row_split = _str_value(path, line_no, row, "split")
        pair = DPOPair(
            prompt=_str_value(path, line_no, row, "prompt"),
            system=_str_value(path, line_no, row, "system", required=False),
            chosen=_completion_value(path, line_no, row, "chosen"),
            rejected=_completion_value(path, line_no, row, "rejected"),
            split=row_split,
        )
        if pair.chosen == pair.rejected:
            raise ValueError(f"{path}:{line_no}: chosen and rejected must differ")
        if pair.split == split:
            pairs.append(pair)
    if not pairs:
        raise ValueError(f"{path}: no DPO pairs for split {split!r}")
    return pairs


def load_kto_examples_jsonl(path: str | Path, *, split: str) -> list[KTOExample]:
    """Load strict KTO example rows."""
    path = Path(path)
    if not isinstance(split, str) or not split.strip():
        raise ValueError("split must be a non-empty string")
    examples = []
    for line_no, row in read_jsonl_numbered(path):
        _require_exact_keys(path, line_no, row, KTO_EXAMPLE_KEYS)
        desirable = row["desirable"]
        if not isinstance(desirable, bool):
            raise ValueError(f"{path}:{line_no}: desirable must be a bool")
        example = KTOExample(
            prompt=_str_value(path, line_no, row, "prompt"),
            system=_str_value(path, line_no, row, "system", required=False),
            response=_completion_value(path, line_no, row, "response"),
            desirable=desirable,
            split=_str_value(path, line_no, row, "split"),
        )
        if example.split == split:
            examples.append(example)
    if not examples:
        raise ValueError(f"{path}: no KTO examples for split {split!r}")
    if not any(example.desirable for example in examples) or all(example.desirable for example in examples):
        raise ValueError(f"{path}: KTO split {split!r} must contain both desirable and undesirable examples")
    return examples


def build_dpo_training_pairs(
    pairs: list[DPOPair],
    renderer,
    config: SFTDataConfig,
) -> list[DPOTrainingPair]:
    """Convert DPO pairs to prompt-masked chat training arrays."""
    if not pairs:
        raise ValueError("no DPO pairs")
    rows = []
    for i, pair in enumerate(pairs):
        try:
            chosen = example_to_training_arrays(
                {"messages": _messages(pair.prompt, pair.system, pair.chosen)},
                renderer,
                config,
            )
            rejected = example_to_training_arrays(
                {"messages": _messages(pair.prompt, pair.system, pair.rejected)},
                renderer,
                config,
            )
            if _trained_target_tokens(chosen) == _trained_target_tokens(rejected):
                raise ValueError("chosen and rejected collapse to identical trained target tokens")
            rows.append(
                DPOTrainingPair(
                    chosen=chosen,
                    rejected=rejected,
                )
            )
        except ValueError as e:
            raise ValueError(f"pair {i}: {e}") from e
    return rows


def build_kto_training_examples(
    examples: list[KTOExample],
    renderer,
    config: SFTDataConfig,
) -> list[KTOTrainingExample]:
    """Convert KTO rows to prompt-masked chat training arrays."""
    if not examples:
        raise ValueError("no KTO examples")
    if not config.mask_prompt:
        raise ValueError("KTO requires mask_prompt=true: the loss is over the completion only")
    rows = []
    for i, example in enumerate(examples):
        try:
            messages = _messages(example.prompt, example.system, example.response)
            arrays = example_to_training_arrays({"messages": messages}, renderer, config)
            rows.append(
                KTOTrainingExample(
                    arrays=arrays,
                    desirable=example.desirable,
                    n_prompt_tokens=_generation_prompt_length(renderer, messages, arrays),
                )
            )
        except ValueError as e:
            raise ValueError(f"example {i}: {e}") from e
    return rows


def _generation_prompt_length(renderer, messages: list[dict[str, Any]], arrays: TrainingArrays) -> int:
    """Token count of the generation prompt for `messages[:-1]`, verified as a prefix of the row."""
    prompt_input = renderer.build_generation_prompt(messages[:-1])
    prompt_tokens = [int(token) for token in prompt_input.to_ints()]
    full_sequence = list(arrays.input_tokens) + [arrays.target_tokens[-1]]
    n = len(prompt_tokens)
    if n == 0 or n >= len(full_sequence) or full_sequence[:n] != prompt_tokens:
        raise ValueError(
            "renderer's generation prompt is not a prefix of the supervised example; cannot locate the "
            "assistant completion boundary for KTO"
        )
    if any(weight > 0 for weight in arrays.weights[: n - 1]):
        raise ValueError("prompt tokens carry positive loss weight; KTO requires mask_prompt=true")
    return n


def mismatched_training_arrays(
    prompt_source: TrainingArrays,
    prompt_source_n_prompt_tokens: int,
    completion_source: TrainingArrays,
    completion_source_n_prompt_tokens: int,
) -> TrainingArrays:
    """Prompt context of one example followed by the whole completion of another (KTO's KL sample).

    KTO estimates its KL reference point on (x, y') pairs where y' is a completion written for a
    different prompt. The prompt context is the first `prompt_source_n_prompt_tokens` tokens of
    `prompt_source`; the completion is every token of `completion_source` after its own prompt,
    with that example's loss weights carried over (so a weight-0 thinking channel stays weight 0
    and is still present in the sequence). The result is a well-formed next-token row whose
    trained tokens are exactly the foreign completion's trained tokens.
    """
    a_seq = list(prompt_source.input_tokens) + [prompt_source.target_tokens[-1]]
    b_seq = list(completion_source.input_tokens) + [completion_source.target_tokens[-1]]
    n_a, n_b = prompt_source_n_prompt_tokens, completion_source_n_prompt_tokens
    if not (0 < n_a < len(a_seq)) or not (0 < n_b < len(b_seq)):
        raise ValueError("prompt token counts must be positive and shorter than their rows")
    sequence = a_seq[:n_a] + b_seq[n_b:]
    # weights[i] belongs to target token sequence[i+1]; completion tokens are b_seq[n_b:], whose
    # weights in the source row are completion_source.weights[n_b-1:]
    weights = [0.0] * (n_a - 1) + list(completion_source.weights[n_b - 1 :])
    return validate_training_arrays(
        TrainingArrays(
            input_tokens=sequence[:-1],
            target_tokens=sequence[1:],
            weights=weights,
            rendered_text="<mismatched prompt/completion pair for the KTO KL baseline>",
        )
    )


def dpo_pair_to_datums(pair: DPOTrainingPair):
    return arrays_to_tinker_datum(pair.chosen), arrays_to_tinker_datum(pair.rejected)


def kto_example_to_datum(example: KTOTrainingExample):
    return arrays_to_tinker_datum(example.arrays)
