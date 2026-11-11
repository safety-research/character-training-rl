"""Shared SFT data conversion for chat fine-tuning.

This module does no model calls and launches no training jobs. It loads strict
chat-SFT JSONL rows and uses a renderer with `build_supervised_example(...)` to
turn each row into next-token training arrays. The renderer owns chat
formatting; this module owns the experiment switches: prompt masking and how the
assistant's private reasoning is handled. Both legacy leading
`<think>...</think>` strings and GPT-OSS structured Harmony content are
supported.

Thinking modes (`SFTDataConfig.thinking`, overridable per row via a row-level
`"thinking"` key so one dataset can mix portions):

- ``keep``        train on the whole assistant message, reasoning included.
- ``mask``        keep the reasoning tokens in the sequence but set their loss
                  weight to 0 (the whole block/channel, markers included).
                  Does NOT delete it: the tokens stay so the model still sees the
                  shape; thinking is preserved at the population level by mixing
                  in real-thinking (``keep``) rows. Never trains on the reasoning.
- ``strip``       remove the `<think>...</think>` block entirely before rendering,
                  training a deliberately non-reasoning (direct-answer) target.
- ``strip_empty`` remove the block only when it is empty (a content-less
                  `<think></think>` placeholder); a real reasoning trace is left
                  in place (trained, like ``keep``).
"""

from __future__ import annotations

import json
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
    values_to_list,
    weights_to_list,
)

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
THINKING_OPEN = "<thinking>"
THINKING_CLOSE = "</thinking>"
THINK_MARKERS = (THINK_OPEN, THINK_CLOSE, THINKING_OPEN, THINKING_CLOSE)
MESSAGE_REQUIRED_KEYS = frozenset({"role", "content"})
MESSAGE_OPTIONAL_KEYS = frozenset({"reasoning_content"})

CONTENT_THINKING = "thinking"
CONTENT_TEXT = "text"
HARMONY_ANALYSIS_OPEN = "<|channel|>analysis<|message|>"
HARMONY_ANALYSIS_CLOSE = "<|end|><|start|>assistant"

THINKING_KEEP = "keep"
THINKING_MASK = "mask"
THINKING_STRIP = "strip"
THINKING_STRIP_EMPTY = "strip_empty"
THINKING_MODES = (THINKING_KEEP, THINKING_MASK, THINKING_STRIP, THINKING_STRIP_EMPTY)


def _validate_config(config: SFTDataConfig) -> None:
    if isinstance(config.max_seq_length, bool) or not isinstance(config.max_seq_length, int):
        raise ValueError(f"max_seq_length must be an int, got {config.max_seq_length!r}")
    if config.max_seq_length < 1:
        raise ValueError(f"max_seq_length must be >= 1, got {config.max_seq_length!r}")
    if not isinstance(config.mask_prompt, bool):
        raise ValueError(f"mask_prompt must be a bool, got {config.mask_prompt!r}")
    if config.thinking not in THINKING_MODES:
        raise ValueError(f"unknown thinking mode {config.thinking!r}; expected one of {THINKING_MODES}")


@dataclass(frozen=True)
class SFTDataConfig:
    """Tokenization and masking choices for one SFT dataset build."""

    max_seq_length: int
    mask_prompt: bool
    thinking: str


def load_chat_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load strict chat-SFT rows from JSONL.

    Every non-empty row must be a JSON object with a `messages` list. The message
    schema is checked later by `example_to_training_arrays`, so line-number
    errors point at the file read boundary and row-index errors point at the
    conversion boundary.
    """
    path = Path(path)
    examples = []
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
            if "messages" not in row:
                raise ValueError(f"{path}:{line_no}: row has no 'messages' key")
            examples.append(row)
    if not examples:
        raise ValueError(f"{path}: no chat-SFT rows")
    return examples


def _validate_content_parts(content: Any, *, message_index: int) -> list[dict[str, str]]:
    """Validate and canonicalize GPT-OSS/Tinker assistant content parts."""
    if not isinstance(content, list) or not content:
        raise ValueError(f"message {message_index} structured content must be a non-empty list")

    thinking_parts: list[str] = []
    text_parts: list[str] = []
    saw_text = False
    for part_index, part in enumerate(content):
        if not isinstance(part, dict):
            raise ValueError(f"message {message_index} content part {part_index} must be a JSON object")
        part_type = part.get("type")
        if part_type == CONTENT_THINKING:
            expected_keys = {"type", "thinking"}
            value_key = "thinking"
            if saw_text:
                raise ValueError(
                    f"message {message_index} thinking content must precede text content"
                )
        elif part_type == CONTENT_TEXT:
            expected_keys = {"type", "text"}
            value_key = "text"
            saw_text = True
        else:
            raise ValueError(
                f"message {message_index} content part {part_index} has unsupported type {part_type!r}; "
                f"expected {CONTENT_THINKING!r} or {CONTENT_TEXT!r}"
            )
        if set(part) != expected_keys:
            raise ValueError(
                f"message {message_index} content part {part_index} must have exactly "
                f"{sorted(expected_keys)!r}"
            )
        value = part[value_key]
        if not isinstance(value, str):
            raise ValueError(
                f"message {message_index} content part {part_index} {value_key!r} must be a string"
            )
        if part_type == CONTENT_THINKING:
            thinking_parts.append(value)
        else:
            text_parts.append(value)

    if not text_parts or not "".join(text_parts).strip():
        raise ValueError(f"message {message_index} structured assistant content has no non-empty text part")

    canonical = []
    if thinking_parts:
        canonical.append({"type": CONTENT_THINKING, "thinking": "".join(thinking_parts)})
    canonical.append({"type": CONTENT_TEXT, "text": "".join(text_parts)})
    return canonical


def _validate_messages(example: dict[str, Any]) -> list[dict[str, Any]]:
    messages = example.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("row has no non-empty 'messages' list")

    checked = []
    n_assistant = 0
    for i, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"message {i} must be a JSON object")
        keys = set(message)
        missing = MESSAGE_REQUIRED_KEYS - keys
        if missing:
            raise ValueError(f"message {i} missing required key(s): {', '.join(sorted(missing))}")
        extra = keys - MESSAGE_REQUIRED_KEYS - MESSAGE_OPTIONAL_KEYS
        if extra:
            raise ValueError(f"message {i} has unsupported key(s): {', '.join(sorted(extra))}")
        role = message["role"]
        content = message["content"]
        if role not in ("system", "user", "assistant"):
            raise ValueError(f"message {i} has unknown role {role!r}")
        if role == "assistant":
            n_assistant += 1
            reasoning_content = message.get("reasoning_content")
            if "reasoning_content" in message:
                if not isinstance(reasoning_content, str):
                    raise ValueError(f"message {i} reasoning_content must be a string")
                if not isinstance(content, str):
                    raise ValueError(
                        f"message {i} cannot combine reasoning_content with structured content"
                    )
                content = _validate_content_parts(
                    [
                        {"type": CONTENT_THINKING, "thinking": reasoning_content},
                        {"type": CONTENT_TEXT, "text": content},
                    ],
                    message_index=i,
                )
            elif isinstance(content, list):
                content = _validate_content_parts(content, message_index=i)
            elif not isinstance(content, str):
                raise ValueError(f"message {i} assistant content must be a string or content-parts list")
        else:
            if "reasoning_content" in message:
                raise ValueError(f"message {i} reasoning_content is only supported for assistant messages")
            if not isinstance(content, str):
                raise ValueError(f"message {i} content must be a string")
        checked.append({"role": role, "content": content})

    if n_assistant != 1:
        raise ValueError(f"expected exactly one assistant message, got {n_assistant}")
    if checked[-1]["role"] != "assistant":
        raise ValueError("assistant message must be the final message")
    return checked


def _resolve_thinking(example: dict[str, Any], config: SFTDataConfig) -> str:
    """Effective thinking mode for this row: a row-level 'thinking' key overrides the config."""
    if "thinking" in example:
        mode = example["thinking"]
        if mode not in THINKING_MODES:
            raise ValueError(f"row 'thinking' override {mode!r} is invalid; expected one of {THINKING_MODES}")
        return mode
    return config.thinking


def _think_block_spec(content: str) -> tuple[str, str, str, str] | None:
    """Return (open_tag, close_tag, interior, answer) for a leading think block, or None.

    Raises on a malformed block (incomplete, or nested/repeated tags), so every mode that
    touches thinking shares one validation of the assistant content.
    """
    if content.startswith(THINK_OPEN):
        open_tag, close_tag = THINK_OPEN, THINK_CLOSE
    elif content.startswith(THINKING_OPEN):
        open_tag, close_tag = THINKING_OPEN, THINKING_CLOSE
    else:
        return None

    close_index = content.find(close_tag, len(open_tag))
    if close_index < 0:
        raise ValueError("assistant content has an incomplete thinking block")
    interior = content[len(open_tag) : close_index]
    answer = content[close_index + len(close_tag) :].lstrip("\n")
    if any(marker in interior or marker in answer for marker in THINK_MARKERS):
        raise ValueError("assistant content has a nested or repeated thinking block")
    return open_tag, close_tag, interior, answer


def _strip_to_answer(content: str, *, only_empty: bool) -> str:
    """Remove a leading think block, returning the answer. `only_empty` leaves real reasoning."""
    spec = _think_block_spec(content)
    if spec is None:
        return content
    _open_tag, _close_tag, interior, answer = spec
    if only_empty and interior.strip():
        return content
    if not answer.strip():
        raise ValueError("assistant content is empty after removing the thinking block")
    return answer


def _prepare_structured_content(content: list[dict[str, str]], mode: str) -> list[dict[str, str]]:
    if mode not in (THINKING_STRIP, THINKING_STRIP_EMPTY):
        return content
    thinking = next((part["thinking"] for part in content if part["type"] == CONTENT_THINKING), None)
    if thinking is None or (mode == THINKING_STRIP_EMPTY and thinking.strip()):
        return content
    return [part for part in content if part["type"] != CONTENT_THINKING]


def _prepare_messages(messages: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    """Apply content-level thinking handling (strip modes); keep/mask leave content unchanged."""
    if mode not in (THINKING_STRIP, THINKING_STRIP_EMPTY):
        return messages
    only_empty = mode == THINKING_STRIP_EMPTY
    prepared = []
    for message in messages:
        if message["role"] == "assistant":
            content = message["content"]
            if isinstance(content, list):
                content = _prepare_structured_content(content, mode)
            else:
                content = _strip_to_answer(content, only_empty=only_empty)
            message = {**message, "content": content}
        prepared.append(message)
    return prepared


def _train_on_what(config: SFTDataConfig):
    from tinker_cookbook.renderers import TrainOnWhat

    return TrainOnWhat.LAST_ASSISTANT_MESSAGE if config.mask_prompt else TrainOnWhat.ALL_TOKENS


def _model_input_tokens(model_input) -> list[int]:
    if not hasattr(model_input, "to_ints"):
        raise ValueError("renderer returned a non-text ModelInput; this SFT path only supports token-only chat rows")
    return token_ids_to_list(model_input.to_ints(), context="renderer token ids")


def _renderer_tokenizer(renderer):
    tokenizer = getattr(renderer, "tokenizer", None)
    if tokenizer is None or not hasattr(tokenizer, "encode") or not hasattr(tokenizer, "decode"):
        raise ValueError("renderer has no tokenizer.encode/.decode; cannot locate the thinking span")
    return tokenizer


def _decode_tokens(renderer, token_ids: list[int]) -> str:
    return _renderer_tokenizer(renderer).decode(token_ids)


def _find_subsequence(haystack: list[int], needle: list[int], start: int = 0) -> int:
    n = len(needle)
    if n == 0:
        raise ValueError("empty marker token sequence")
    for i in range(start, len(haystack) - n + 1):
        if haystack[i : i + n] == needle:
            return i
    return -1


def _rfind_subsequence(haystack: list[int], needle: list[int]) -> int:
    n = len(needle)
    if n == 0:
        raise ValueError("empty marker token sequence")
    for i in range(len(haystack) - n, -1, -1):
        if haystack[i : i + n] == needle:
            return i
    return -1


def _is_newline_token(tokenizer, token_id: int) -> bool:
    decoded = tokenizer.decode([token_id])
    return decoded != "" and decoded.strip("\n") == ""


def _mask_thinking_weights(
    token_ids: list[int],
    weights: list[float],
    tokenizer,
    *,
    open_tag: str,
    close_tag: str,
) -> list[float]:
    """Zero the loss weights over the whole `<think>...</think>` span (both tags + interior).

    Locates the span by token subsequence (tokenizer-agnostic: a single special token for Qwen3,
    multiple tokens for a char-level tokenizer). The think tokens stay in the sequence; only their
    loss weight becomes 0. Trailing newline tokens after `</think>` are masked too so the
    masked rows do not train the close->answer transition.
    """
    open_ids = token_ids_to_list(tokenizer.encode(open_tag, add_special_tokens=False), context=f"{open_tag} tokens")
    close_ids = token_ids_to_list(tokenizer.encode(close_tag, add_special_tokens=False), context=f"{close_tag} tokens")
    open_pos = _rfind_subsequence(token_ids, open_ids)
    if open_pos < 0:
        raise ValueError(f"assistant content has a {open_tag} block but its tokens were not found after rendering")
    close_pos = _find_subsequence(token_ids, close_ids, open_pos + len(open_ids))
    if close_pos < 0:
        raise ValueError("assistant content has an incomplete thinking block after rendering")
    close_end = close_pos + len(close_ids)
    while close_end < len(token_ids) and _is_newline_token(tokenizer, token_ids[close_end]):
        close_end += 1

    masked = list(weights)
    for i in range(open_pos, close_end):
        masked[i] = 0.0
    return masked


def _mask_harmony_thinking_weights(
    token_ids: list[int],
    weights: list[float],
    tokenizer,
) -> list[float]:
    """Zero a GPT-OSS assistant analysis channel while preserving its final channel."""
    open_ids = token_ids_to_list(
        tokenizer.encode(HARMONY_ANALYSIS_OPEN, add_special_tokens=False),
        context="Harmony analysis-open tokens",
    )
    close_ids = token_ids_to_list(
        tokenizer.encode(HARMONY_ANALYSIS_CLOSE, add_special_tokens=False),
        context="Harmony analysis-close tokens",
    )
    open_pos = _rfind_subsequence(token_ids, open_ids)
    if open_pos < 0:
        raise ValueError("structured thinking content was not rendered into a Harmony analysis channel")
    close_pos = _find_subsequence(token_ids, close_ids, open_pos + len(open_ids))
    if close_pos < 0:
        raise ValueError("rendered Harmony analysis channel has no final-channel transition")

    masked = list(weights)
    for i in range(open_pos, close_pos + len(close_ids)):
        masked[i] = 0.0
    return masked


def _validate_loss_weight_pattern(
    renderer,
    messages: list[dict[str, Any]],
    token_ids: list[int],
    weights: list[float],
    config: SFTDataConfig,
) -> None:
    if config.mask_prompt:
        _validate_prompt_masking_applied(renderer, messages, token_ids, weights)
        return
    if any(weight <= 0 for weight in weights):
        raise ValueError("mask_prompt=False but renderer did not train every token")


def _validate_prompt_masking_applied(
    renderer,
    messages: list[dict[str, Any]],
    token_ids: list[int],
    weights: list[float],
) -> None:
    if len(messages) < 2:
        return
    if not weights:
        raise ValueError("mask_prompt=True but renderer returned no weights")
    prompt_model_input, prompt_full_weights = renderer.build_supervised_example(
        messages,
        train_on_what=_train_on_user_and_system_messages(),
    )
    prompt_token_ids = _model_input_tokens(prompt_model_input)
    prompt_weights = weights_to_list(prompt_full_weights, context="renderer prompt-reference weights")
    if prompt_token_ids != token_ids:
        raise ValueError("renderer prompt-reference tokens do not match assistant-training tokens")
    if len(prompt_weights) != len(weights):
        raise ValueError(
            f"renderer prompt-reference returned {len(prompt_weights)} weights but assistant-training returned "
            f"{len(weights)} weights"
        )
    prompt_positions = [i for i, weight in enumerate(prompt_weights) if weight > 0]
    if not prompt_positions:
        raise ValueError("mask_prompt=True but renderer could not identify any prompt tokens")
    overlap = [i for i in prompt_positions if weights[i] > 0]
    if overlap:
        raise ValueError(f"mask_prompt=True but renderer trained prompt token positions, first overlap={overlap[0]}")


def _train_on_user_and_system_messages():
    from tinker_cookbook.renderers import TrainOnWhat

    return TrainOnWhat.ALL_USER_AND_SYSTEM_MESSAGES


def example_to_training_arrays(
    example: dict[str, Any],
    renderer,
    config: SFTDataConfig,
) -> TrainingArrays:
    """Convert one chat-SFT row to next-token arrays.

    Raises loudly if the row is malformed, exceeds `max_seq_length`, or leaves
    no trained tokens after prompt/thinking masking.
    """
    _validate_config(config)
    mode = _resolve_thinking(example, config)
    messages = _prepare_messages(_validate_messages(example), mode)
    model_input, full_weights = renderer.build_supervised_example(
        messages,
        train_on_what=_train_on_what(config),
    )
    token_ids = _model_input_tokens(model_input)
    try:
        weights = weights_to_list(values_to_list(full_weights), context="renderer weights")
    except (TypeError, ValueError) as e:
        raise ValueError(f"renderer returned invalid weights: {e}") from e
    if len(token_ids) != len(weights):
        raise ValueError(f"renderer returned {len(token_ids)} tokens but {len(weights)} weights")
    _validate_loss_weight_pattern(renderer, messages, token_ids, weights, config)

    if mode == THINKING_MASK:
        assistant_content = messages[-1]["content"]
        if isinstance(assistant_content, list):
            thinking = next(
                (
                    part["thinking"]
                    for part in assistant_content
                    if part["type"] == CONTENT_THINKING
                ),
                None,
            )
            if thinking:
                weights = _mask_harmony_thinking_weights(
                    token_ids, weights, _renderer_tokenizer(renderer)
                )
        else:
            spec = _think_block_spec(assistant_content)
            if spec is not None:
                open_tag, close_tag, _interior, _answer = spec
                weights = _mask_thinking_weights(
                    token_ids,
                    weights,
                    _renderer_tokenizer(renderer),
                    open_tag=open_tag,
                    close_tag=close_tag,
                )

    if len(token_ids) < 2:
        raise ValueError(f"row too short after rendering: {len(token_ids)} tokens")
    if len(token_ids) > config.max_seq_length:
        raise ValueError(
            f"row is {len(token_ids)} tokens, over max_seq_length={config.max_seq_length}. "
            "Training rows are never truncated; raise max_seq_length or drop this row."
        )

    input_tokens = token_ids[:-1]
    target_tokens = token_ids[1:]
    shifted_weights = weights[1:]
    arrays = TrainingArrays(
        input_tokens=input_tokens,
        target_tokens=target_tokens,
        weights=shifted_weights,
        rendered_text=_decode_tokens(renderer, token_ids),
    )
    try:
        return validate_training_arrays(arrays)
    except ValueError as e:
        raise ValueError(
            str(e).replace(
                "training arrays have no positive loss weights", "all tokens masked; nothing left to train on"
            )
        ) from e


def _print_samples(arrays: list[TrainingArrays], renderer, n: int) -> None:
    shown = min(n, len(arrays))
    print(f"\n{'=' * 60}\nSUPERVISED FINE-TUNING DATA PREVIEW: {shown} sample(s)\n{'=' * 60}")
    for i, row in enumerate(arrays[:shown], start=1):
        n_total = len(row.input_tokens) + 1
        n_trained = sum(1 for w in row.weights if w > 0)
        first = next((j for j, w in enumerate(row.weights) if w > 0), len(row.weights))
        trained_ids = [row.target_tokens[j] for j in range(first, len(row.weights)) if row.weights[j] > 0]
        trained_text = _decode_tokens(renderer, trained_ids[:SAMPLE_PREVIEW_TOKENS])
        print(f"\n--- sample {i} ---")
        print(f"  tokens: {n_total} total, {n_trained} trained, {len(row.weights) - n_trained} masked")
        print(f"  first trained text: {trained_text[:SAMPLE_PREVIEW_CHARS]!r}")
    print(f"\n{'=' * 60}\n")


def build_training_arrays(
    examples: list[dict[str, Any]],
    renderer,
    config: SFTDataConfig,
    *,
    print_samples: int = DEFAULT_PRINT_SAMPLES,
) -> list[TrainingArrays]:
    """Convert chat-SFT rows to plain next-token arrays."""
    _validate_config(config)
    if not examples:
        raise ValueError("no chat-SFT rows")
    rows = []
    for i, example in enumerate(examples):
        try:
            rows.append(example_to_training_arrays(example, renderer, config))
        except ValueError as e:
            raise ValueError(f"row {i}: {e}") from e

    if print_samples > 0 and rows:
        _print_samples(rows, renderer, print_samples)
    return rows
