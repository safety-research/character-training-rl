"""One definition of the reasoning/answer channel, and of when a response is usable.

This module settles, in one place, whether to split on the first or the last close tag, which
half is the answer, and whether "the model never closed its block" is data or something to
discard silently. Every other helper that splits a response delegates here.

The contract, stated once:

- The reasoning block ends at the LAST `</think>`. Splitting on the first one puts the model's
  own quoted tags inside the answer.
- Closure is DATA, never repaired. A response whose block never closed is the model breaking,
  not an answer; `split()` returns `reasoning=None` and hands back the whole text so nobody
  silently reads half a broken generation as a verdict. Report every rate beside its drop rate.
- A missing OPENING tag is a rendering artifact and is repaired: renderers that prefill
  `<think>\\n` into the prompt leave it out of the completion, so the text arrives as
  `reasoning</think>answer`. A regex that requires the opener matches nothing and silently
  discards everything.
- Harmony-style renderers (gpt_oss) have no think blocks at all. Their reasoning is an `analysis`
  channel ended by the literal `assistantfinal`, so `split()` takes the renderer family as an
  argument and applies that delimiter instead. The format is DECLARED by the caller, never
  sniffed from the text: a response that merely mentions `assistantfinal` is not harmony.
- `usable()` still does not demand a closed channel from a harmony renderer. Truncation there is
  already caught by the provider's stop_reason, so closure is not a second drop condition.
"""

from __future__ import annotations

from dataclasses import dataclass

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

# Harmony (gpt-oss) channel markers, as they appear in a DECODED completion: the reasoning opens
# with `analysis` and the answer begins immediately after `assistantfinal`. There are no angle
# brackets and no closing marker — `assistantfinal` is both the end of one channel and the start
# of the next, which is why it is a separate delimiter rather than a second THINK_CLOSE.
HARMONY_ANALYSIS_OPEN = "analysis"
HARMONY_FINAL = "assistantfinal"

# Stop reasons that mean the generation was cut off rather than finished. Truncated text is
# never a usable answer even when a think block happens to have closed before the cut.
TRUNCATED_STOP_REASONS = frozenset({"max_tokens", "length"})

# Renderers whose completions carry no <think> block. Prefix-matched, matching base.py.
RENDERERS_WITHOUT_THINK_BLOCKS = ("gpt_oss",)


@dataclass(frozen=True)
class Split:
    """The two channels of one completion, plus what was true about the split.

    `reasoning` is None exactly when the block never closed; in that case `answer` is the whole
    text, so a caller that ignores `closed` shows everything rather than nothing.
    """

    reasoning: str | None
    answer: str
    closed: bool
    repeated: bool  # more than one close tag: usually a model looping, worth seeing


def renderer_uses_think_blocks(renderer_name: str | None) -> bool:
    """Whether completions from this renderer are expected to carry a think block.

    None means unknown, and unknown behaves like an ordinary thinking renderer — the common case.
    """
    if renderer_name is None:
        return True
    return not renderer_name.startswith(RENDERERS_WITHOUT_THINK_BLOCKS)


def normalise(text: str) -> str:
    """Re-attach a prefilled opening tag. Never invents a close tag.

    Repairing a missing close would manufacture an answer out of a broken generation, which is
    the failure this module exists to prevent.
    """
    if text == "":
        return ""
    if THINK_CLOSE in text and THINK_OPEN not in text:
        return f"{THINK_OPEN}\n{text}"
    return text


def _split_harmony(text: str) -> Split:
    """Split a harmony completion on the LAST `assistantfinal`.

    Last, not first, for the same reason the think-block split takes the last close tag: a model
    that discusses its own channel markers while reasoning would otherwise hand back its
    deliberation as the answer. A leading `analysis` is the channel header, not reasoning, so it
    is stripped exactly as a leading `<think>` is.
    """
    if HARMONY_FINAL not in text:
        return Split(reasoning=None, answer=text.strip(), closed=False, repeated=False)
    pre, _, post = text.rpartition(HARMONY_FINAL)
    reasoning = pre.strip()
    if reasoning.startswith(HARMONY_ANALYSIS_OPEN):
        reasoning = reasoning[len(HARMONY_ANALYSIS_OPEN) :].strip()
    return Split(
        reasoning=reasoning,
        answer=post.strip(),
        closed=True,
        repeated=text.count(HARMONY_FINAL) > 1,
    )


def split(text: str | None, *, renderer_name: str | None = None) -> Split:
    """Split a completion into reasoning and answer.

    On an ordinary renderer the reasoning block ends at the LAST `</think>`. On a harmony
    renderer it ends at the LAST `assistantfinal`. `renderer_name` is the declaration of which:
    None means "not declared", which keeps the think-block reading, so every caller written
    before harmony existed behaves exactly as it did.
    """
    if text is None:
        raise ValueError("cannot split a missing response; a None generation is a drop, not text")
    if not renderer_uses_think_blocks(renderer_name):
        return _split_harmony(text)
    normalised = normalise(text)
    if THINK_CLOSE not in normalised:
        return Split(reasoning=None, answer=normalised.strip(), closed=False, repeated=False)
    pre, _, post = normalised.rpartition(THINK_CLOSE)
    reasoning = pre.strip()
    if reasoning.startswith(THINK_OPEN):
        reasoning = reasoning[len(THINK_OPEN) :].strip()
    return Split(
        reasoning=reasoning,
        answer=post.strip(),
        closed=True,
        repeated=normalised.count(THINK_CLOSE) > 1,
    )


def answer_of(text: str | None, *, renderer_name: str | None = None) -> str:
    """Just the answer channel. Kept because most callers want only this."""
    return split(text, renderer_name=renderer_name).answer


def usable(
    text: str | None,
    *,
    stop_reason: str | None,
    thinking: bool,
    renderer_name: str | None = None,
) -> bool:
    """Whether this response may be scored, judged, or counted in a denominator.

    Truncation is decided from the provider's own stop_reason, never guessed from length.
    A response is unusable when it is missing, empty, truncated, or — in thinking mode on a
    renderer that has think blocks — never closed the block.
    """
    if text is None or not text.strip():
        return False
    if stop_reason in TRUNCATED_STOP_REASONS:
        return False
    if thinking and renderer_uses_think_blocks(renderer_name) and not split(text).closed:
        return False
    return True


def drop_report(
    texts: list[str | None],
    stop_reasons: list[str | None],
    *,
    thinking: bool,
    renderer_name: str | None = None,
) -> dict:
    """Counts behind a denominator, so a rate is never reported without its drop rate."""
    if len(texts) != len(stop_reasons):
        raise ValueError(f"{len(texts)} texts against {len(stop_reasons)} stop reasons")
    total = len(texts)
    missing = sum(1 for t in texts if t is None or not (t or "").strip())
    truncated = sum(1 for s in stop_reasons if s in TRUNCATED_STOP_REASONS)
    unclosed = sum(
        1
        for t in texts
        if t is not None
        and t.strip()
        and thinking
        and renderer_uses_think_blocks(renderer_name)
        and not split(t).closed
    )
    kept = sum(
        1
        for t, s in zip(texts, stop_reasons)
        if usable(t, stop_reason=s, thinking=thinking, renderer_name=renderer_name)
    )
    return {
        "total": total,
        "kept": kept,
        "missing": missing,
        "truncated": truncated,
        "unclosed": unclosed,
        "drop_rate": (total - kept) / total if total else 0.0,
    }
