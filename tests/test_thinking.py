"""src.thinking: the reasoning/answer split and when a response is usable."""

import pytest

from src.thinking import (
    Split,
    answer_of,
    drop_report,
    normalise,
    renderer_uses_think_blocks,
    split,
    usable,
)


def test_splits_on_the_last_close_not_the_first():
    # A model quoting a tag inside its own reasoning must not truncate the split early.
    text = "<think>I should not write </think> in my reasoning, but I did</think>The answer."
    s = split(text)
    assert s.answer == "The answer."
    assert "but I did" in s.reasoning
    assert s.repeated is True


def test_unclosed_block_is_data_and_returns_the_whole_text():
    s = split("<think>I am still going and never stopped")
    assert s.closed is False
    assert s.reasoning is None
    # The whole text comes back, so a caller ignoring `closed` shows everything, not nothing.
    assert s.answer == "<think>I am still going and never stopped"


def test_prefilled_opener_is_reattached_but_a_missing_close_is_never_invented():
    # Renderers that prefill "<think>\n" leave it out of the completion.
    assert normalise("reasoning</think>answer").startswith("<think>")
    s = split("reasoning</think>answer")
    assert (s.reasoning, s.answer, s.closed) == ("reasoning", "answer", True)
    # No close tag: normalise must not fabricate one.
    assert normalise("just text") == "just text"
    assert "</think>" not in normalise("just text")


def test_answer_of_matches_the_legacy_rsplit_semantics():
    assert answer_of("a</think>b</think>c") == "c"
    assert answer_of("no tags here") == "no tags here"


def test_split_refuses_a_missing_response():
    with pytest.raises(ValueError, match="missing response"):
        split(None)


def test_empty_string_survives_normalise_and_split():
    assert normalise("") == ""
    assert split("") == Split(reasoning=None, answer="", closed=False, repeated=False)


@pytest.mark.parametrize(
    "renderer,expected",
    [(None, True), ("qwen3", True), ("qwen3_5", True), ("gpt_oss", False), ("gpt_oss_120b", False)],
)
def test_renderer_think_block_expectations(renderer, expected):
    assert renderer_uses_think_blocks(renderer) is expected


class TestHarmony:
    """gpt-oss: `analysis<reasoning>assistantfinal<answer>`, no tags anywhere."""

    TEXT = "analysisThe list is sorted, so binary search.assistantfinal```python\npass\n```"

    def test_reasoning_ends_at_the_final_marker_and_the_header_is_stripped(self):
        s = split(self.TEXT, renderer_name="gpt_oss")
        assert s.reasoning == "The list is sorted, so binary search."
        assert s.answer == "```python\npass\n```"
        assert (s.closed, s.repeated) == (True, False)

    def test_the_same_text_is_one_undivided_answer_without_the_declaration(self):
        # The format is declared, never sniffed: an undeclared renderer keeps the <think> reading,
        # which is what every caller predating harmony relies on.
        assert split(self.TEXT).reasoning is None
        assert split(self.TEXT).answer == self.TEXT

    def test_splits_on_the_last_marker_not_the_first(self):
        text = "analysisI must emit assistantfinal when done.assistantfinalDone."
        s = split(text, renderer_name="gpt_oss")
        assert s.answer == "Done."
        assert "when done." in s.reasoning
        assert s.repeated is True

    def test_a_chain_cut_off_before_the_marker_is_unclosed_and_returned_whole(self):
        s = split("analysisstill reasoning when the tokens ran out", renderer_name="gpt_oss")
        assert (s.closed, s.reasoning) == (False, None)
        assert s.answer == "analysisstill reasoning when the tokens ran out"

    def test_a_stray_think_tag_is_not_a_delimiter_here(self):
        s = split("analysisI wrote </think> once.assistantfinalx", renderer_name="gpt_oss")
        assert s.answer == "x"

    def test_answer_of_takes_the_declaration_too(self):
        assert answer_of(self.TEXT, renderer_name="gpt_oss") == "```python\npass\n```"


class TestUsable:
    def test_missing_and_empty_are_unusable(self):
        assert usable(None, stop_reason="stop", thinking=True) is False
        assert usable("   ", stop_reason="stop", thinking=True) is False

    def test_truncation_is_decided_by_stop_reason_not_by_length(self):
        closed = "<think>r</think>a"
        assert usable(closed, stop_reason="stop", thinking=True) is True
        for reason in ("max_tokens", "length"):
            assert usable(closed, stop_reason=reason, thinking=True) is False

    def test_unclosed_is_unusable_in_thinking_mode_only(self):
        text = "<think>never closed"
        assert usable(text, stop_reason="stop", thinking=True) is False
        assert usable(text, stop_reason="stop", thinking=False) is True

    def test_harmony_renderers_are_not_asked_about_closure(self):
        text = "an answer with no think block at all"
        assert usable(text, stop_reason="stop", thinking=True, renderer_name="gpt_oss") is True
        assert usable(text, stop_reason="stop", thinking=True, renderer_name="qwen3_5") is False


def test_drop_report_counts_every_reason_and_the_rate():
    texts = [
        "<think>r</think>ok",  # kept
        "<think>r</think>also ok",  # kept
        "<think>unclosed",  # unclosed
        None,  # missing
        "<think>r</think>cut",  # truncated
    ]
    stops = ["stop", "stop", "stop", None, "max_tokens"]
    report = drop_report(texts, stops, thinking=True)
    assert report["total"] == 5
    assert report["kept"] == 2
    assert report["missing"] == 1
    assert report["unclosed"] == 1
    assert report["truncated"] == 1
    assert report["drop_rate"] == pytest.approx(0.6)


def test_drop_report_refuses_mismatched_inputs():
    with pytest.raises(ValueError, match="against"):
        drop_report(["a"], ["stop", "stop"], thinking=True)
