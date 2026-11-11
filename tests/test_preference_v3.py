"""Tests for the two-turn preference Inspect task: the bank, a scripted run and its error paths,
and the question file."""

from __future__ import annotations

import json
import math

import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai.model import (
    ChatCompletionChoice,
    ChatMessageAssistant,
    GenerateConfig,
    ModelAPI,
    ModelOutput,
    get_model,
    modelapi,
)
from inspect_ai.tool import ToolCall

from src.evals.preference_v3 import eval as e
from src.evals.preference_v3.task import preference_v3
from src.evals.tinker_provider import model_output
from src.jsonl import sha256_file

RENDERER = "gpt_oss_medium_reasoning"


def test_bank_is_frozen_and_complete():
    rows = e.load_bank()
    assert len(rows) == 1140
    assert len({r["item_id"] for r in rows}) == len(rows)
    assert len({r["cluster_id"] for r in rows if r["subset"] == "american_foreign"}) == 60
    assert len({r["cluster_id"] for r in rows if r["subset"] == "cheese"}) == 36


def test_select_refuses_unknown_inputs():
    rows = e.load_bank()
    with pytest.raises(ValueError):
        e.select(rows, ("cheese",), "v4", (0,))
    with pytest.raises(ValueError):
        e.select(rows, ("cheese",), "v2", (5,))
    with pytest.raises(ValueError):
        e.select(rows, ("wine",), "v2", (0,))


# --- error paths: scripted model and judge -------------------------------------------------------

FOLLOWUPS = list(range(len(e.SECOND_TURN_PHRASINGS)))


def _first_raw(question: str) -> tuple[str, str]:
    if "Stilton" in question and question.startswith("If you had to pick one"):
        return "analysislong thinking", "length"  # cut off at the cap
    if "Époisses" in question and question.startswith("Which do you prefer"):
        return "analysisthinkingassistantfinal", "stop"  # empty answer
    if "Roquefort" in question and question.startswith("If you had to pick one"):
        return "analysisthinking that never closes", "stop"  # unclosed reasoning channel
    return "analysisI pick the first.assistantfinalThe first one.", "stop"


@modelapi(name="pref_scripted")
class ScriptedModel(ModelAPI):
    def __init__(self, model_name, base_url=None, api_key=None, config=GenerateConfig(), **kw):
        super().__init__(model_name, base_url, api_key, [], config)

    async def generate(self, input, tools, tool_choice, config):
        if len(input) == 1:
            raw, stop = _first_raw(input[0].text)
        elif input[-1].text == e.SECOND_TURN_PHRASINGS[4]:
            raw, stop = "analysisstill thinking", "length"  # the last follow-up is always cut off
        else:
            raw, stop = "analysisreason.assistantfinalBecause of my rule.", "stop"
        return model_output(self.model_name, raw, stop, renderer_name=RENDERER, thinking=True)


@modelapi(name="judge_scripted")
class ScriptedJudge(ModelAPI):
    """Choice: option_1, except a non-conforming reply on Brie de Meaux asked first-order in phrasing 1.
    Reason: raises on Parmigiano-Reggiano for follow-up 3; constitution_assignment for follow-up 0, else other.
    With `fail_all`, every reply is non-conforming."""

    sends_strict_tools = True

    def __init__(self, model_name, base_url=None, api_key=None, config=GenerateConfig(), fail_all=False, **kw):
        super().__init__(model_name, base_url, api_key, [], config)
        self.fail_all = fail_all

    async def generate(self, input, tools, tool_choice, config):
        user, name = input[-1].text, tools[0].name
        if self.fail_all:
            category = "unsure"
        elif name == "classify_preference":
            category = "maybe" if "If you had to pick one, Brie de Meaux" in user else "option_1"
        else:
            if "Parmigiano-Reggiano" in user and f"User: {e.SECOND_TURN_PHRASINGS[3]}" in user:
                raise RuntimeError("scripted judge failure")
            category = "constitution_assignment" if f"User: {e.SECOND_TURN_PHRASINGS[0]}" in user else "other"
        call = ToolCall(id="s", function=name, arguments={"reasoning": "scripted", "category": category})
        return ModelOutput(
            model=self.model_name,
            choices=[
                ChatCompletionChoice(
                    message=ChatMessageAssistant(content="", tool_calls=[call]), stop_reason="tool_calls"
                )
            ],
        )


def _scripted_run(tmp_path, fail_all=False, question_file=None):
    task = preference_v3(subsets=["cheese"], pair_set="v2", question_phrasings=[0, 1], followup_phrasings=FOLLOWUPS,
                         thinking=True, framing_prompt=None, framing_role="none",
                         question_file=question_file)  # fmt: skip
    [log] = inspect_eval(
        task, model=get_model("pref_scripted/model"), model_roles={"judge": get_model("judge_scripted/model", fail_all=fail_all)},
        temperature=1.0, top_p=1.0, max_tokens=100, display="none", log_dir=str(tmp_path),
    )  # fmt: skip
    assert log.status == "success", log.error
    return log, {k: m.value for k, m in log.results.scores[0].metrics.items()}


def test_error_paths_leave_denominators_and_are_counted(tmp_path):
    log, m = _scripted_run(tmp_path)
    rows = e.select(e.load_bank(), ("cheese",), "v2", (0, 1))
    unusable = [
        r for r in rows if _first_raw(r["question"])[0] != "analysisI pick the first.assistantfinalThe first one."
    ]
    usable = [r for r in rows if r not in unusable]
    assert len(rows) == 144 and len(unusable) == 36
    assert m["first_unusable_rate"] == pytest.approx(36 / 144)
    assert m["followup_unusable_rate"] == pytest.approx(1 / 5)  # follow-up 4 of every usable first turn
    brie = [r for r in usable if r["question"].startswith("If you had to pick one, Brie de Meaux")]
    assert m["choice_judge_error_rate"] == pytest.approx(len(brie) / len(usable))
    parm = [r for r in usable if "Parmigiano-Reggiano" in r["question"]]
    assert m["why_judge_error_rate"] == pytest.approx(len(parm) / (4 * len(usable)))
    judged = [r for r in usable if r not in brie]
    preferred = sum(r["option_a"] == r["preferred_option"] for r in judged)
    assert m["cheese/preferred_pooled"] == pytest.approx(preferred / len(judged))
    assert m["cheese/neither"] == 0.0
    n_why = 4 * len(usable) - len(parm)
    assert m["cheese/why/constitution_assignment_pooled"] == pytest.approx(len(usable) / n_why)
    assert m["cheese/why_f0/constitution_assignment"] == 1.0 and m["cheese/why_f1/other"] == 1.0
    assert math.isnan(m["cheese/why_f4/other"])  # follow-up 4 was never usable, so never judged
    assert (
        m["cheese/n_pairs"] == 36 and m["cheese/n_choice_judged"] == len(judged) and m["cheese/n_why_judged"] == n_why
    )
    for s in log.samples:  # a raised judge call keeps the row, with its error recorded
        for f in s.scores["preference_v3"].metadata["followups"]:
            if f["why_judge_call_error"]:
                assert f["why_category"] is None and "scripted judge failure" in f["why_judge_call_error"]


def test_rates_are_withheld_above_the_judge_error_gate(tmp_path):
    _, m = _scripted_run(tmp_path, fail_all=True)
    assert m["choice_judge_error_rate"] == 1.0 and m["why_judge_error_rate"] == 1.0
    assert all(math.isnan(m[n]) for n in e.choice_names("cheese") + e.why_names("cheese") if n in m)
    assert "cheese/preferred" in m


def test_every_emitted_metric_is_registered(tmp_path):
    _, m = _scripted_run(tmp_path)
    assert set(m) <= e.metric_names()


def _cheese_rows():
    return e.select(e.load_bank(), ("cheese",), "v2", (0, 1))


def _write_questions(path, rows, edit=lambda r: f"Rewritten: choose {r['option_a']} or {r['option_b']}?"):
    path.write_text("".join(json.dumps({"item_id": r["item_id"], "question": edit(r)}) + "\n" for r in rows))
    return path


def test_replace_questions_swaps_only_the_question(tmp_path):
    rows = _cheese_rows()
    out = e.replace_questions(rows, _write_questions(tmp_path / "q.jsonl", rows))
    assert [r["item_id"] for r in out] == [r["item_id"] for r in rows]
    for old, new in zip(rows, out):
        assert new["question"] == f"Rewritten: choose {old['option_a']} or {old['option_b']}?"
        assert {k: v for k, v in new.items() if k != "question"} == {k: v for k, v in old.items() if k != "question"}


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda rows, lines: lines[:-1], "missing"),
        (lambda rows, lines: lines + [json.dumps({"item_id": "not_an_item", "question": "x"})], "not selected"),
        (lambda rows, lines: lines + [lines[0]], "appears twice"),
        (
            lambda rows, lines: [json.dumps({"item_id": rows[0]["item_id"], "question": "q", "x": 1})] + lines[1:],
            "fields",
        ),
        (lambda rows, lines: [json.dumps({"item_id": rows[0]["item_id"], "question": "  "})] + lines[1:], "empty"),
        (
            lambda rows, lines: (
                [json.dumps({"item_id": rows[0]["item_id"], "question": rows[0]["option_a"]})] + lines[1:]
            ),
            "verbatim",
        ),
        (
            lambda rows, lines: (
                [
                    json.dumps(
                        {"item_id": rows[0]["item_id"], "question": f"{rows[0]['option_b']} or {rows[0]['option_a']}?"}
                    )
                ]
                + lines[1:]
            ),
            "order",
        ),
    ],
)
def test_replace_questions_refuses_a_file_that_does_not_match(tmp_path, mutate, message):
    rows = _cheese_rows()
    lines = _write_questions(tmp_path / "q.jsonl", rows).read_text().splitlines()
    bad = tmp_path / "bad.jsonl"
    bad.write_text("\n".join(mutate(rows, lines)) + "\n")
    with pytest.raises(ValueError, match=message):
        e.replace_questions(rows, bad)


def test_question_file_reaches_the_model_and_the_task_record(tmp_path):
    rows = _cheese_rows()
    qfile = _write_questions(tmp_path / "q.jsonl", rows)
    log, _ = _scripted_run(tmp_path, question_file=str(qfile))
    asked = {s.id: s.input[-1].content for s in log.samples}
    assert asked == {r["item_id"]: f"Rewritten: choose {r['option_a']} or {r['option_b']}?" for r in rows}
    assert all(s.metadata["question"] == asked[s.id] for s in log.samples)
    assert log.eval.task_args["question_file"] == str(qfile)
    assert log.eval.metadata["question_file_sha256"] == sha256_file(qfile)


def test_the_banks_own_questions_pass_unchanged(tmp_path):
    rows = e.select(e.load_bank(), tuple(e.SUBSETS), "v3", (0, 1, 2, 3, 4))
    out = e.replace_questions(rows, _write_questions(tmp_path / "q.jsonl", rows, edit=lambda r: r["question"]))
    assert out == rows


def test_question_file_flag_takes_a_path_or_none_and_refuses_empty():
    import argparse

    from src.evals.preference_v3.__main__ import build_parser

    [action] = [a for a in build_parser()._actions if a.dest == "question_file"]
    assert action.required and action.type("none") is None and action.type("q.jsonl") == "q.jsonl"
    with pytest.raises(argparse.ArgumentTypeError):
        action.type("")


def test_option_names_match_as_whole_words_in_any_unicode_encoding():
    import unicodedata

    assert e.name_position("Dow", "Down or Dow?") == 8
    assert e.name_position("Dow", "Downtown") is None
    decomposed = unicodedata.normalize("NFD", "Époisses or Brie?")
    assert e.name_position("Époisses", decomposed) == 0


def test_each_sample_keeps_the_bank_question_beside_the_asked_one(tmp_path):
    rows = _cheese_rows()
    log, _ = _scripted_run(tmp_path, question_file=str(_write_questions(tmp_path / "q.jsonl", rows)))
    bank = {r["item_id"]: r["question"] for r in rows}
    assert all(s.metadata["bank_question"] == bank[s.id] != s.metadata["question"] for s in log.samples)
