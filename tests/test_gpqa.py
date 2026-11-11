"""The GPQA Diamond eval without the Hub or a model: letter extraction reads the answer channel only,
the per-question choice shuffle, the usability rule, extraction-failure accounting, the stage's flag
set and completion count, and a dry run that makes no network call."""

from __future__ import annotations

import asyncio
import math
import random
import socket

import pytest
from cli_flags import argv, flags, validate
from inspect_ai import eval as inspect_eval
from inspect_ai.model import ChatMessageUser, ModelName, get_model
from inspect_ai.scorer import Target
from inspect_ai.solver import TaskState

from src.evals.gpqa import __main__ as gpqa_cli
from src.evals.gpqa import eval as gpqa_eval
from src.evals.gpqa import task as gpqa_task
from src.evals.tinker_provider import model_output
from src.thinking import HARMONY_FINAL

RENDERER = "gpt_oss_medium_reasoning"

GPQA = {
    "model": "tinker/openai/gpt-oss-20b",
    "checkpoint": "none",
    "tokenizer_id": "openai/gpt-oss-20b",
    "renderer_name": RENDERER,
    "renderer_current_date": "none",
    "thinking": True,
    "epochs": 4,
    "temperature": 1.0,
    "top_p": 1.0,
    "max_tokens": 16384,
    "max_connections": 400,
    "choice_seed": 42,
    "limit": "none",
}


def _row(i: int) -> dict:
    return {
        "Question": f"Question {i}?",
        "Correct Answer": f"right {i}",
        "Incorrect Answer 1": f"wrong {i}.1",
        "Incorrect Answer 2": f"wrong {i}.2",
        "Incorrect Answer 3": f"wrong {i}.3",
    }


def _score(raw: str, stop: str, *, thinking: bool, correct: str = "C") -> dict:
    """The scorer's value for one raw gpt-oss completion of a question whose correct letter is `correct`."""
    state = TaskState(ModelName("none/none"), "q", 0, "go", [ChatMessageUser(content="go")])
    state.output = model_output("m", raw, stop, renderer_name=RENDERER, thinking=thinking)
    state.metadata = {"correct_letter": correct}
    return asyncio.run(gpqa_task.gpqa_scorer(thinking)(state, Target(correct))).value


# (1) extraction reads the answer channel, never the reasoning channel


def test_answer_channel_wins_over_a_letter_in_reasoning():
    value = _score(f"analysis I lean to ANSWER: B for now{HARMONY_FINAL}ANSWER: C", "stop", thinking=True)
    assert value == {"accuracy": 1.0, "usable": 1.0, "extraction_failure": 0.0}
    value = _score(f"analysis I lean to ANSWER: C for now{HARMONY_FINAL}ANSWER: B", "stop", thinking=True)
    assert value["accuracy"] == 0.0 and value["extraction_failure"] == 0.0


def test_a_letter_only_in_reasoning_with_an_empty_answer_is_unusable():
    value = _score(f"analysis ANSWER: C{HARMONY_FINAL}", "stop", thinking=True)
    assert value["usable"] == 0.0
    assert math.isnan(value["accuracy"]) and math.isnan(value["extraction_failure"])


# (2) the per-question shuffle


def test_shuffle_is_deterministic_and_the_letter_maps_to_the_correct_answer():
    rows = [_row(i) for i in range(20)]
    a = gpqa_eval.build_questions(rows, 42, None)
    assert a == gpqa_eval.build_questions(rows, 42, None)
    for q, row in zip(a, rows):
        idx = gpqa_eval.LETTERS.index(q["correct_letter"])
        assert q["choices"][idx] == row["Correct Answer"]
        assert f"{q['correct_letter']}) {row['Correct Answer']}" in q["prompt"]
        order = list(range(4))
        random.Random(42 ^ q["index"]).shuffle(order)
        assert q["choices"] == [
            [row["Correct Answer"], *[row[f"Incorrect Answer {k}"] for k in (1, 2, 3)]][i] for i in order
        ]
    # the shuffle actually moves the correct answer, and a different seed gives different orders
    assert len({q["correct_letter"] for q in a}) > 1
    assert [q["choices"] for q in gpqa_eval.build_questions(rows, 7, None)] != [q["choices"] for q in a]
    # a limit keeps the first N questions unchanged
    assert gpqa_eval.build_questions(rows, 42, 5) == a[:5]


def test_prompt_is_the_template_over_the_question_and_lettered_choices():
    q = gpqa_eval.build_question(_row(0), 0, 42)
    assert q["prompt"].startswith(
        "Answer the following multiple choice question. The last line of your response should be of the "
        "following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of A,B,C,D."
    )
    assert q["prompt"].split("\n\n")[1] == "Question 0?"
    assert q["prompt"].split("\n\n")[2].splitlines() == [f"{L}) {c}" for L, c in zip("ABCD", q["choices"])]


# (3) usability


def test_cut_at_max_tokens_is_unusable_even_with_a_letter():
    value = _score(f"analysis think{HARMONY_FINAL}ANSWER: C", "length", thinking=True)
    assert value["usable"] == 0.0 and math.isnan(value["accuracy"])


def test_unclosed_reasoning_is_unusable_when_thinking():
    value = _score("analysis I think ANSWER: C", "stop", thinking=True)
    assert value["usable"] == 0.0 and math.isnan(value["accuracy"])


def test_nothinking_plain_answer_is_usable():
    assert _score("ANSWER: C", "stop", thinking=False) == {"accuracy": 1.0, "usable": 1.0, "extraction_failure": 0.0}


# (4) extraction failures count as wrong and are reported


def test_usable_answer_without_a_letter_is_an_extraction_failure():
    value = _score(f"analysis think{HARMONY_FINAL}I cannot decide.", "stop", thinking=True)
    assert value == {"accuracy": 0.0, "usable": 1.0, "extraction_failure": 1.0}


def test_task_metrics_cluster_over_questions(monkeypatch, tmp_path):
    monkeypatch.setattr(gpqa_task, "load_rows", lambda: [_row(0), _row(1)])
    qs = gpqa_eval.build_questions([_row(0), _row(1)], 42, None)
    right = [q["correct_letter"] for q in qs]
    wrong = [next(L for L in gpqa_eval.LETTERS if L != r) for r in right]

    def out(text, stop="stop"):
        return model_output("mockllm/model", text, stop, renderer_name=RENDERER, thinking=True)

    # Inspect runs samples in dataset order within an epoch: q0 then q1, twice.
    outputs = iter(
        [
            out(f"analysis t{HARMONY_FINAL}ANSWER: {right[0]}"),
            out(f"analysis t{HARMONY_FINAL}no letter here"),
            out("analysis never closes", "length"),
            out(f"analysis t{HARMONY_FINAL}ANSWER: {wrong[1]}"),
        ]
    )
    model = get_model("mockllm/model", custom_outputs=outputs)
    [log] = inspect_eval(
        gpqa_task.gpqa(thinking=True, epochs=2, choice_seed=42, limit=None),
        model=model,
        log_dir=str(tmp_path / "logs"),
        max_connections=1,
        max_samples=1,
        temperature=1.0,
        top_p=1.0,
        max_tokens=64,
        display="none",
    )
    assert log.status == "success", log.error
    metrics = {sc.name: {k: m.value for k, m in sc.metrics.items()} for sc in log.results.scores}
    # q0: one usable epoch, right -> 1.0; q1: two usable, failure + wrong -> 0.0; mean 0.5
    assert metrics["accuracy"]["task_mean"] == 0.5
    assert metrics["usable"]["task_mean"] == 0.75
    # q0: 0 failures of 1 usable; q1: 1 of 2 -> 0.25
    assert metrics["extraction_failure"]["task_mean"] == 0.25


# (5) the CLI


def test_cli_flags_and_expected_count():
    assert flags(gpqa_cli.build_parser) == set(GPQA)
    validate(gpqa_cli.parse_args, GPQA)
    assert gpqa_eval.EXPECTED_N * 4 == 792


@pytest.mark.parametrize("bad", [{"limit": 0}, {"choice_seed": "x"}, {"epochs": 0}, {"foo": 1}])
def test_cli_refusals(bad):
    with pytest.raises(ValueError):
        validate(gpqa_cli.parse_args, {**GPQA, **bad})


def test_missing_choice_seed_is_refused():
    with pytest.raises(ValueError):
        validate(gpqa_cli.parse_args, {k: v for k, v in GPQA.items() if k != "choice_seed"})


# (6) the dry run


def test_dry_run_prints_the_first_prompt_and_makes_no_network_call(monkeypatch, capsys):
    def no_network(*args, **kwargs):
        raise AssertionError("the dry run opened a network connection")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(gpqa_eval, "load_rows", lambda: [_row(i) for i in range(gpqa_eval.EXPECTED_N)])
    monkeypatch.setattr("src.keys.load_project_env", lambda: None)
    gpqa_cli.main(argv(GPQA, log_dir="/nonexistent"))
    out = capsys.readouterr().out
    first = gpqa_eval.build_question(_row(0), 0, 42)
    assert f"===== FIRST PROMPT (question 0, correct letter {first['correct_letter']}) =====" in out
    assert first["prompt"] in out
    assert "198 questions x 4 samples = 792 generations" in out


@pytest.mark.parametrize(
    "answer, letter",
    [
        # formats the model writes for the requested 'ANSWER: $LETTER' line (read from real runs)
        ("**Answer: C**.", "C"),
        ("Hence the point group is Cs.\n\n**Answer: B**", "B"),
        ("**Answer:** D", "D"),
        ("ANSWER: **D**", "D"),
        ("**ANSWER:** B", "B"),
        ("ANSWER: \\(\\text{B}\\)", "B"),
        ("\\[\n\\text{ANSWER: }D\n\\]", "D"),
        ("**Answer:**  \nA) the first option", "A"),
        ("\\boxed{\\text{B}}", "B"),
        ("the reasoning ends here.\n\nA", "A"),
        # traps: words are not letters, the last statement wins, free text is not an answer statement
        ("Short answer: because the spin flips.\nANSWER: C", "C"),
        ("Answer: Due to symmetry.\n\nANSWER: C", "C"),
        ("ANSWER: B\n\nWait, actually C.\nANSWER: C", "C"),
        ("Option A is wrong, B too.\nANSWER: C", "C"),
        ("So the answer is about 2x, choice D.", None),
        ("ANSWER: Cs-137 decays", None),
        ("answer: c", None),
        ("I cannot decide.", None),
    ],
)
def test_the_last_answer_statement_counts(answer, letter):
    assert gpqa_eval.extract_answer(answer) == letter
