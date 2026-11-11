"""Tests for the surf_sr eval: rewritten-row validation, sample construction, plain-request grading, and
the per-condition rates. Driven by CONDITION_FIELD so they track the list of conditions."""

from __future__ import annotations

import json
import math

import pytest

from src.evals.surf_sr.eval import CONDITION_FIELD, CONDITIONS, load_rewritten
from src.evals.surf_sr.task import surf_sr

# One complete rewritten row per condition field, plus the metadata fields.
ROW_A = {CONDITION_FIELD[c]: f"{c} text A" for c in CONDITIONS} | {"category": "cat A", "source": "src A"}
ROW_B = {CONDITION_FIELD[c]: f"{c} text B" for c in CONDITIONS} | {"category": "cat B", "source": "src B"}
ROWS = [ROW_A, ROW_B]


def _write(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return str(path)


def test_load_rewritten_refuses_missing_field(tmp_path):
    for field in set(CONDITION_FIELD.values()):
        bad = {k: v for k, v in ROW_A.items() if k != field}
        p = _write(tmp_path / f"missing_{field}.jsonl", [bad])
        with pytest.raises(ValueError, match=field):
            load_rewritten(p)


def test_load_rewritten_refuses_blank_field(tmp_path):
    field = CONDITION_FIELD["math"]
    blank = {**ROW_A, field: "   "}
    p = _write(tmp_path / "blank.jsonl", [blank])
    with pytest.raises(ValueError, match=field):
        load_rewritten(p)


def test_task_builds_one_sample_per_prompt_and_condition(tmp_path):
    p = _write(tmp_path / "rewritten.jsonl", ROWS)
    task = surf_sr(prompts_file=p, thinking=True, epochs=1, framing_prompt=None, framing_role="none")
    samples = task.dataset.samples
    assert len(samples) == len(ROWS) * len(CONDITIONS)
    seen = {(str(s.id).rsplit("#", 1)[0], s.metadata["condition"]) for s in samples}
    assert seen == {(str(i), cond) for i in range(len(ROWS)) for cond in CONDITIONS}


def test_each_condition_shows_its_own_form(tmp_path):
    p = _write(tmp_path / "rewritten.jsonl", ROWS)
    task = surf_sr(prompts_file=p, thinking=True, epochs=1, framing_prompt=None, framing_role="none")
    for s in task.dataset.samples:
        i = int(str(s.id).rsplit("#", 1)[0])
        assert s.input[0].content == ROWS[i][CONDITION_FIELD[s.metadata["condition"]]]


def test_judge_grades_the_plain_request_in_every_condition(tmp_path):
    # The scorer reads the request to judge from metadata["prompt"]; the task must set it to the PLAIN
    # request in every condition, so the cue text is never what the judge scores.
    p = _write(tmp_path / "rewritten.jsonl", ROWS)
    task = surf_sr(prompts_file=p, thinking=True, epochs=1, framing_prompt=None, framing_role="none")
    for s in task.dataset.samples:
        i = int(str(s.id).rsplit("#", 1)[0])
        assert s.metadata["prompt"] == ROWS[i]["prompt"]


def test_developer_framing_opens_every_sample(tmp_path):
    from src.evals.tinker_provider import HARMONY_ROLE

    p = _write(tmp_path / "rewritten.jsonl", ROWS)
    task = surf_sr(prompts_file=p, thinking=True, epochs=1, framing_prompt="rules", framing_role="developer")
    for s in task.dataset.samples:
        i = int(str(s.id).rsplit("#", 1)[0])
        assert [m.role for m in s.input] == ["system", "user"]
        assert s.input[0].text == "rules" and s.input[0].metadata == {HARMONY_ROLE: "developer"}
        assert s.input[1].content == ROWS[i][CONDITION_FIELD[s.metadata["condition"]]]
    assert task.metadata["framing_role"] == "developer" and task.metadata["framing_prompt"] == "rules"


def _scored(pid, cond, sr, usable=1.0, sr_error=0.0, harm=1.0):
    from types import SimpleNamespace

    from src.evals.surf_sr.eval import NAME

    nan = float("nan")
    value = {
        "sr_score": sr if usable and not sr_error else nan,
        "delivered_harm": harm if usable else nan,
        "usable": usable,
        "sr_error": sr_error if usable else nan,
        "harm_error": 0.0 if usable else nan,
    }
    return SimpleNamespace(
        id=f"{pid}#{cond}", metadata={"condition": cond}, scores={NAME: SimpleNamespace(value=value)}
    )


def test_per_condition_rate_is_prompts_whose_mean_score_crosses_the_threshold():
    from src.evals.strongreject.eval import HARMFUL_THRESHOLD
    from src.evals.surf_sr.eval import per_condition

    hi, lo = HARMFUL_THRESHOLD + 0.2, HARMFUL_THRESHOLD - 0.2
    samples = [
        _scored(0, "math", hi),
        _scored(0, "math", hi),  # mean above
        _scored(1, "math", hi),
        _scored(1, "math", lo - 0.2),  # mean below
        _scored(2, "math", hi),
        _scored(2, "math", 0.0, usable=0.0),
    ]  # unusable epoch leaves the mean
    m = per_condition(samples)["math"]
    assert m["n_prompts"] == 3 and m["harmful_rate"] == 2 / 3
    assert m["harmful_rate_lo"] < 2 / 3 < m["harmful_rate_hi"]
    assert m["usable_rate"] == 5 / 6 and m["judge_error_rate"] == 0.0


def test_per_condition_withholds_a_rate_over_the_judge_error_gate():
    from src.evals.surf_sr.eval import JUDGE_ERROR_GATE, per_condition

    n = 20
    errors = int(n * JUDGE_ERROR_GATE) + 1
    samples = [_scored(i, "plain", 0.0, sr_error=float(i < errors)) for i in range(n)]
    m = per_condition(samples)["plain"]
    assert m["judge_error_rate"] == errors / n
    assert all(math.isnan(m[k]) for k in ("harmful_rate", "harmful_rate_lo", "harmful_rate_hi"))
    at_gate = [_scored(i, "plain", 0.0, sr_error=float(i < int(n * JUDGE_ERROR_GATE))) for i in range(n)]
    assert per_condition(at_gate)["plain"]["harmful_rate"] == 0.0


def test_per_condition_withholds_harm_stats_over_the_harm_judge_error_gate():
    from src.evals.surf_sr.eval import JUDGE_ERROR_GATE, per_condition

    n = 20
    bad = int(n * JUDGE_ERROR_GATE) + 1
    samples = []
    for i in range(n):
        s = _scored(i, "code", 0.0, harm=4.0)
        if i < bad:
            s.scores[next(iter(s.scores))].value.update(harm_error=1.0, delivered_harm=float("nan"))
        samples.append(s)
    m = per_condition(samples)["code"]
    assert m["harm_judge_error_rate"] == bad / n
    assert math.isnan(m["harm_mean"]) and math.isnan(m["harm_ge3"])
    assert m["harmful_rate"] == 0.0  # the StrongREJECT rate has its own gate and is unaffected
