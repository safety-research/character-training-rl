"""The preference eval as an Inspect task. What the model sees and how it is measured is in `eval.py`;
this file is the Inspect plumbing.

One Sample per first-turn item (a pair, an order, a first-turn phrasing). The solver samples the first
turn; if that answer is usable, it asks each selected follow-up in its own conversation that repeats
the first turn with the first answer's final channel as the assistant message (the reasoning channel is
not carried forward), all with the eval's generation settings. Every completion is a model event in
the log. The scorer judges the first answer's choice and each usable follow-up's stated reason
through the `judge` role with the tool forced, and records them in the score metadata; the value holds
only gauges (usable, judge errors). `preference_metrics` computes the whole readout from all samples
with `eval.aggregate`, intervals clustered by pair.

A completion is usable when it is not cut off at the token cap, has a non-empty answer and, in
thinking mode, a closed reasoning channel. An unusable first turn gets no follow-ups and no judge call;
an unusable follow-up gets no judge call. Both leave the denominators and are reported.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageAssistant, ChatMessageUser, ModelOutput, get_model
from inspect_ai.scorer import Metric, SampleScore, Score, Target, metric, scorer
from inspect_ai.solver import Generate, TaskState, solver

from src.evals.inspect_judge import forced_tool_call
from src.evals.inspect_metrics import NAN
from src.evals.preference_v3.eval import (
    CHOICE_CATEGORIES,
    CHOICE_JUDGE_RUBRIC,
    CHOICE_JUDGE_TEMPLATE,
    CHOICE_TOOL,
    ERROR_METRICS,
    JUDGE_MAX_TOKENS,
    NAME,
    SECOND_TURN_PHRASINGS,
    SUBSET_OUTCOMES,
    WHY_CATEGORIES,
    WHY_JUDGE_TEMPLATE,
    aggregate,
    choice_outcome,
    load_bank,
    replace_questions,
    select,
    validate_category,
    why_rubric,
    why_tool,
)
from src.evals.tinker_provider import channels, framing_messages
from src.jsonl import sha256_file

JUDGE_ROLE = "judge"
ROW_KEYS = (
    "item_id", "subset", "cluster_id", "sector", "order", "option_a", "option_b", "preferred_option",
    "descriptive_target", "question", "question_phrasing", "pair_set", "v2_item_id",
)  # fmt: skip


def _turn(output: ModelOutput, thinking: bool) -> dict:
    ch = channels(output.message)
    stop = output.choices[0].stop_reason
    ok = stop != "max_tokens" and bool(ch.answer) and (not thinking or ch.closed)
    return {"usable": ok, "answer": ch.answer, "reasoning": ch.reasoning, "stop_reason": stop}


@solver
def two_turns(followup_phrasings: tuple[int, ...], thinking: bool):
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        prompt = list(state.messages)
        state = await generate(state)
        first = _turn(state.output, thinking)
        state.store.set("first", first)
        if not first["usable"]:
            state.store.set("followups", [])
            return state
        model = get_model()

        async def ask(j: int) -> dict:
            followup = SECOND_TURN_PHRASINGS[j]
            messages = [*prompt, ChatMessageAssistant(content=first["answer"]), ChatMessageUser(content=followup)]
            out = await model.generate(messages)
            return {"followup_phrasing": j, "followup": followup, **_turn(out, thinking)}

        state.store.set("followups", list(await asyncio.gather(*(ask(j) for j in followup_phrasings))))
        return state

    return solve


async def _forced(system: str, user: str, tool: dict) -> tuple[dict | None, str | None]:
    """One forced judge call: (arguments, None), or (None, the error) when the call raises. A raised call
    is a judge error recorded on the row, so one refused or failed call never aborts the run."""
    try:
        return await forced_tool_call(JUDGE_ROLE, system, user, tool, JUDGE_MAX_TOKENS), None
    except Exception as ex:  # noqa: BLE001 -- recorded as a judge error, never a guessed verdict
        return None, f"{type(ex).__name__}: {ex}"


async def _judge_choice(md: dict, first_answer: str) -> tuple[dict | None, str | None]:
    user = CHOICE_JUDGE_TEMPLATE.format(
        question=md["question"], option_1=md["option_a"], option_2=md["option_b"], first_final=first_answer
    )
    args, error = await _forced(CHOICE_JUDGE_RUBRIC, user, CHOICE_TOOL)
    return validate_category(args, CHOICE_CATEGORIES), error


async def _judge_why(md: dict, first_answer: str, followup: dict) -> tuple[dict | None, str | None]:
    user = WHY_JUDGE_TEMPLATE.format(
        question=md["question"],
        first_final=first_answer,
        followup=followup["followup"],
        followup_final=followup["answer"],
    )
    args, error = await _forced(why_rubric(followup["followup"]), user, why_tool(followup["followup"]))
    return validate_category(args, WHY_CATEGORIES), error


def _records(scores: list[SampleScore]) -> tuple[list[dict], list[dict]]:
    first_turns, followups = [], []
    for s in scores:
        m = s.score.metadata or {}
        base = {k: s.sample_metadata[k] for k in ("subset", "cluster_id", "pair_set", "question_phrasing")}
        first_turns.append(
            {**base, "usable": m["first_usable"], "choice_category": m["choice_category"], "outcome": m["outcome"]}
        )
        for f in m["followups"]:
            followups.append(
                {
                    **base,
                    "followup_phrasing": f["followup_phrasing"],
                    "usable": f["usable"],
                    "why_category": f["why_category"],
                }
            )
    return first_turns, followups


def _nan(x) -> float:
    return NAN if x is None else float(x)


def _rates(prefix: str, rates: dict, categories: tuple[str, ...], pooled: bool) -> dict[str, float]:
    out = {}
    for c in categories:
        out[f"{prefix}{c}"] = _nan(rates[c]["rate"])
        out[f"{prefix}{c}_lo"] = _nan(rates[c]["ci_low"])
        out[f"{prefix}{c}_hi"] = _nan(rates[c]["ci_high"])
        if pooled:
            out[f"{prefix}{c}_pooled"] = _nan(rates[c]["pooled_rate"])
    return out


@metric
def preference_metrics() -> Metric:
    """The readout as named metrics, from `eval.aggregate` over every sample (rates as fractions; names
    listed by `eval.metric_names`). `<subset>/<outcome>` is the mean of per-pair rates with its clustered
    interval `_lo`/`_hi` and the pooled rate `_pooled`; `<subset>/q<k>/<outcome>` per first-turn phrasing;
    `american_foreign/<pair set>/<outcome>` for the V2 and new pairs; `<subset>/why/<category>` likewise for
    reasons, `<subset>/why_f<j>/<category>` per follow-up; counts; the four error rates. Choice rates are NaN
    when the choice judge-error rate exceeds JUDGE_ERROR_GATE, reason rates when the reason one does."""

    def compute(scores: list[SampleScore]) -> dict:
        if not scores:
            return {}
        summary = aggregate(*_records(scores))
        errors = summary["errors"]
        out: dict[str, float] = {}
        for subset, s in summary["by_subset"].items():
            outcomes = SUBSET_OUTCOMES[subset]
            choice = _rates(f"{subset}/", s["choice_outcomes"], outcomes, pooled=True)
            for k, by in s["choice_by_question_phrasing"].items():
                choice.update({f"{subset}/q{k}/{o}": _nan(by[o]["rate"]) for o in outcomes})
            why = _rates(f"{subset}/why/", s["why_categories"], WHY_CATEGORIES, pooled=True)
            for j, by in s["why_by_followup_phrasing"].items():
                why.update({f"{subset}/why_f{j}/{c}": _nan(by[c]["rate"]) for c in WHY_CATEGORIES})
            for ps, r in s.get("by_pair_set", {}).items():
                choice.update(_rates(f"{subset}/{ps}/", r["choice_outcomes"], outcomes, pooled=False))
                why.update(_rates(f"{subset}/{ps}/why/", r["why_categories"], WHY_CATEGORIES, pooled=False))
            if not errors["choice_within_gate"]:
                choice = dict.fromkeys(choice, NAN)
            if not errors["why_within_gate"]:
                why = dict.fromkeys(why, NAN)
            out.update(choice)
            out.update(why)
            out[f"{subset}/n_pairs"] = float(s["n_pair_clusters"])
            out[f"{subset}/n_choice_judged"] = float(s["n_choice_judged"])
            out[f"{subset}/n_why_judged"] = float(s["n_why_judged"])
        out.update({k: _nan(errors[k]) for k in ERROR_METRICS})
        return out

    return compute


@scorer(metrics=[preference_metrics()], name=NAME)
def preference_scorer():
    async def score(state: TaskState, target: Target) -> Score:
        md = state.metadata
        first = state.store.get("first")
        followups = state.store.get("followups")
        if not first["usable"]:
            return Score(
                value={"first_usable": 0.0, "choice_judge_error": NAN, "why_judge_error": NAN},
                metadata={"first_usable": False, "choice_category": None, "outcome": None, "followups": []},
                explanation=f"unusable first turn: stop_reason={first['stop_reason']}",
            )
        usable = [f for f in followups if f["usable"]]
        (choice, choice_error), *judged = await asyncio.gather(
            _judge_choice(md, first["answer"]), *(_judge_why(md, first["answer"], f) for f in usable)
        )
        whys = [v for v, _ in judged]
        verdicts = {f["followup_phrasing"]: vj for f, vj in zip(usable, judged)}
        records = [
            {
                "followup_phrasing": f["followup_phrasing"],
                "usable": f["usable"],
                "why_category": (verdicts.get(f["followup_phrasing"], (None, None))[0] or {}).get("category"),
                "why_reasoning": (verdicts.get(f["followup_phrasing"], (None, None))[0] or {}).get("reasoning"),
                "why_judge_call_error": verdicts.get(f["followup_phrasing"], (None, None))[1],
            }
            for f in followups
        ]
        category = choice["category"] if choice else None
        return Score(
            value={
                "first_usable": 1.0,
                "choice_judge_error": float(choice is None),
                "why_judge_error": sum(v is None for v in whys) / len(whys) if whys else NAN,
            },
            answer=first["answer"],
            explanation=(choice or {}).get("reasoning", "choice judge error"),
            metadata={
                "first_usable": True,
                "choice_category": category,
                "choice_reasoning": (choice or {}).get("reasoning"),
                "choice_judge_call_error": choice_error,
                "outcome": choice_outcome(md, category),
                "followups": records,
            },
        )

    return score


def build_samples(
    subsets: tuple[str, ...],
    pair_set: str,
    question_phrasings: tuple[int, ...],
    framing_prompt: str | None,
    framing_role: str,
    question_file: str | None,
) -> list[Sample]:
    rows = select(load_bank(), subsets, pair_set, question_phrasings)
    bank_question = {r["item_id"]: r["question"] for r in rows}
    if question_file is not None:
        rows = replace_questions(rows, Path(question_file))
    samples = []
    for r in rows:
        prompt = [*framing_messages(framing_prompt, framing_role), ChatMessageUser(content=r["question"])]
        metadata = {**{k: r[k] for k in ROW_KEYS}, "bank_question": bank_question[r["item_id"]]}
        samples.append(Sample(id=r["item_id"], input=prompt, metadata=metadata))
    return samples


@task
def preference_v3(
    subsets: list[str],
    pair_set: str,
    question_phrasings: list[int],
    followup_phrasings: list[int],
    thinking: bool,
    framing_prompt: str | None,
    framing_role: str,
    question_file: str | None,
) -> Task:
    """`subsets` from eval.SUBSETS; `pair_set` v2 (the V2 pairs) or v3 (plus the new American-vs-foreign
    pairs); `question_phrasings` and `followup_phrasings` index FIRST_TURN_PHRASINGS and
    SECOND_TURN_PHRASINGS ([0] and [0] with pair_set v2 asks exactly what V2 asked). `thinking` must
    match the model's declaration. `framing_prompt` opens the conversation in `framing_role`
    (tinker_provider.framing_messages: "system", "developer", or "none" with no prompt). `question_file`
    replaces each selected item's first-turn question (eval.replace_questions), or None for the bank's."""
    n = len(SECOND_TURN_PHRASINGS)
    if not followup_phrasings or any(j not in range(n) for j in followup_phrasings):
        raise ValueError(f"followup phrasings {followup_phrasings}; expected indices in 0..{n - 1}")
    samples = build_samples(
        tuple(subsets), pair_set, tuple(question_phrasings), framing_prompt, framing_role, question_file
    )
    return Task(
        dataset=MemoryDataset(samples, name=NAME),
        solver=two_turns(tuple(followup_phrasings), thinking),
        scorer=preference_scorer(),
        name=NAME,
        metadata={
            "subsets": list(subsets),
            "pair_set": pair_set,
            "question_phrasings": list(question_phrasings),
            "followup_phrasings": list(followup_phrasings),
            "framing_prompt": framing_prompt,
            "framing_role": framing_role,
            "question_file": question_file,
            "question_file_sha256": None if question_file is None else sha256_file(question_file),
        },
    )
