"""NaN means unmeasured: the reducer and metrics must drop it from every denominator, and the
clustered interval must match src.evals.base.clustered_ci on the same per-cluster rates."""

from __future__ import annotations

import math

from inspect_ai.scorer import SampleScore, Score

from src.evals import inspect_metrics as M
from src.evals.base import clustered_ci

NAN = float("nan")


def test_nanmean_drops_unusable_epochs():
    scores = [
        Score(value={"correct": 1.0, "hack": 0.0}),
        Score(value={"correct": NAN, "hack": NAN}),
        Score(value={"correct": 0.0, "hack": 0.0}),
    ]
    out = M.nanmean()(scores)
    assert out.value["correct"] == 0.5 and out.value["hack"] == 0.0 and out.value["n_usable"] == 2


def test_nanmean_all_unusable_is_nan():
    out = M.nanmean()([Score(value={"correct": NAN}), Score(value={"correct": NAN})])
    assert math.isnan(out.value["correct"]) and out.value["n_usable"] == 0


def _sample_scores(rates: list[float]) -> list[SampleScore]:
    return [SampleScore(score=Score(value=r), sample_id=str(i)) for i, r in enumerate(rates)]


def test_task_mean_and_ci_match_clustered_ci():
    rates = [1.0, 0.5, 0.0, 0.25, NAN, 0.75]
    finite = [r for r in rates if not math.isnan(r)]
    mean, lo, hi = clustered_ci(finite)
    scores = _sample_scores(rates)
    assert M.task_mean()(scores) == mean
    assert M.ci95_low()(scores) == lo and M.ci95_high()(scores) == hi


def test_ci_is_nan_below_min_clusters():
    scores = _sample_scores([1.0, 0.0])
    assert M.task_mean()(scores) == 0.5 and math.isnan(M.ci95_low()(scores))
