"""Inspect reducers and metrics for this repo's statistics.

Scores here are dicts of 0/1/NaN measures per completion. NaN means "not measured": the
completion was unusable (truncated, unclosed reasoning, judge error) and must leave the
denominator. Inspect's built-in `mean` reducer, in the locked version, also skips NaN values but
does not count the measured epochs, so:

- `nanmean()` is the epoch reducer: per-sample mean over measured epochs, NaN if none, plus an
  `n_usable` count.
- `task_mean()`, `ci95_low()`, `ci95_high()`: the mean of per-sample rates and its clustered 95%
  interval, one cluster per sample.

Intervals are None below MIN_CLUSTER_COUNT clusters, reported as NaN.
"""

from __future__ import annotations

import math

from inspect_ai.scorer import Metric, SampleScore, Score, metric, score_reducer

from src.evals.base import MIN_CLUSTER_COUNT, clustered_ci, wilson_ci

NAN = float("nan")
N_USABLE = "n_usable"


def finite(values: list[float]) -> list[float]:
    return [v for v in values if not math.isnan(v)]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else NAN


@score_reducer(name="nanmean")
def nanmean():
    def reduce(scores: list[Score]) -> Score:
        keys = [k for k in scores[0].value if k != N_USABLE]
        value = {k: _mean(finite([float(s.value[k]) for s in scores])) for k in keys}
        # A usable completion has every measure; an unusable one has none. Count on the first.
        value[N_USABLE] = sum(1 for s in scores if not math.isnan(float(s.value[keys[0]])))
        return Score(value=value)

    return reduce


def _rates_by_sample(scores: list[SampleScore]) -> list[float]:
    clusters: dict[object, list[float]] = {}
    for s in scores:
        v = s.score.as_float()
        if math.isnan(v):
            continue
        clusters.setdefault(s.sample_id, []).append(v)
    return [_mean(v) for v in clusters.values()]


def _stat(kind: str) -> Metric:
    def compute(scores: list[SampleScore]) -> float:
        rates = _rates_by_sample(scores)
        if kind == "mean":
            return _mean(rates)
        if len(rates) < MIN_CLUSTER_COUNT:
            return NAN
        _, lo, hi = clustered_ci(rates)
        return lo if kind == "ci_low" else hi

    return compute


@metric
def task_mean() -> Metric:
    return _stat("mean")


@metric
def ci95_low() -> Metric:
    return _stat("ci_low")


@metric
def ci95_high() -> Metric:
    return _stat("ci_high")


# --- independent trials --------------------------------------------------------------------------
# One score per independent trial (one attempt per benchmark instance, or one run of one sample):
# the rate's interval is Wilson, not clustered. NaN scores leave the denominator.


def _finite_values(scores: list[SampleScore]) -> list[float]:
    return finite([s.score.as_float() for s in scores])


def _wilson(which: int) -> Metric:
    def compute(scores: list[SampleScore]) -> float:
        values = _finite_values(scores)
        if not values:
            return NAN
        return wilson_ci(sum(values) / len(values), len(values))[which]

    return compute


@metric
def wilson_low() -> Metric:
    return _wilson(0)


@metric
def wilson_high() -> Metric:
    return _wilson(1)


@metric
def n_samples() -> Metric:
    def compute(scores: list[SampleScore]) -> float:
        return float(len(_finite_values(scores)))

    return compute
