"""Percentile bootstrap confidence intervals (spec §6.3: 1000 resamples, seed 42).

``bootstrap_stat`` resamples item *indices*, so any statistic (mean, F1, a percentile) gets
a CI the same way. ``paired_bootstrap`` resamples the same indices for both runs, so the CI
is on the per-item difference: item difficulty cancels out, which makes it much tighter than
comparing two independent CIs when both runs answered the same questions.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

import numpy as np
import numpy.typing as npt

Indices = npt.NDArray[np.int64]


@dataclass(frozen=True)
class CI:
    value: float
    low: float
    high: float
    n: int

    def to_dict(self) -> dict[str, float | int]:
        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in asdict(self).items()}


@dataclass(frozen=True)
class PairedResult:
    diff: float  # b - a
    low: float
    high: float
    n: int
    significant: bool  # CI excludes 0

    def to_dict(self) -> dict[str, float | int | bool]:
        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def _resample(n: int, n_resamples: int, seed: int) -> Indices:
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(n_resamples, n), dtype=np.int64)


def _percentiles(stats: npt.NDArray[np.float64], level: float) -> tuple[float, float]:
    stats = stats[~np.isnan(stats)]
    if stats.size == 0:
        return math.nan, math.nan
    alpha = (1.0 - level) / 2.0
    return float(np.quantile(stats, alpha)), float(np.quantile(stats, 1.0 - alpha))


def bootstrap_stat(
    n: int,
    stat: Callable[[Indices], float],
    n_resamples: int = 1000,
    seed: int = 42,
    level: float = 0.95,
) -> CI:
    """CI of ``stat(indices)`` over ``n`` items. Same seed -> same resamples -> same CI."""
    if n == 0:
        return CI(math.nan, math.nan, math.nan, 0)
    point = stat(np.arange(n, dtype=np.int64))
    draws = _resample(n, n_resamples, seed)
    stats = np.array([stat(row) for row in draws], dtype=np.float64)
    low, high = _percentiles(stats, level)
    return CI(float(point), low, high, n)


def bootstrap_mean(
    values: Sequence[float | None], n_resamples: int = 1000, seed: int = 42, level: float = 0.95
) -> CI:
    """CI of the mean; ``None`` values (metric undefined for that item) are dropped first."""
    arr = np.array([v for v in values if v is not None], dtype=np.float64)
    return bootstrap_stat(len(arr), lambda idx: float(arr[idx].mean()), n_resamples, seed, level)


def paired_bootstrap(
    a: Sequence[float],
    b: Sequence[float],
    n_resamples: int = 1000,
    seed: int = 42,
    level: float = 0.95,
) -> PairedResult:
    """CI of mean(b - a) over paired items (same index = same question in both runs)."""
    if len(a) != len(b):
        raise ValueError("paired bootstrap needs equally long, aligned sequences")
    diff = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
    ci = bootstrap_stat(len(diff), lambda idx: float(diff[idx].mean()), n_resamples, seed, level)
    significant = bool(ci.n) and not (ci.low <= 0.0 <= ci.high)
    return PairedResult(ci.value, ci.low, ci.high, ci.n, significant)
