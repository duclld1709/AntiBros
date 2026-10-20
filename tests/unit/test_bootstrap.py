from __future__ import annotations

import math

import numpy as np
import pytest

from antibros.eval.bootstrap import bootstrap_mean, bootstrap_stat, paired_bootstrap


def test_same_seed_same_ci_and_point_estimate() -> None:
    values = [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0]
    a = bootstrap_mean(values, 1000, 42, 0.95)
    b = bootstrap_mean(values, 1000, 42, 0.95)
    assert a == b
    assert a.value == pytest.approx(0.7)
    assert a.low <= a.value <= a.high
    assert a.n == 10
    assert bootstrap_mean(values, 1000, 7, 0.95) != a  # different seed, different draws


def test_none_values_are_dropped_and_empty_is_nan() -> None:
    ci = bootstrap_mean([1.0, None, 0.0, None], 200, 42)
    assert ci.n == 2 and ci.value == pytest.approx(0.5)
    empty = bootstrap_mean([None, None], 200, 42)
    assert empty.n == 0 and math.isnan(empty.value)


def test_constant_values_give_degenerate_ci() -> None:
    ci = bootstrap_mean([0.25] * 8, 300, 42)
    assert ci.value == ci.low == ci.high == pytest.approx(0.25)


def test_bootstrap_stat_with_custom_statistic() -> None:
    data = np.array([1, 2, 3, 4, 100], dtype=float)
    ci = bootstrap_stat(len(data), lambda idx: float(np.median(data[idx])), 500, 42)
    assert ci.value == 3.0
    assert ci.low <= 3.0 <= ci.high


def test_paired_bootstrap_detects_consistent_improvement() -> None:
    a = [0.2, 0.4, 0.5, 0.1, 0.3, 0.6, 0.2, 0.4]
    b = [x + 0.2 for x in a]
    res = paired_bootstrap(a, b, 1000, 42)
    assert res.diff == pytest.approx(0.2)
    assert res.significant and res.low == pytest.approx(0.2)
    noisy = paired_bootstrap([0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 0, 1], 1000, 42)
    assert not noisy.significant
    assert paired_bootstrap(a, b, 1000, 42) == res
    with pytest.raises(ValueError):
        paired_bootstrap([1.0], [1.0, 2.0])
