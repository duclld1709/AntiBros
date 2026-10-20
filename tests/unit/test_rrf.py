from __future__ import annotations

import pytest

from antibros.retrieval.base import rrf


def test_rrf_scores_and_order() -> None:
    fused = rrf([["a", "b", "c"], ["b", "c", "d"]], k=60)
    scores = dict(fused)
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["a"] == pytest.approx(1 / 61)
    assert scores["d"] == pytest.approx(1 / 63)
    assert [d for d, _ in fused] == ["b", "c", "a", "d"]


def test_rrf_ignores_raw_scores_and_is_deterministic_on_ties() -> None:
    # "x" and "y" both appear once at rank 1: tie broken by best rank, then ID.
    assert [d for d, _ in rrf([["y"], ["x"]])] == ["x", "y"]
    assert rrf([["y"], ["x"]]) == rrf([["x"], ["y"]])


def test_rrf_k_controls_rank_weighting() -> None:
    small = dict(rrf([["a", "b"]], k=1))
    assert small["a"] / small["b"] == pytest.approx(3 / 2)


def test_rrf_empty() -> None:
    assert rrf([]) == []
    assert rrf([[], []]) == []
