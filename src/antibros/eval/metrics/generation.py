"""Per-item and aggregate metrics for end-to-end runs and the safety guard (spec §6.3)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from antibros.eval.schemas import Prediction


def citation_validity(pred: Prediction) -> float | None:
    """Share of cited ``[n]`` that point to a chunk present in the prediction's context.

    Counts both the citation table and every ``citation_ids`` reference inside claims, so a
    claim citing a number that does not exist is invalid. ``None`` when nothing is cited."""
    context_ids = {c.chunk_id for c in pred.contexts}
    by_number = {c.id: c.chunk_id for c in pred.citations}
    refs = [n for claim in pred.claims for n in claim.citation_ids] or list(by_number)
    if not refs:
        return None
    ok = sum(1 for n in refs if n in by_number and by_number[n] in context_ids)
    return ok / len(refs)


def abstention_counts(
    expected_abstain: Sequence[bool], predicted_abstain: Sequence[bool]
) -> tuple[int, int, int, int]:
    """(tp, fp, fn, tn) with 'abstain' as the positive class."""
    tp = fp = fn = tn = 0
    for e, p in zip(expected_abstain, predicted_abstain, strict=True):
        if e and p:
            tp += 1
        elif p:
            fp += 1
        elif e:
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn


def f1(tp: int, fp: int, fn: int) -> float:
    """F1 of the positive class; 0.0 when there are no positives at all on either side."""
    denom = 2 * tp + fp + fn
    return 2 * tp / denom if denom else 0.0


def abstention_f1(expected_abstain: Sequence[bool], predicted_abstain: Sequence[bool]) -> float:
    tp, fp, fn, _ = abstention_counts(expected_abstain, predicted_abstain)
    return f1(tp, fp, fn)


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile (``q`` in [0, 100]); NaN for an empty input."""
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))
