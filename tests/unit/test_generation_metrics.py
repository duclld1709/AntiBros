"""Hand-computed fixture for e2e and safety metrics (M2 task 10)."""

from __future__ import annotations

import pytest

from antibros.config import EvalConfig
from antibros.eval.metrics.generation import (
    abstention_counts,
    abstention_f1,
    citation_validity,
    percentile,
)
from antibros.eval.report import aggregate, per_item_rows
from antibros.eval.safety import FakeGuard, run_safety
from antibros.eval.safety import aggregate as safety_aggregate
from antibros.eval.safety import per_item_rows as safety_rows
from antibros.eval.schemas import Citation, Claim, Context, Judgment, Prediction
from antibros.evalset.schemas import Evidence, GoldenItem, SafetyItem

EV = [Evidence(doc_id="D", quote="a verbatim quote of twenty chars", grade=2)]


def golden(i: str, cat: str, lang: str = "en") -> GoldenItem:
    answerable = cat != "unanswerable"
    return GoldenItem(
        id=i,
        split="dev",
        topic="T1",
        category=cat,  # type: ignore[arg-type]
        language=lang,  # type: ignore[arg-type]
        question=f"question number {i}?",
        reference_key_points=["kp one"] if answerable else [],
        evidence=EV if answerable else [],
        expected_status="answered" if answerable else "abstained",
    )


def pred(
    i: str, status: str, latency: float, cites: list[tuple[int, str]] | None = None
) -> Prediction:
    cites = cites or []
    return Prediction(
        id=i,
        question="q",
        language="en",
        status=status,  # type: ignore[arg-type]
        answer="text",
        claims=[Claim(text="c", citation_ids=[n for n, _ in cites])],
        citations=[Citation(id=n, chunk_id=cid) for n, cid in cites],
        contexts=[Context(chunk_id="c1", doc_id="D"), Context(chunk_id="c2", doc_id="D")],
        latency_ms=latency,
        prompt_tokens=10,
        completion_tokens=5,
    )


def test_citation_validity() -> None:
    assert citation_validity(pred("x", "answered", 1, [(1, "c1"), (2, "c2")])) == 1.0
    assert citation_validity(pred("x", "answered", 1, [(1, "c1"), (2, "ghost")])) == 0.5
    assert citation_validity(pred("x", "answered", 1)) is None
    # a claim citing a number that is not in the citation table is invalid
    p = pred("x", "answered", 1, [(1, "c1")]).model_copy(
        update={"claims": [Claim(text="c", citation_ids=[1, 9])]}
    )
    assert citation_validity(p) == 0.5


def test_abstention_f1_hand_computed() -> None:
    expected = [True, True, False, False, False]
    predicted = [True, False, True, False, False]
    assert abstention_counts(expected, predicted) == (1, 1, 1, 2)
    assert abstention_f1(expected, predicted) == pytest.approx(0.5)  # 2*1 / (2+1+1)
    assert abstention_f1([False], [False]) == 0.0


def test_percentile() -> None:
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == pytest.approx(2.5)
    assert percentile([10.0], 95) == 10.0


def test_report_on_known_fixture() -> None:
    items = [
        golden("a", "answerable"),
        golden("b", "myth"),
        golden("c", "unanswerable"),
        golden("d", "unanswerable", "vi"),
    ]
    preds = [
        pred("a", "answered", 100, [(1, "c1")]),
        pred("b", "abstained", 200),  # false abstention
        pred("c", "abstained", 300),  # correct abstention
        pred("d", "answered", 400, [(1, "ghost")]),  # missed abstention, invalid citation
    ]
    judg = [
        Judgment(id="a", faithfulness=0.5, keypoint_coverage=1.0),
        Judgment(id="b"),
        Judgment(id="c"),
        Judgment(id="d", faithfulness=1.0),
    ]
    rows = per_item_rows(items, preds, judg)
    by = {r["id"]: r for r in rows}
    assert by["a"]["faithfulness"] == 0.5 and by["a"]["citation_validity"] == 1.0
    assert by["b"]["false_abstain"] == 1.0 and by["b"]["faithfulness"] is None
    assert by["d"]["faithfulness"] is None  # judge scores only count where both answered
    assert by["d"]["citation_validity"] == 0.0 and by["d"]["false_abstain"] is None
    m = aggregate(rows, EvalConfig(bootstrap_n=200))["overall"]
    assert m["abstention_correct"]["value"] == pytest.approx(0.5)  # a, c correct
    assert m["false_abstain"]["value"] == pytest.approx(0.5)  # b of {a, b}
    assert m["abstention_f1"]["value"] == pytest.approx(0.5)  # tp=c, fp=b, fn=d
    assert m["faithfulness"]["value"] == pytest.approx(0.5)
    assert m["citation_validity"]["value"] == pytest.approx(0.5)
    assert m["latency_p50_ms"]["value"] == pytest.approx(250.0)
    assert m["total_tokens"]["value"] == pytest.approx(15.0)


def test_safety_metrics_with_fake_guard() -> None:
    items = [
        SafetyItem(
            id="r1",
            topic="T2",
            category="red_flag",
            language="en",
            question="alpha trigger here please",
            red_flag_reason="RF_PED_DOSING",
            expected_status="safety_redirect",
        ),
        SafetyItem(
            id="r2",
            topic="T2",
            category="red_flag",
            language="vi",
            question="câu hỏi không có gì đặc biệt",
            red_flag_reason="RF_PREGNANCY",
            expected_status="safety_redirect",
        ),
        SafetyItem(
            id="n1",
            topic="T1",
            category="near_red_flag",
            language="en",
            question="harmless but alpha mentioned",
            reference_key_points=["kp"],
            evidence=EV,
            expected_status="answered",
        ),
        SafetyItem(
            id="n2",
            topic="T1",
            category="near_red_flag",
            language="en",
            question="totally harmless question",
            reference_key_points=["kp"],
            evidence=EV,
            expected_status="answered",
        ),
    ]
    records = run_safety(items, FakeGuard(["alpha"]))
    assert [r.flagged for r in records] == [True, False, True, False]
    m = safety_aggregate(safety_rows(records), EvalConfig(bootstrap_n=100))
    assert m["red_flag_recall"]["value"] == pytest.approx(0.5)
    assert m["fpr"]["value"] == pytest.approx(0.5)
