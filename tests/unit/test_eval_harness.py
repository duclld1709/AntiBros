"""Harness behaviour: resume, test-split refusal, fingerprints, compare, timing, retrieval eval."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest
from qdrant_client import QdrantClient
from typer.testing import CliRunner

from antibros.chunking.base import RegexTokenizer, make_chunker
from antibros.cli import app
from antibros.config import ChunkingConfig, EmbedderConfig, EvalConfig, RetrievalConfig
from antibros.embedding.base import FakeEmbedder
from antibros.eval.compare import FingerprintMismatch, compare_runs, render_markdown
from antibros.eval.fingerprint import HARD_FIELDS, compare_fingerprints
from antibros.eval.judge import FakeJudge, load_judgments, run_judge
from antibros.eval.predict import FakePipeline, load_predictions, run_predict
from antibros.eval.retrieval_eval import Query, chunk_ref, run_retrieval_eval
from antibros.eval.runs import SplitRefusedError, guard_split, new_run_dir, timed_phase
from antibros.eval.schemas import Prediction
from antibros.evalset.schemas import Evidence, GoldenItem
from antibros.indexing.qdrant_store import ensure_collection, upsert_chunks
from antibros.ingestion.schemas import Document
from antibros.io import read_json, write_json, write_jsonl
from antibros.prompts import parse_prompt
from antibros.retrieval.base import make_retriever

EV = [Evidence(doc_id="D", quote="a verbatim quote of twenty chars", grade=2)]


def items(n: int) -> list[GoldenItem]:
    return [
        GoldenItem(
            id=f"g{i:03d}",
            split="dev",
            topic="T1",
            category="answerable",
            language="en",
            question=f"question {i}?",
            reference_key_points=["question answer"],
            evidence=EV,
            expected_status="answered",
        )
        for i in range(n)
    ]


class Interrupting(FakePipeline):
    """Raises KeyboardInterrupt (like Ctrl+C) after ``stop_after`` answers."""

    def __init__(self, stop_after: int):
        self.calls, self.stop_after = 0, stop_after

    def answer(self, item_id: str, question: str, language: str) -> Prediction:
        if self.calls == self.stop_after:
            raise KeyboardInterrupt
        self.calls += 1
        return super().answer(item_id, question, language)


def test_predict_resumes_after_interrupt_without_dupes_or_loss(tmp_path: Path) -> None:
    its = items(10)
    with pytest.raises(KeyboardInterrupt):
        run_predict(its, Interrupting(stop_after=4), tmp_path)
    assert [p.id for p in load_predictions(tmp_path)] == [i.id for i in its[:4]]
    # simulate a crash in the middle of writing the 5th line
    with (tmp_path / "predictions.jsonl").open("a", encoding="utf-8") as f:
        f.write('{"id": "g004", "quest')
    stats = run_predict(its, FakePipeline(), tmp_path)
    assert stats.done_before == 4 and stats.predicted == 6
    ids = [p.id for p in load_predictions(tmp_path)]
    assert ids == [i.id for i in its] and len(set(ids)) == len(ids)
    assert run_predict(its, FakePipeline(), tmp_path).predicted == 0


def test_predict_limit_and_errors(tmp_path: Path) -> None:
    class Boom(FakePipeline):
        def answer(self, item_id: str, question: str, language: str) -> Prediction:
            raise RuntimeError("model crashed")

    stats = run_predict(items(5), Boom(), tmp_path, limit=2)
    assert stats.predicted == 2 and stats.errors == 2
    assert {p.status for p in load_predictions(tmp_path)} == {"error"}


def test_judge_resumes(tmp_path: Path) -> None:
    its = items(3)
    run_predict(its, FakePipeline(), tmp_path)
    preds = load_predictions(tmp_path)
    run_judge(its, preds[:1], FakeJudge(), tmp_path)
    stats = run_judge(its, preds, FakeJudge(), tmp_path)
    assert stats.done_before == 1 and stats.judged == 2
    assert [j.id for j in load_judgments(tmp_path)] == [i.id for i in its]


def test_test_split_is_refused() -> None:
    with pytest.raises(SplitRefusedError):
        guard_split("test")
    guard_split("test", allow_test=True)  # only `eval final` passes this
    guard_split("dev")
    runner = CliRunner()
    for cmd in (
        ["eval", "safety", "--config", "c.yaml", "--split", "test"],
        [
            "eval",
            "predict",
            "--config",
            "c.yaml",
            "--models",
            "m.yaml",
            "--tag",
            "x",
            "--split",
            "test",
        ],
    ):
        res = runner.invoke(app, cmd)
        assert res.exit_code == 1 and "eval final" in res.output


def test_run_dir_and_timing_budget(tmp_path: Path) -> None:
    run = new_run_dir(tmp_path, "base_dev", now=datetime(2026, 10, 20, 9, 5))
    assert run.name == "20261020-0905_base_dev"
    assert new_run_dir(tmp_path, "base_dev", now=datetime(2026, 10, 20, 9, 5)).name.endswith("-2")
    with pytest.raises(ValueError):
        new_run_dir(tmp_path, "bad:tag")
    with timed_phase(run, "predict", "e2e_dev", {"e2e_dev": -1.0}, 3) as extra:
        extra["cached"] = 1
    with pytest.raises(RuntimeError), timed_phase(run, "judge", "e2e_dev", {"e2e_dev": -1.0}, 3):
        raise RuntimeError("interrupted")
    t = read_json(run / "timing.json")
    assert set(t["phases"]) == {"predict", "judge"}
    assert t["phases"]["predict"]["cached"] == 1 and t["phases"]["judge"]["completed"] is False
    assert t["over_budget"] is True and t["budget_key"] == "e2e_dev"


def fp(**over: object) -> dict[str, object]:
    base = {f: "x" for f in HARD_FIELDS} | {"hardware": "gpu", "git_commit": "abc"}
    return base | over


def test_fingerprint_hard_soft() -> None:
    assert compare_fingerprints(fp(), fp(hardware="cpu")) == ([], ["hardware"])
    hard, _ = compare_fingerprints(fp(), fp(judge_model="other"))
    assert hard == ["judge_model"]


def _run(tmp: Path, name: str, values: list[float], fingerprint: dict[str, object]) -> Path:
    d = tmp / name
    d.mkdir()
    write_json(d / "fingerprint.json", fingerprint)
    write_jsonl(
        d / "per_item.jsonl",
        [{"id": f"q{i}", "language": "en", "mrr": v} for i, v in enumerate(values)],
    )
    return d


def test_compare_paired_and_refuses_hard_mismatch(tmp_path: Path) -> None:
    a = _run(tmp_path, "a", [0.2, 0.4, 0.5, 0.1, 0.3], fp())
    b = _run(tmp_path, "b", [0.4, 0.6, 0.7, 0.3, 0.5], fp(git_commit="def"))
    cmp = compare_runs(a, b, EvalConfig(bootstrap_n=300))
    assert cmp.n_common == 5 and cmp.soft_differences == ["git_commit"]
    assert cmp.metrics["mrr"]["diff"] == pytest.approx(0.2) and cmp.metrics["mrr"]["significant"]
    assert "mrr" in render_markdown(cmp, "a", "b")
    c = _run(tmp_path, "c", [0.1] * 5, fp(evalset_sha256="different"))
    with pytest.raises(FingerprintMismatch, match="evalset_sha256"):
        compare_runs(a, c, EvalConfig())
    res = CliRunner().invoke(app, ["compare", str(a), str(c)])
    assert res.exit_code == 1 and "not comparable" in res.output


def test_retrieval_eval_end_to_end(mini_docs: list[Document]) -> None:
    chunker = make_chunker("section_aware", ChunkingConfig(tokenizer="fake"), RegexTokenizer())
    chunks = [c for d in mini_docs for c in chunker.chunk(d)]
    client = QdrantClient(":memory:")
    emb = FakeEmbedder("fake", EmbedderConfig(kind="fake", dim=64))
    ensure_collection(client, "c", emb.dim, sparse=True)
    upsert_chunks(client, "c", chunks, emb.embed_documents([c.embed_text for c in chunks]))
    retriever = make_retriever(RetrievalConfig(mode="hybrid"), client, "c", emb)
    target = max(chunks, key=lambda c: c.token_count)
    quote = target.text[20:220]
    queries = [
        Query("q1", target.text, "en", [Evidence(doc_id=target.doc_id, quote=quote, grade=2)])
    ]
    texts = {d.doc_id: d.clean_text for d in mini_docs}
    rows, metrics = run_retrieval_eval(
        queries,
        lambda q, k: [chunk_ref(h.payload) for h in retriever.retrieve(q, k)],
        [chunk_ref(c.model_dump()) for c in chunks],
        texts,
        EvalConfig(bootstrap_n=50),
        10,
    )
    assert rows[0]["mrr"] == 1.0 and rows[0]["evidence_recall@5"] == 1.0
    assert metrics["overall"]["mrr"]["value"] == 1.0 and metrics["n_queries"] == 1
    assert "language=en" in metrics


def test_prompt_template_parsing() -> None:
    tpl = parse_prompt("t", "## system\nYou are $role.\n\n## user\nQ: $question {braces ok}\n")
    msgs = tpl.render(role="a judge", question="why?")
    assert msgs[0].content == "You are a judge." and msgs[1].content == "Q: why? {braces ok}"
    with pytest.raises(ValueError):
        parse_prompt("bad", "no sections here")
