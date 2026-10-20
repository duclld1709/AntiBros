"""M5: `antibros eval final` preflight, run order, resume, re-run discipline, rehearsal and the
final comparison; error classification for the judge; Ollama digest lookup."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml
from typer.testing import CliRunner

from antibros.config import ModelProfile, ModelSpec, load_app_config, load_model_profile
from antibros.eval.compare import compare_runs
from antibros.eval.final import (
    LOG_NAME,
    FinalDeps,
    FinalError,
    GitState,
    InfraFailure,
    preflight,
    read_log,
    run_plan,
)
from antibros.eval.fingerprint import evalset_identity
from antibros.eval.judge import INFRA_ERROR, INVALID_OUTPUT, FakeJudge, load_judgments, run_judge
from antibros.eval.plan import (
    FinalEvalPlan,
    GuardSelection,
    Hypothesis,
    PlannedRun,
    config_hashes,
    file_sha256,
    load_plan,
    save_plan,
)
from antibros.eval.predict import load_predictions
from antibros.eval.safety import FakeGuard
from antibros.eval.schemas import Citation, Claim, Context, Judgment, Prediction
from antibros.evalset.schemas import Evidence, GoldenItem, SafetyItem
from antibros.io import write_jsonl
from antibros.llm import StructuredOutputError
from antibros.llm.ollama import installed_digests

EVALSET = Path("evalsets/v1").resolve()
GIT = GitState(commit="c0ffee", dirty=[], plan_tracked=True)


# --- the committed plan ------------------------------------------------------------------


def test_locked_plan_still_matches_the_committed_configs() -> None:
    """Any change to a planned config, the model profile or AppConfig defaults after M4 would
    break `eval final`; this test catches it before CI goes green."""
    plan = load_plan(Path("configs/final_eval_plan.yaml"))
    assert plan.status == "locked"
    for run in plan.runs:
        assert config_hashes(run.config) == (run.config_sha256, run.config_files), run.name
    assert file_sha256(plan.models_profile) == plan.models["profile_sha256"]
    assert evalset_identity(EVALSET) == plan.evalset


# --- a self-contained plan in tmp_path ---------------------------------------------------


def golden(n_dev: int = 3, n_test: int = 4) -> list[GoldenItem]:
    ev = [Evidence(doc_id="D", quote="a verbatim quote of twenty chars", grade=2)]
    out = []
    for i in range(n_dev + n_test):
        abstain = i % 3 == 2
        out.append(
            GoldenItem(
                id=f"g{i:03d}",
                split="dev" if i < n_dev else "test",
                topic="T1",
                category="unanswerable" if abstain else "answerable",
                language="vi" if i % 2 else "en",
                question=f"question {i}?",
                reference_key_points=[] if abstain else ["question answer"],
                evidence=[] if abstain else ev,
                expected_status="abstained" if abstain else "answered",
            )
        )
    return out


def safety() -> list[SafetyItem]:
    rows = []
    for i, (cat, q) in enumerate(
        [("red_flag", "danger one"), ("near_red_flag", "fine one"), ("red_flag", "danger two")] * 2
    ):
        rows.append(
            SafetyItem(
                id=f"s{i:03d}",
                split="dev" if i < 3 else "test",
                topic="T2",
                category=cat,  # type: ignore[arg-type]
                language="en",
                question=q,
                expected_status="safety_redirect" if cat == "red_flag" else "answered",
                red_flag_reason="RF_MEDICATION" if cat == "red_flag" else None,
                reference_key_points=[] if cat == "red_flag" else ["a key point"],
                evidence=[]
                if cat == "red_flag"
                else [Evidence(doc_id="D", quote="a verbatim quote of twenty chars", grade=2)],
            )
        )
    return rows


class Pipe:
    """Answers with one cited claim; can raise after N answers or fail every item."""

    def __init__(
        self, stop_after: int | None = None, fail: bool = False, abstain_all: bool = False
    ):
        self.calls, self.stop_after, self.fail, self.abstain_all = 0, stop_after, fail, abstain_all
        self.closed = False

    def answer(self, item_id: str, question: str, language: str) -> Prediction:
        if self.stop_after is not None and self.calls == self.stop_after:
            raise KeyboardInterrupt
        self.calls += 1
        if self.fail:
            raise ConnectionError("ollama crashed")
        if self.abstain_all:
            return Prediction(id=item_id, question=question, language=language, status="abstained")
        ctx = Context(chunk_id="D::r::0", doc_id="D", text="question answer")
        return Prediction(
            id=item_id,
            question=question,
            language=language,
            status="answered",
            answer="question answer [1]",
            claims=[Claim(text="question answer", citation_ids=[1])],
            citations=[Citation(id=1, chunk_id="D::r::0")],
            contexts=[ctx],
        )

    def close(self) -> None:
        self.closed = True


class Env:
    def __init__(self, tmp: Path):
        self.tmp, self.runs = tmp, tmp / "experiments"
        self.pipes: dict[str, Pipe] = {}
        base = load_app_config(Path("configs/base.yaml")).model_dump(mode="json")
        base["evalset"]["dir"] = str(EVALSET)
        base["ingestion"]["processed_dir"] = str(tmp / "processed")
        base["pipeline"]["kind"] = "fake"
        base["judge"]["kind"] = "fake"
        base["guard"]["mode"] = "fake"
        base["name"] = "baseline"
        (tmp / "baseline.yaml").write_text(yaml.safe_dump(base), encoding="utf-8")
        (tmp / "full.yaml").write_text("extends: baseline.yaml\nname: full\n", encoding="utf-8")
        (tmp / "guard.yaml").write_text("extends: full.yaml\nname: guard\n", encoding="utf-8")
        profile = ModelProfile(
            name="t", pipeline=ModelSpec(provider="fake", model="p", digest="dp")
        )
        profile = profile.model_copy(
            update={"judge": ModelSpec(provider="fake", model="j", digest="dj")}
        )
        self.profile_path = tmp / "models.yaml"
        self.profile_path.write_text(
            yaml.safe_dump(profile.model_dump(mode="json")), encoding="utf-8"
        )
        self.plan_path = tmp / "plan.yaml"
        self.write_plan()
        self.pipeline_factory: Callable[[str], Pipe] = lambda name: Pipe()

    def write_plan(self, status: str = "locked") -> None:
        runs = []
        for name, kind, cfg, eval_set in (
            ("baseline_test", "e2e", "baseline.yaml", "golden"),
            ("full_test", "e2e", "full.yaml", "golden"),
            ("guard_test", "safety", "guard.yaml", "safety"),
        ):
            path = self.tmp / cfg
            sha, files = config_hashes(path)
            runs.append(
                PlannedRun(
                    name=name,
                    kind=kind,  # type: ignore[arg-type]
                    config=path,
                    eval_set=eval_set,  # type: ignore[arg-type]
                    config_sha256=sha,
                    config_files=files,
                )
            )
        profile = load_model_profile(self.profile_path)
        plan = FinalEvalPlan(
            status=status,  # type: ignore[arg-type]
            models_profile=self.profile_path,
            models={
                "pipeline": profile.pipeline.model_dump(mode="json") if profile.pipeline else {},
                "judge": profile.judge.model_dump(mode="json") if profile.judge else {},
                "profile_sha256": file_sha256(self.profile_path),
            },
            evalset=evalset_identity(EVALSET),
            hypotheses=[Hypothesis(id="H1", statement="s", runs=["a"], metrics=["m"])],
            runs=runs,
            guard_selection=GuardSelection(rule="r", candidates=[self.tmp / "guard.yaml"]),
        )
        save_plan(self.plan_path, plan)

    def deps(self) -> FinalDeps:
        def make_pipeline(cfg: Any, profile: Any) -> Pipe:
            pipe = self.pipeline_factory(cfg.name)
            self.pipes[cfg.name] = pipe
            return pipe

        return FinalDeps(
            make_pipeline=make_pipeline,
            make_judge=lambda cfg, profile: FakeJudge(),
            make_guard=lambda cfg, profile: FakeGuard(["danger"]),
            golden=lambda cfg: golden(),
            safety=lambda cfg: safety(),
            hardware=lambda: "test-gpu",
            installed_digest=lambda spec: pytest.fail("fake provider must not query Ollama"),
            echo=lambda msg: None,
        )

    def run(self, git: GitState = GIT, **kw: Any) -> Any:
        return run_plan(self.plan_path, runs_dir=self.runs, deps=self.deps(), git=git, **kw)

    def log(self) -> list[Any]:
        return read_log(self.runs / LOG_NAME)


@pytest.fixture
def env(tmp_path: Path) -> Env:
    return Env(tmp_path)


def test_full_plan_runs_in_order_logs_and_compares(env: Env) -> None:
    result = env.run()
    assert list(result.run_dirs) == ["baseline_test", "full_test", "guard_test"]
    log = env.log()
    assert [(e.run, e.status) for e in log] == [
        ("baseline_test", "started"),
        ("baseline_test", "completed"),
        ("full_test", "started"),
        ("full_test", "completed"),
        ("guard_test", "started"),
        ("guard_test", "completed"),
    ]
    assert {e.commit for e in log} == {"c0ffee"} and {e.hardware for e in log} == {"test-gpu"}
    base = result.run_dirs["baseline_test"]
    preds = load_predictions(base)
    assert {p.id for p in preds} == {it.id for it in golden() if it.split == "test"}
    assert env.pipes["baseline"].closed  # pipeline LLM unloaded before the judge
    meta = json.loads((base / "run.json").read_text(encoding="utf-8"))
    assert meta["split"] == "test" and meta["planned_run"] == "baseline_test"
    fp = json.loads((base / "fingerprint.json").read_text(encoding="utf-8"))
    assert fp["split"] == "test" and fp["hardware"]
    assert (base / "metrics.json").exists() and (base / "timing.json").exists()
    guard_metrics = json.loads((result.run_dirs["guard_test"] / "metrics.json").read_text("utf-8"))
    assert guard_metrics["red_flag_recall"]["value"] == 1.0
    md = result.comparison.read_text(encoding="utf-8")
    assert "## overall" in md and "## language=vi" in md and "## category=unanswerable" in md
    assert "**abstention_f1**" in md and "Guard (H2" in md

    again = env.run()  # nothing is ever run twice
    assert again.skipped == ["baseline_test", "full_test", "guard_test"]
    assert len(env.log()) == 6


def test_interrupted_run_resumes_in_place_on_same_commit_only(env: Env) -> None:
    env.pipeline_factory = lambda name: Pipe(stop_after=2)
    with pytest.raises(KeyboardInterrupt):
        env.run()
    assert [e.status for e in env.log()] == ["started", "interrupted"]
    run_dir = Path(env.log()[0].run_dir)
    assert len(load_predictions(run_dir)) == 2

    env.pipeline_factory = lambda name: Pipe()
    with pytest.raises(FinalError, match="same commit"):
        env.run(git=GitState(commit="other", dirty=[], plan_tracked=True))
    result = env.run()
    assert result.run_dirs["baseline_test"] == run_dir
    ids = [p.id for p in load_predictions(run_dir)]
    assert len(ids) == len(set(ids)) == 4  # resumed, nothing duplicated
    statuses = [(e.run, e.status) for e in env.log() if e.run == "baseline_test"]
    assert statuses[-2:] == [("baseline_test", "resumed"), ("baseline_test", "completed")]


def test_failed_run_needs_a_single_recorded_rerun(env: Env) -> None:
    env.pipeline_factory = lambda name: Pipe(fail=True)
    with pytest.raises(InfraFailure, match="prediction"):
        env.run()
    assert env.log()[-1].status == "failed"
    with pytest.raises(FinalError, match="--rerun baseline_test"):
        env.run()  # a failed run is never silently resumed
    with pytest.raises(FinalError, match="only a failed run"):
        env.run(rerun={"baseline_test": "oom", "full_test": "oom"})
    # every decision is made before anything runs: the refusal recorded nothing
    assert not (env.runs / "deviations.md").exists() and env.log()[-1].status == "failed"

    env.pipeline_factory = lambda name: Pipe(fail=name == "baseline")
    with pytest.raises(InfraFailure):
        env.run(rerun={"baseline_test": "Ollama crashed (CUDA error); restarted Ollama"})
    deviations = (env.runs / "deviations.md").read_text(encoding="utf-8")
    assert "re-run of `baseline_test`" in deviations and "restarted Ollama" in deviations
    rerun_entry = [e for e in env.log() if e.attempt == 1]
    assert rerun_entry and rerun_entry[0].run_dir.endswith("_rerun1")
    with pytest.raises(FinalError, match="already re-run once"):
        env.run(rerun={"baseline_test": "again"})


def test_rerun_after_failure_completes_and_is_documented(env: Env) -> None:
    env.pipeline_factory = lambda name: Pipe(fail=True)
    with pytest.raises(InfraFailure):
        env.run()
    env.pipeline_factory = lambda name: Pipe()
    result = env.run(rerun={"baseline_test": "OOM after a driver update"})
    assert result.run_dirs["baseline_test"].name.endswith("_rerun1")
    assert result.comparison is not None
    assert (env.runs / "deviations.md").exists()


def test_preflight_refuses_unsafe_states(env: Env) -> None:
    plan = load_plan(env.plan_path)

    def problems(git: GitState = GIT, **kw: Any) -> list[str]:
        return preflight(
            kw.get("plan", plan),
            env.plan_path,
            git=git,
            evalset_dir=EVALSET,
            installed_digest=kw.get("installed", lambda s: s.digest),
            rehearsal=kw.get("rehearsal", False),
        )

    assert problems() == []
    assert problems(plan=plan.model_copy(update={"status": "draft"}))
    dirty = GitState("c", ["src/antibros/cli.py", "experiments/x/predictions.jsonl"], True)
    assert any("not clean" in p for p in problems(dirty))
    assert problems(GitState("c", ["experiments/run/x.json"], True)) == []  # outputs are fine
    assert problems(dirty, rehearsal=True) == []
    assert any("not committed" in p for p in problems(GitState("c", [], False)))

    (env.tmp / "full.yaml").write_text("extends: baseline.yaml\nname: full2\n", encoding="utf-8")
    assert any("full_test" in p for p in problems())


def test_preflight_checks_installed_ollama_digests(env: Env) -> None:
    spec = ModelSpec(provider="ollama", model="qwen3.5:4b", digest="d" * 64)
    profile = ModelProfile(name="o", pipeline=spec, judge=spec.model_copy(update={"model": "g"}))
    env.profile_path.write_text(yaml.safe_dump(profile.model_dump(mode="json")), encoding="utf-8")
    env.write_plan()
    plan = load_plan(env.plan_path)

    def check(installed: Any) -> list[str]:
        return preflight(
            plan,
            env.plan_path,
            git=GIT,
            evalset_dir=EVALSET,
            installed_digest=installed,
            rehearsal=False,
        )

    assert check(lambda s: "d" * 64) == []
    assert any("not installed" in p for p in check(lambda s: None))
    assert any("installed digest" in p for p in check(lambda s: "e" * 64))

    def down(spec: ModelSpec) -> str:
        raise httpx.ConnectError("refused")

    assert any("cannot query Ollama" in p for p in check(down))


def test_rehearsal_uses_dev_limit_and_never_logs(env: Env) -> None:
    dirty = GitState("c0ffee", ["src/antibros/api.py"], True)
    result = env.run(git=dirty, rehearsal=True, limit=2)
    assert not (env.runs / LOG_NAME).exists()
    base = result.run_dirs["baseline_test"]
    assert base.name.startswith("rehearsal-")
    preds = load_predictions(base)
    assert len(preds) == 2 and all(p.id in {"g000", "g001"} for p in preds)
    assert "REHEARSAL" in result.comparison.read_text(encoding="utf-8")
    assert env.run(git=dirty, rehearsal=True, limit=2).skipped  # completed rehearsal is kept
    with pytest.raises(FinalError):
        env.run(git=dirty)  # the real run still needs a clean tree


def test_cli_eval_final_refuses_outside_a_git_checkout(tmp_path: Path, monkeypatch: Any) -> None:
    from antibros.cli import app

    monkeypatch.chdir(tmp_path)
    res = CliRunner().invoke(app, ["eval", "final", "--plan", "plan.yaml"])
    assert res.exit_code == 1 and "Refusing to run" in res.output


# --- harness pieces ------------------------------------------------------------------------


def test_run_judge_separates_infrastructure_from_invalid_output(tmp_path: Path) -> None:
    items = [it for it in golden() if it.category == "answerable"][:2]
    preds = [Pipe().answer(it.id, it.question, it.language) for it in items]

    class Flaky:
        def __init__(self) -> None:
            self.n = 0

        def judge(self, item: GoldenItem, pred: Prediction) -> Judgment:
            self.n += 1
            if self.n == 1:
                raise StructuredOutputError("bad json")
            raise httpx.ReadTimeout("judge timed out")

    run_judge(items, preds, Flaky(), tmp_path)
    errors = [j.error or "" for j in load_judgments(tmp_path)]
    assert errors[0].startswith(INVALID_OUTPUT) and errors[1].startswith(INFRA_ERROR)


def test_compare_subsets_and_paired_abstention_f1(env: Env) -> None:
    env.pipeline_factory = lambda name: Pipe(abstain_all=name == "full")
    result = env.run()
    a, b = result.run_dirs["baseline_test"], result.run_dirs["full_test"]
    cfg = load_app_config(env.tmp / "baseline.yaml").eval
    overall = compare_runs(a, b, cfg)
    vi = compare_runs(a, b, cfg, {"language": "vi"})
    assert overall.n_common == 4 and vi.n_common == 2
    f1 = overall.metrics["abstention_f1"]
    assert f1["mean_a"] == 0.0 and f1["mean_b"] > 0.0 and f1["diff"] == f1["mean_b"]


def test_installed_digests_reads_api_tags() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tags"
        return httpx.Response(
            200,
            json={
                "models": [{"name": "qwen3.5:4b", "model": "qwen3.5:4b", "digest": "sha256:abc"}]
            },
        )

    digests = installed_digests("http://fake:11434/v1", transport=httpx.MockTransport(handler))
    assert digests == {"qwen3.5:4b": "abc"}


def test_report_rows_written_for_jsonl_helpers(tmp_path: Path) -> None:
    write_jsonl(tmp_path / "x.jsonl", [{"a": "Tiếng Việt"}])
    assert "Tiếng Việt" in (tmp_path / "x.jsonl").read_text(encoding="utf-8")


# --- error analysis ---------------------------------------------------------------------


def _row(**kw: Any) -> dict[str, Any]:
    base = {
        "id": "g1",
        "language": "en",
        "category": "answerable",
        "status": "answered",
        "expected_abstain": False,
        "predicted_abstain": False,
        "mrr": 1.0,
        "faithfulness": 1.0,
        "citation_precision": 1.0,
        "keypoint_coverage": 1.0,
    }
    return base | kw


def test_error_labels_follow_the_declared_rules() -> None:
    from antibros.eval.errors import ErrorAnalysisConfig, e2e_errors

    ctx = [Context(chunk_id=f"D::r::{i}", doc_id="D") for i in range(5)]
    pred = Prediction(
        id="g1",
        question="Tôi nên ăn bao nhiêu đạm?",
        language="vi",
        status="answered",
        contexts=ctx,
        steps=[{"node": "query_analyze", "queries": ["protein per day"]}],
    )
    cfg = ErrorAnalysisConfig(judge_threshold=0.5)

    def label(**kw: Any) -> str | None:
        errs = e2e_errors([_row(**kw)], [pred], cfg)
        return errs[0].label if errs else None

    assert label() is None  # correct and well-scored: not an error
    assert label(mrr=0.0) is None  # a retrieval miss alone is not a wrong answer
    assert label(status="safety_redirect", predicted_abstain=True) == "guard"
    assert label(status="abstained", predicted_abstain=True) == "wrong_abstain"
    assert label(expected_abstain=True) == "wrong_abstain"
    assert label(mrr=0.0, faithfulness=0.2) == "retrieval_miss"
    assert label(mrr=1 / 7, faithfulness=0.2) == "ranking"  # rank 7 > 5 passages in context
    assert label(citation_precision=0.3, faithfulness=0.3) == "citation"
    assert label(faithfulness=0.3) == "unfaithful"
    assert label(keypoint_coverage=0.0) == "incomplete"
    vi = e2e_errors([_row(language="vi", mrr=0.0, faithfulness=0.1)], [pred], cfg)[0]
    assert "translation?" in vi.hints and "protein per day" in vi.hints


def test_safety_errors_and_render() -> None:
    from antibros.eval.errors import render_errors, safety_errors

    rows = [
        {"id": "s1", "language": "vi", "category": "red_flag", "red_flag_recall": 0.0},
        {"id": "s2", "language": "en", "category": "near_red_flag", "false_positive": 1.0},
        {"id": "s3", "language": "en", "category": "red_flag", "red_flag_recall": 1.0},
    ]
    errs = safety_errors(rows)
    assert [(e.id, e.evidence) for e in errs] == [
        ("s1", "red flag missed"),
        ("s2", "false positive"),
    ]
    md = render_errors([(Path("experiments/x"), errs, 3)])
    assert "2 wrong of 3 items" in md and "guard 2" in md
