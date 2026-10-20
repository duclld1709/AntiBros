"""``antibros eval final``: the only way to run the test split (spec §7, ADR-0014).

Before anything runs, the pre-registered plan is checked: it is locked and committed, the
working tree is clean (outputs under ``experiments/`` excepted), every planned config still
hashes to the pinned sha256, the model profile and digests match (including the digests the
local Ollama server reports), and the evalset checksums verify.

Then the planned runs execute in order (``baseline_test`` → ``full_test`` → ``guard_test``).
An e2e run is predict → unload → judge → report; the guard run is the safety evaluation.
Every state change of a test run is appended to ``experiments/test_runs.log`` (JSONL):

- ``started`` / ``resumed`` / ``interrupted`` / ``failed`` / ``completed``;
- an interrupted run (Ctrl+C, sleep, power loss) resumes in its own directory, on the same
  commit and hardware: predictions and judgments already written are kept, so this is not a
  re-run;
- a failed run (prediction or judge infrastructure errors, OOM, runtime crash) is never
  resumed. It may be re-run **once** with ``--rerun <name> --reason ...``, in a new directory,
  and the re-run is recorded in ``experiments/deviations.md``;
- a completed run is never run again.

``--rehearsal`` runs the same flow on the dev split with ``--limit`` questions, in
``rehearsal-*`` directories, without touching ``test_runs.log``; it skips only the clean-tree
check, so a rehearsal can run before the final commit.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from antibros.config import AppConfig, ModelProfile, ModelSpec, load_app_config, load_model_profile
from antibros.eval.compare import Comparison, compare_runs
from antibros.eval.fingerprint import build_fingerprint, evalset_identity
from antibros.eval.judge import INFRA_ERROR, Judge, load_judgments, run_judge
from antibros.eval.plan import FinalEvalPlan, PlannedRun, config_hashes, file_sha256, load_plan
from antibros.eval.predict import Pipeline, load_predictions, run_predict
from antibros.eval.report import write_report
from antibros.eval.runs import timed_phase, write_config_snapshot
from antibros.eval.safety import Guard, aggregate, per_item_rows, run_safety
from antibros.evalset.freeze import verify_checksums
from antibros.evalset.schemas import GoldenItem, SafetyItem
from antibros.io import (
    append_jsonl,
    read_json,
    read_jsonl,
    read_text,
    write_json,
    write_jsonl,
    write_text,
)
from antibros.logging import get_logger

log = get_logger(__name__)

LOG_NAME = "test_runs.log"
DEVIATIONS = "deviations.md"
STATE = "final_state.json"
# Paths the harness itself writes; changes there do not make the tree "dirty".
OUTPUT_PREFIXES = ("experiments/",)
LogStatus = Literal["started", "resumed", "interrupted", "failed", "completed"]


class FinalError(RuntimeError):
    """The plan cannot (or must not) run; nothing on the test split was touched."""


class InfraFailure(RuntimeError):
    """A run hit an infrastructure fault; it is logged as failed and may be re-run once."""


# --- environment ----------------------------------------------------------------------------


@dataclass(frozen=True)
class GitState:
    commit: str
    dirty: list[str]  # POSIX paths from `git status --porcelain`
    plan_tracked: bool


def read_git_state(plan_path: Path) -> GitState:
    def git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=False)

    head = git("rev-parse", "HEAD")
    status = git("status", "--porcelain=v1", "--untracked-files=all")
    if head.returncode or status.returncode:
        raise FinalError("not a git checkout (eval final needs the commit of the plan)")
    dirty = []
    for line in status.stdout.splitlines():
        path = line[3:].split(" -> ")[-1].strip().strip('"')
        dirty.append(path)
    tracked = git("ls-files", "--error-unmatch", plan_path.as_posix()).returncode == 0
    return GitState(head.stdout.strip(), dirty, tracked)


def ollama_digests(spec: ModelSpec) -> str | None:
    """Digest of ``spec.model`` as installed in the local Ollama server (``None`` if absent)."""
    from antibros.config import Settings
    from antibros.llm.ollama import installed_digests

    return installed_digests(Settings().llm_base_url).get(spec.model)


# --- log ----------------------------------------------------------------------------------


class LogEntry(BaseModel):
    time: str
    plan_sha256: str
    run: str
    attempt: int = 0  # 0 = planned run, 1 = the single allowed re-run
    status: LogStatus
    commit: str
    hardware: str
    run_dir: str
    note: str = ""


def read_log(path: Path) -> list[LogEntry]:
    if not path.exists():
        return []
    return [LogEntry.model_validate_json(line) for line in read_text(path).splitlines() if line]


@dataclass(frozen=True)
class RunState:
    status: Literal["new", "resume", "failed", "done"]
    attempt: int = 0
    run_dir: str | None = None
    commit: str | None = None
    hardware: str | None = None
    note: str = ""


def run_state(entries: Sequence[LogEntry], plan_sha: str, name: str) -> RunState:
    mine = [e for e in entries if e.plan_sha256 == plan_sha and e.run == name]
    if not mine:
        return RunState("new")
    if any(e.status == "completed" for e in mine):
        done = next(e for e in mine if e.status == "completed")
        return RunState("done", done.attempt, done.run_dir, done.commit, done.hardware)
    last = mine[-1]
    status: Literal["resume", "failed"] = "failed" if last.status == "failed" else "resume"
    return RunState(status, last.attempt, last.run_dir, last.commit, last.hardware, last.note)


# --- preflight ------------------------------------------------------------------------------


def preflight(
    plan: FinalEvalPlan,
    plan_path: Path,
    *,
    git: GitState,
    evalset_dir: Path,
    installed_digest: Callable[[ModelSpec], str | None],
    rehearsal: bool,
) -> list[str]:
    """Every reason the plan must not run; empty when it may."""
    problems: list[str] = []
    if plan.status != "locked":
        problems.append("plan is not locked (run `antibros plan lock` at the end of M4)")
    if not git.plan_tracked or plan_path.as_posix() in git.dirty:
        problems.append(f"{plan_path.as_posix()} is not committed")
    dirty = [p for p in git.dirty if not p.startswith(OUTPUT_PREFIXES)]
    if dirty and not rehearsal:
        problems.append(f"working tree is not clean: {', '.join(sorted(dirty)[:5])}")
    for run in plan.runs:
        try:
            sha, files = config_hashes(run.config)
        except (OSError, ValueError) as e:
            problems.append(f"{run.name}: cannot load {run.config.as_posix()}: {e}")
            continue
        if sha != run.config_sha256 or files != run.config_files:
            problems.append(f"{run.name}: {run.config.as_posix()} no longer matches the plan")
    if file_sha256(plan.models_profile) != plan.models.get("profile_sha256"):
        problems.append(f"{plan.models_profile.as_posix()} changed since the plan was locked")
    profile = load_model_profile(plan.models_profile)
    for role in ("pipeline", "judge"):
        spec: ModelSpec | None = getattr(profile, role)
        pinned = plan.models.get(role, {})
        if spec is None or spec.model != pinned.get("model") or spec.digest != pinned.get("digest"):
            problems.append(f"{role} model differs from the plan")
            continue
        if spec.provider == "ollama":
            try:
                installed = installed_digest(spec)
            except Exception as e:  # server down is a problem to report, not a crash
                problems.append(f"cannot query Ollama for {spec.model}: {e}")
                continue
            if installed is None:
                problems.append(f"{spec.model} is not installed in Ollama")
            elif installed != spec.digest:
                pinned_digest = (spec.digest or "")[:12]
                problems.append(
                    f"{spec.model}: installed digest {installed[:12]} != plan {pinned_digest}"
                )
    checksums = verify_checksums(evalset_dir)
    if not checksums.ok:
        problems.append(f"evalset checksums fail: {'; '.join(checksums.problems[:3])}")
    elif evalset_identity(evalset_dir) != plan.evalset:
        problems.append("evalset identity differs from the plan")
    return problems


# --- execution ----------------------------------------------------------------------------


def _close(obj: object) -> None:
    close = getattr(obj, "close", None)
    if callable(close):
        close()


@dataclass
class FinalDeps:
    """Factories for the expensive parts, so tests run the whole flow with fakes."""

    make_pipeline: Callable[[AppConfig, ModelProfile], Pipeline]
    make_judge: Callable[[AppConfig, ModelProfile], Judge]
    make_guard: Callable[[AppConfig, ModelProfile], Guard]
    golden: Callable[[AppConfig], list[GoldenItem]]
    safety: Callable[[AppConfig], list[SafetyItem]]
    hardware: Callable[[], str]
    installed_digest: Callable[[ModelSpec], str | None] = ollama_digests
    echo: Callable[[str], None] = lambda msg: log.info("eval_final", msg=msg)


def default_deps(echo: Callable[[str], None]) -> FinalDeps:
    from antibros.eval.fingerprint import hardware
    from antibros.eval.judge import make_judge
    from antibros.eval.predict import make_pipeline
    from antibros.eval.safety import make_guard
    from antibros.evalset.store import Layout, load_items_strict

    def items(name: Literal["golden", "safety"]) -> Callable[[AppConfig], list[Any]]:
        return lambda cfg: load_items_strict(Layout(cfg.evalset.dir).final(name), name)

    return FinalDeps(
        make_pipeline=make_pipeline,
        make_judge=make_judge,
        make_guard=make_guard,
        golden=items("golden"),
        safety=items("safety"),
        hardware=hardware,
        echo=echo,
    )


@dataclass
class FinalResult:
    run_dirs: dict[str, Path] = field(default_factory=dict)
    comparison: Path | None = None
    skipped: list[str] = field(default_factory=list)


@dataclass
class _Ctx:
    plan: FinalEvalPlan
    plan_sha: str
    profile: ModelProfile
    git: GitState
    hardware: str
    rehearsal: bool
    limit: int | None
    runs_dir: Path
    deps: FinalDeps

    @property
    def split(self) -> str:
        return "dev" if self.rehearsal else "test"

    @property
    def prefix(self) -> str:
        return f"{'rehearsal' if self.rehearsal else 'final'}-{self.plan_sha[:8]}"


def run_plan(
    plan_path: Path,
    *,
    runs_dir: Path,
    deps: FinalDeps,
    git: GitState,
    rehearsal: bool = False,
    limit: int | None = None,
    rerun: Mapping[str, str] | None = None,
) -> FinalResult:
    plan = load_plan(plan_path)
    plan_sha = file_sha256(plan_path)
    evalset_dir = load_app_config(plan.runs[0].config).evalset.dir
    problems = preflight(
        plan,
        plan_path,
        git=git,
        evalset_dir=evalset_dir,
        installed_digest=deps.installed_digest,
        rehearsal=rehearsal,
    )
    if problems:
        raise FinalError("plan checks failed:\n- " + "\n- ".join(problems))
    rerun = dict(rerun or {})
    unknown = set(rerun) - {r.name for r in plan.runs}
    if unknown:
        raise FinalError(f"--rerun names no planned run: {sorted(unknown)}")
    ctx = _Ctx(
        plan,
        plan_sha,
        load_model_profile(plan.models_profile),
        git,
        deps.hardware(),
        rehearsal,
        limit if rehearsal else None,
        runs_dir,
        deps,
    )
    log_path = runs_dir / LOG_NAME
    entries = [] if rehearsal else read_log(log_path)
    # Decide every run's fate before executing any, so a refusal never leaves a half plan.
    decisions: list[tuple[PlannedRun, RunState, Path]] = []
    for run in plan.runs:
        state = _rehearsal_state(ctx, run) if rehearsal else run_state(entries, plan_sha, run.name)
        decisions.append((run, state, _decide(ctx, run, state, rerun)))

    result = FinalResult()
    for run, state, run_dir in decisions:
        result.run_dirs[run.name] = run_dir
        if state.status == "done":
            deps.echo(f"{run.name}: already completed in {run_dir.as_posix()}, skipping")
            result.skipped.append(run.name)
            continue
        attempt = state.attempt + (1 if run.name in rerun else 0)
        if run.name in rerun:
            _record_deviation(ctx, run, state, run_dir, rerun[run.name])
        status: LogStatus = "resumed" if state.status == "resume" else "started"
        _log(ctx, log_path, run, attempt, status, run_dir)
        deps.echo(f"{run.name}: {status} → {run_dir.as_posix()}")
        try:
            _execute(ctx, run, run_dir)
        except KeyboardInterrupt:
            _log(ctx, log_path, run, attempt, "interrupted", run_dir)
            raise
        except Exception as e:
            _log(ctx, log_path, run, attempt, "failed", run_dir, note=str(e)[:300])
            raise InfraFailure(f"{run.name} failed: {e}") from e
        _log(ctx, log_path, run, attempt, "completed", run_dir)
        deps.echo(f"{run.name}: completed")

    e2e = [r.name for r in plan.runs if r.kind == "e2e"]
    if {"baseline_test", "full_test"} <= set(e2e):
        result.comparison = write_comparison(
            ctx,
            result.run_dirs["baseline_test"],
            result.run_dirs["full_test"],
            result.run_dirs.get("guard_test"),
        )
    return result


def _rehearsal_state(ctx: _Ctx, run: PlannedRun) -> RunState:
    run_dir = ctx.runs_dir / f"{ctx.prefix}_{run.name}"
    state = read_json(run_dir / STATE) if (run_dir / STATE).exists() else {}
    if state.get("completed"):
        return RunState("done", 0, run_dir.as_posix())
    return RunState("resume" if run_dir.exists() else "new", 0, run_dir.as_posix())


def _decide(ctx: _Ctx, run: PlannedRun, state: RunState, rerun: Mapping[str, str]) -> Path:
    base = ctx.runs_dir / f"{ctx.prefix}_{run.name}"
    if run.name in rerun:
        if ctx.rehearsal:
            raise FinalError("--rerun applies to test runs only")
        if state.status != "failed":
            raise FinalError(f"{run.name} is {state.status}; only a failed run can be re-run")
        if state.attempt >= 1:
            raise FinalError(f"{run.name} was already re-run once; no further re-runs (ADR-0014)")
        if not rerun[run.name].strip():
            raise FinalError("--reason is required for a re-run")
        return base.with_name(f"{base.name}_rerun1")
    if state.status == "failed":
        raise FinalError(
            f"{run.name} failed ({state.note or 'see test_runs.log'}). Fix the infrastructure, "
            f"then re-run it once with --rerun {run.name} --reason '...'"
        )
    if (
        state.status == "resume"
        and not ctx.rehearsal
        and (state.commit != ctx.git.commit or state.hardware != ctx.hardware)
    ):
        raise FinalError(
            f"{run.name} was started on commit {state.commit} / {state.hardware}; "
            f"resume only on the same commit and hardware (now {ctx.git.commit} / "
            f"{ctx.hardware})"
        )
    return Path(state.run_dir) if state.run_dir else base


def _log(
    ctx: _Ctx,
    path: Path,
    run: PlannedRun,
    attempt: int,
    status: LogStatus,
    run_dir: Path,
    note: str = "",
) -> None:
    if ctx.rehearsal:
        return
    entry = LogEntry(
        time=datetime.now(UTC).isoformat(timespec="seconds"),
        plan_sha256=ctx.plan_sha,
        run=run.name,
        attempt=attempt,
        status=status,
        commit=ctx.git.commit,
        hardware=ctx.hardware,
        run_dir=run_dir.as_posix(),
        note=note,
    )
    append_jsonl(path, entry.model_dump(mode="json"))


def _record_deviation(
    ctx: _Ctx, run: PlannedRun, state: RunState, run_dir: Path, reason: str
) -> None:
    path = ctx.runs_dir / DEVIATIONS
    header = "" if path.exists() else "# Deviations from the pre-registered plan\n"
    entry = (
        f"\n## {datetime.now(UTC).date().isoformat()}: re-run of `{run.name}`\n\n"
        f"- Plan sha256: `{ctx.plan_sha}`; commit `{ctx.git.commit}`; hardware {ctx.hardware}\n"
        f"- Failed attempt: `{state.run_dir}` ({state.note or 'no detail'})\n"
        f"- New attempt: `{run_dir.as_posix()}`\n"
        f"- Reason (infrastructure only, never results): {reason.strip()}\n"
    )
    previous = read_text(path) if path.exists() else header
    write_text(path, previous + entry)


def _execute(ctx: _Ctx, run: PlannedRun, run_dir: Path) -> None:
    cfg = load_app_config(run.config)
    snapshot = {
        "app": cfg.model_dump(mode="json"),
        "models": ctx.profile.model_dump(mode="json"),
        "config_path": run.config.as_posix(),
        "plan": {"sha256": ctx.plan_sha, "run": run.name, "rehearsal": ctx.rehearsal},
    }
    if not run_dir.exists():
        run_dir.mkdir(parents=True)
        write_config_snapshot(run_dir, snapshot)
        write_json(
            run_dir / "run.json",
            {
                "kind": run.kind,
                "eval_set": run.eval_set,
                "split": ctx.split,
                "plan_sha256": ctx.plan_sha,
                "planned_run": run.name,
            },
        )
        uses_llm = run.kind == "e2e" or cfg.guard.mode in ("llm", "rule_llm")
        p, j = ctx.profile.pipeline, ctx.profile.judge
        write_json(
            run_dir / "fingerprint.json",
            build_fingerprint(
                evalset_dir=cfg.evalset.dir,
                eval_set=run.eval_set,
                split=ctx.split,
                metric_set="e2e-v1" if run.kind == "e2e" else "safety-v1",
                bootstrap_seed=cfg.eval.bootstrap_seed,
                config=snapshot,
                judge=j.model_dump() if j and run.kind == "e2e" else None,
                pipeline=(p.model_dump() if p and uses_llm else {"model": cfg.guard.mode}),
            ),
        )
    elif read_json(run_dir / "run.json").get("plan_sha256") != ctx.plan_sha:
        raise FinalError(f"{run_dir} belongs to another plan")
    if run.kind == "e2e":
        _execute_e2e(ctx, cfg, run_dir)
    else:
        _execute_safety(ctx, cfg, run_dir)
    write_json(run_dir / STATE, {"completed": True})


def _phase_done(run_dir: Path, phase: str) -> bool:
    state = read_json(run_dir / STATE) if (run_dir / STATE).exists() else {}
    return bool(state.get(phase))


def _mark(run_dir: Path, phase: str) -> None:
    state = read_json(run_dir / STATE) if (run_dir / STATE).exists() else {}
    write_json(run_dir / STATE, {**state, phase: True})


def _execute_e2e(ctx: _Ctx, cfg: AppConfig, run_dir: Path) -> None:
    items = [it for it in ctx.deps.golden(cfg) if it.split == ctx.split]
    if ctx.limit is not None:
        items = items[: ctx.limit]
    budget = f"e2e_{ctx.split}"
    if not _phase_done(run_dir, "predict"):
        pipeline = ctx.deps.make_pipeline(cfg, ctx.profile)
        try:
            with timed_phase(run_dir, "predict", budget, cfg.eval.budgets_s, len(items)):
                run_predict(items, pipeline, run_dir)
        finally:
            _close(pipeline)  # unloads the pipeline LLM before the judge is loaded
        failed = [p for p in load_predictions(run_dir) if p.status == "error"]
        if failed:
            raise InfraFailure(f"{len(failed)} prediction(s) failed, e.g. {failed[0].error}")
        _mark(run_dir, "predict")
    if not _phase_done(run_dir, "judge"):
        judge = ctx.deps.make_judge(cfg, ctx.profile)
        preds = load_predictions(run_dir)
        try:
            with timed_phase(run_dir, "judge", budget, cfg.eval.budgets_s, len(preds)):
                run_judge(items, preds, judge, run_dir)
        finally:
            _close(judge)
        infra = [j for j in load_judgments(run_dir) if (j.error or "").startswith(INFRA_ERROR)]
        if infra:
            raise InfraFailure(f"{len(infra)} judgment(s) failed, e.g. {infra[0].error}")
        _mark(run_dir, "judge")
    write_report(run_dir, cfg, items)


def _execute_safety(ctx: _Ctx, cfg: AppConfig, run_dir: Path) -> None:
    items = [it for it in ctx.deps.safety(cfg) if it.split == ctx.split]
    if ctx.limit is not None:
        items = items[: ctx.limit]
    guard = ctx.deps.make_guard(cfg, ctx.profile)
    with timed_phase(run_dir, "safety", f"safety_{ctx.split}", cfg.eval.budgets_s, len(items)):
        records = run_safety(items, guard)
    rows = per_item_rows(records)
    write_jsonl(run_dir / "per_item.jsonl", rows)
    write_json(run_dir / "metrics.json", aggregate(rows, cfg.eval))


# --- final comparison ----------------------------------------------------------------------

PRIMARY = ("faithfulness", "citation_precision", "abstention_f1", "evidence_recall@10")


def write_comparison(ctx: _Ctx, baseline: Path, full: Path, guard: Path | None) -> Path:
    """Paired bootstrap `full` (B) vs `baseline` (A): overall, by language, by category."""
    cfg = load_app_config(ctx.plan.runs[0].config).eval
    per_item = read_jsonl(baseline / "per_item.jsonl")
    languages = sorted({str(r["language"]) for r in per_item})
    categories = sorted({str(r["category"]) for r in per_item})
    blocks: list[tuple[str, dict[str, str] | None]] = [("overall", None)]
    blocks += [(f"language={x}", {"language": x}) for x in languages]
    blocks += [(f"category={x}", {"category": x}) for x in categories]
    lines = [
        "# Final evaluation: `full` (B) vs `baseline` (A)"
        + (" — REHEARSAL on dev" if ctx.rehearsal else ""),
        "",
        f"Plan sha256 `{ctx.plan_sha}` · commit `{ctx.git.commit}` · hardware {ctx.hardware} · "
        f"split `{ctx.split}`.",
        f"Runs: A = `{baseline.as_posix()}`, B = `{full.as_posix()}`. Paired bootstrap "
        f"({cfg.bootstrap_n} resamples, seed {cfg.bootstrap_seed}, {cfg.ci_level:.0%} CI); a "
        "difference whose CI contains 0 is not significant. Primary metrics (H1) are bold.",
        "",
    ]
    for title, where in blocks:
        cmp = compare_runs(baseline, full, cfg, where)
        lines += [f"## {title} (n = {cmp.n_common})", "", *_table(cmp), ""]
    if guard is not None and (guard / "metrics.json").exists():
        m = read_json(guard / "metrics.json")
        lines += [
            "## Guard (H2, safety split)",
            "",
            f"Run `{guard.as_posix()}`: red-flag recall {_ci(m['red_flag_recall'])}, "
            f"FPR {_ci(m['fpr'])} (targets: recall ≥ 0.90, FPR ≤ 0.15; spec §13).",
            "",
        ]
    path = ctx.runs_dir / f"{ctx.prefix}_comparison.md"
    write_text(path, "\n".join(lines))
    return path


def _ci(v: Mapping[str, Any]) -> str:
    if v.get("value") is None:
        return "n/a"
    return f"{v['value']:.3f} [{v['low']:.3f}, {v['high']:.3f}] (n = {v['n']})"


def _table(cmp: Comparison) -> list[str]:
    out = [
        "| Metric | A (baseline) | B (full) | B - A | 95% CI | Significant |",
        "|---|---|---|---|---|---|",
    ]
    order = [m for m in PRIMARY if m in cmp.metrics] + [m for m in cmp.metrics if m not in PRIMARY]
    for m in order:
        r = cmp.metrics[m]
        name = f"**{m}**" if m in PRIMARY else m
        sig = "yes" if r["significant"] else "no (CI contains 0)"
        out.append(
            f"| {name} | {r['mean_a']:.3f} | {r['mean_b']:.3f} | {r['diff']:+.3f} | "
            f"[{r['low']:+.3f}, {r['high']:+.3f}] | {sig} |"
        )
    return out
