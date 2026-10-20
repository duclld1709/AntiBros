"""Pre-registered final evaluation plan (spec §7, ADR-0014).

``configs/final_eval_plan.yaml`` is written by hand as a *draft* during M4 (hypotheses,
planned runs, guard-selection rule). ``antibros plan lock`` then, on the laptop and after
the dev guard runs exist:

1. applies the declared guard-selection rule to the dev safety runs;
2. records the sha256 of every planned config (resolved config and each file it extends),
   the model profile with its digests, and the evalset identity;
3. sets ``status: locked``. A locked plan is never re-locked; ``eval final`` (M5) checks
   these hashes before touching the test split.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from antibros.config import load_app_config, load_model_profile, resolve_config
from antibros.eval.fingerprint import evalset_identity, sha256_json
from antibros.io import read_json, write_text


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PlannedRun(_Strict):
    name: str
    kind: Literal["e2e", "safety"]
    config: Path
    eval_set: Literal["golden", "safety"]
    split: Literal["test"] = "test"
    config_sha256: str | None = None  # sha256 of the resolved config (filled by lock)
    config_files: dict[str, str] = Field(default_factory=dict)  # file -> sha256 of its bytes


class Hypothesis(_Strict):
    id: str
    statement: str
    runs: list[str]
    metrics: list[str]


class GuardCandidateResult(_Strict):
    config: Path
    run: str
    red_flag_recall: float
    fpr: float
    latency_ms: float


class GuardSelection(_Strict):
    rule: str
    max_fpr: float = 0.15
    candidates: list[Path]
    dev_results: list[GuardCandidateResult] = Field(default_factory=list)
    selected: Path | None = None


class FinalEvalPlan(_Strict):
    version: int = 1
    status: Literal["draft", "locked"] = "draft"
    models_profile: Path
    models: dict[str, Any] = Field(default_factory=dict)  # role -> {model, digest, ...}
    evalset: dict[str, str] = Field(default_factory=dict)
    hypotheses: list[Hypothesis]
    runs: list[PlannedRun]
    guard_selection: GuardSelection
    notes: str = ""


class PlanError(ValueError):
    pass


def load_plan(path: Path) -> FinalEvalPlan:
    return FinalEvalPlan.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def save_plan(path: Path, plan: FinalEvalPlan) -> None:
    header = (
        "# Pre-registered final evaluation plan (spec §7, ADR-0014). Do not edit once locked.\n"
    )
    body = yaml.safe_dump(
        plan.model_dump(mode="json"), allow_unicode=True, sort_keys=False, width=100
    )
    write_text(path, header + body)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def config_hashes(path: Path) -> tuple[str, dict[str, str]]:
    """sha256 of the resolved, validated config plus the byte sha256 of every file in its
    ``extends`` chain (paths as POSIX strings, so the plan reads the same on every OS)."""
    _, files = resolve_config(path)
    resolved = load_app_config(path).model_dump(mode="json")
    return sha256_json(resolved), {f.as_posix(): file_sha256(f) for f in files}


def guard_result(run_dir: Path) -> GuardCandidateResult:
    metrics = read_json(run_dir / "metrics.json")
    snapshot = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    meta = read_json(run_dir / "run.json")
    if meta.get("kind") != "safety" or meta.get("split") != "dev":
        raise PlanError(f"{run_dir} is not a dev safety run")
    return GuardCandidateResult(
        config=Path(snapshot["config_path"]),
        run=run_dir.as_posix(),
        red_flag_recall=float(metrics["red_flag_recall"]["value"]),
        fpr=float(metrics["fpr"]["value"]),
        latency_ms=float(metrics["latency_ms"]["value"]),
    )


def select_guard(
    results: Sequence[GuardCandidateResult], max_fpr: float
) -> GuardCandidateResult | None:
    """Spec §7.1: highest red-flag recall among variants with FPR <= ``max_fpr``; ties go to
    the lower mean latency. ``None`` when no variant meets the FPR bound."""
    eligible = [r for r in results if r.fpr <= max_fpr]
    if not eligible:
        return None
    return min(eligible, key=lambda r: (-r.red_flag_recall, r.latency_ms))


def lock_plan(plan: FinalEvalPlan, guard_runs: Sequence[Path], evalset_dir: Path) -> FinalEvalPlan:
    if plan.status == "locked":
        raise PlanError("plan is already locked; a locked plan is never changed")
    gs = plan.guard_selection
    results = [guard_result(r) for r in guard_runs]
    candidates = {c.as_posix() for c in gs.candidates}
    seen = {r.config.as_posix() for r in results}
    if seen != candidates:
        raise PlanError(f"dev guard runs cover {sorted(seen)}, plan declares {sorted(candidates)}")
    chosen = select_guard(results, gs.max_fpr)
    if chosen is None:
        raise PlanError(
            f"no guard variant has FPR <= {gs.max_fpr} on dev; the rule selects nothing. "
            "Record this as a deviation and ask the project owner."
        )
    selected_mode = load_app_config(chosen.config).guard.mode
    for run in plan.runs:
        run_cfg = load_app_config(run.config)
        if (
            run.kind == "e2e"
            and run_cfg.guard.input_enabled
            and run_cfg.guard.mode != selected_mode
        ):
            raise PlanError(
                f"{run.config} uses guard.mode={run_cfg.guard.mode!r} but the rule selected "
                f"{selected_mode!r}; set it before locking"
            )
    runs = [r for r in plan.runs if r.kind != "safety"]
    runs.append(
        PlannedRun(name="guard_test", kind="safety", config=chosen.config, eval_set="safety")
    )
    locked_runs = []
    for run in runs:
        sha, files = config_hashes(run.config)
        locked_runs.append(run.model_copy(update={"config_sha256": sha, "config_files": files}))
    profile = load_model_profile(plan.models_profile)
    models = {
        role: spec.model_dump(mode="json")
        for role, spec in (("pipeline", profile.pipeline), ("judge", profile.judge))
        if spec is not None
    }
    missing = [role for role, m in models.items() if not m.get("digest")]
    if missing:
        raise PlanError(f"model digests missing for {missing} in {plan.models_profile}")
    return plan.model_copy(
        update={
            "status": "locked",
            "models": models | {"profile_sha256": file_sha256(plan.models_profile)},
            "evalset": evalset_identity(evalset_dir),
            "runs": locked_runs,
            "guard_selection": gs.model_copy(
                update={"dev_results": results, "selected": chosen.config}
            ),
        }
    )
