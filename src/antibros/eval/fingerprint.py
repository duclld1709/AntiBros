"""Run fingerprints (spec §6.6). ``compare`` refuses runs whose *hard* fields differ, because
their numbers are not comparable (different questions, judge or statistics); *soft*
differences (pipeline model, hardware, commit) are what an experiment is allowed to vary."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from functools import cache
from pathlib import Path
from typing import Any

import antibros
from antibros.evalset.freeze import read_version, verify_checksums

HARD_FIELDS = (
    "evalset_version",
    "evalset_sha256",
    "corpus_manifest_sha256",
    "eval_set",
    "split",
    "metric_set",
    "judge_model",
    "judge_digest",
    "judge_temperature",
    "bootstrap_seed",
)
SOFT_FIELDS = (
    "pipeline_model",
    "pipeline_digest",
    "runtime_version",
    "hardware",
    "git_commit",
    "config_sha256",
)


@cache
def git_commit() -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout
        # Run outputs under experiments/ (written by the harness itself) are not code changes.
        dirty = [line for line in status.splitlines() if not line[3:].startswith("experiments/")]
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{sha}{'-dirty' if dirty else ''}"


@cache
def hardware() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return str(torch.cuda.get_device_name(0))
    except ImportError:  # pragma: no cover - torch is a dependency
        pass
    return f"cpu:{platform.machine()}"


@cache
def runtime_version() -> str:
    parts = [f"python {platform.python_version()}", f"antibros {antibros.__version__}"]
    try:
        import torch

        parts.append(f"torch {torch.__version__}")
    except ImportError:  # pragma: no cover
        pass
    return "; ".join(parts)


def sha256_json(obj: Any) -> str:
    blob = json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def evalset_identity(evalset_dir: Path) -> dict[str, str]:
    """Version and checksum of a frozen evalset; ``unfrozen`` markers otherwise."""
    version = read_version(evalset_dir)
    result = verify_checksums(evalset_dir)
    return {
        "evalset_version": version.get("version", "unfrozen"),
        "evalset_sha256": result.evalset_sha256
        if result.ok and result.evalset_sha256
        else "unfrozen",
        "corpus_manifest_sha256": version.get("corpus_manifest_sha256", "unfrozen"),
    }


def build_fingerprint(
    *,
    evalset_dir: Path,
    eval_set: str,
    split: str,
    metric_set: str,
    bootstrap_seed: int,
    config: dict[str, Any],
    judge: dict[str, Any] | None = None,
    pipeline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    judge = judge or {}
    pipeline = pipeline or {}
    return {
        **evalset_identity(evalset_dir),
        "eval_set": eval_set,
        "split": split,
        "metric_set": metric_set,
        "judge_model": judge.get("model"),
        "judge_digest": judge.get("digest"),
        "judge_temperature": judge.get("temperature"),
        "bootstrap_seed": bootstrap_seed,
        "pipeline_model": pipeline.get("model"),
        "pipeline_digest": pipeline.get("digest"),
        "runtime_version": runtime_version(),
        "hardware": hardware(),
        "git_commit": git_commit(),
        "config_sha256": sha256_json(config),
    }


def compare_fingerprints(a: dict[str, Any], b: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Return (hard, soft) field names whose values differ."""
    hard = [f for f in HARD_FIELDS if a.get(f) != b.get(f)]
    soft = [f for f in SOFT_FIELDS if a.get(f) != b.get(f)]
    return hard, soft
