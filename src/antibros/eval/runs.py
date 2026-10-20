"""Experiment run directories, config snapshots and ``timing.json`` with budget checks (§6.7-6.8).

A run lives in ``experiments/<YYYYMMDD-HHMM>_<tag>/``. Phases of the same run (``predict``
then ``judge``) add to one ``timing.json``; the budget applies to their sum.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from antibros.io import is_valid_run_tag, read_json, write_json, write_text
from antibros.logging import get_logger

log = get_logger(__name__)

TEST_SPLIT_REFUSAL = (
    "Refusing to run on the test split. The test split only runs via "
    "`antibros eval final --plan configs/final_eval_plan.yaml` (spec §6.5)."
)


class SplitRefusedError(RuntimeError):
    pass


def guard_split(split: str, allow_test: bool = False) -> None:
    """Every eval entry point calls this; only ``eval final`` passes ``allow_test=True``."""
    if split == "test" and not allow_test:
        raise SplitRefusedError(TEST_SPLIT_REFUSAL)


def new_run_dir(runs_dir: Path, tag: str, now: datetime | None = None) -> Path:
    if not is_valid_run_tag(tag):
        raise ValueError(f"invalid run tag {tag!r}: use only [A-Za-z0-9_-]")
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M")
    base = runs_dir / f"{stamp}_{tag}"
    path, n = base, 2
    while path.exists():
        path = base.with_name(f"{base.name}-{n}")
        n += 1
    path.mkdir(parents=True)
    return path


def write_config_snapshot(run_dir: Path, config: Mapping[str, Any]) -> None:
    write_text(
        run_dir / "config.yaml",
        yaml.safe_dump(dict(config), allow_unicode=True, sort_keys=True, default_flow_style=False),
    )


@dataclass
class BudgetStatus:
    total_s: float
    budget_s: float | None
    over_budget: bool


@contextmanager
def timed_phase(
    run_dir: Path,
    phase: str,
    budget_key: str,
    budgets: dict[str, float],
    n_items: int,
) -> Iterator[dict[str, Any]]:
    """Time a phase and merge it into ``timing.json``; warn when the run exceeds its budget.

    The yielded dict can be filled with extra per-phase numbers (e.g. ``cached`` counts).
    Timing is written even if the phase is interrupted, marked ``completed: false``."""
    extra: dict[str, Any] = {}
    started = datetime.now(UTC).isoformat(timespec="seconds")
    t0 = time.perf_counter()
    completed = False
    try:
        yield extra
        completed = True
    finally:
        seconds = round(time.perf_counter() - t0, 2)
        path = run_dir / "timing.json"
        data: dict[str, Any] = read_json(path) if path.exists() else {"phases": {}}
        prev = data["phases"].get(phase, {})
        data["phases"][phase] = {
            "started_at": prev.get("started_at", started),
            "seconds": round(prev.get("seconds", 0.0) + seconds, 2),  # resumed runs add up
            "n_items": n_items,
            "completed": completed,
            **extra,
        }
        total = round(sum(p["seconds"] for p in data["phases"].values()), 2)
        budget = budgets.get(budget_key)
        data.update(
            {
                "total_s": total,
                "budget_key": budget_key,
                "budget_s": budget,
                "over_budget": budget is not None and total > budget,
            }
        )
        write_json(path, data)
        if data["over_budget"]:
            log.warning(
                "over_time_budget",
                run=str(run_dir),
                total_s=total,
                budget_s=budget,
                hint="apply the pre-declared mitigation order in spec §6.8 and record an ADR",
            )


def budget_status(run_dir: Path) -> BudgetStatus:
    data = read_json(run_dir / "timing.json")
    return BudgetStatus(data["total_s"], data.get("budget_s"), bool(data.get("over_budget")))
