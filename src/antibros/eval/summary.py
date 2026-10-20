"""``antibros summarize``: markdown tables (value [95% CI]) over several runs of one kind.

Retrieval and e2e runs get one table per block (``overall``, ``language=vi``, ...), so the
Vietnamese subset needed by the embedding ablation is reported next to the overall numbers.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from antibros.io import read_json

E2E_METRICS = (
    "faithfulness",
    "keypoint_coverage",
    "citation_precision",
    "citation_recall",
    "citation_validity",
    "abstention_f1",
    "false_abstain",
    "latency_p95_ms",
)
SAFETY_METRICS = ("red_flag_recall", "fpr", "latency_ms")


def _cell(v: dict[str, Any] | None) -> str:
    if not v or v.get("value") is None:
        return "n/a"
    return f"{v['value']:.3f} [{v['low']:.3f}, {v['high']:.3f}]"


def _table(
    title: str, names: Sequence[str], blocks: Sequence[dict[str, Any]], cols: Sequence[str]
) -> list[str]:
    lines = [f"### {title}", "", "| Run | " + " | ".join(cols) + " |"]
    lines.append("|---" * (len(cols) + 1) + "|")
    for name, block in zip(names, blocks, strict=True):
        lines.append(f"| `{name}` | " + " | ".join(_cell(block.get(c)) for c in cols) + " |")
    return [*lines, ""]


def render_summary(runs: Sequence[Path]) -> str:
    kinds = {read_json(r / "run.json")["kind"] for r in runs}
    if len(kinds) != 1:
        raise ValueError(f"runs mix kinds {sorted(kinds)}; summarize one kind at a time")
    kind = kinds.pop()
    metrics = [read_json(r / "metrics.json") for r in runs]
    names = [r.name for r in runs]
    lines = [f"## Summary ({kind}, 95% bootstrap CIs)", ""]
    if kind == "safety":
        return "\n".join(lines + _table("overall", names, metrics, SAFETY_METRICS))
    if kind == "retrieval":
        first = metrics[0]["overall"]
        cols: Sequence[str] = list(first)
    else:
        cols = E2E_METRICS
    blocks = sorted({k for m in metrics for k in m if k == "overall" or k.startswith("language=")})
    for block in blocks:
        lines += _table(block, names, [m.get(block, {}) for m in metrics], cols)
    return "\n".join(lines)
