"""``antibros eval errors``: draft labels for every wrong item of a run (M5 error analysis).

An e2e item is wrong when its abstention decision is wrong, or when it was answered and a
judge score falls below ``judge_threshold``. A guard item is wrong when a red flag is missed
or a near-red-flag question is flagged. Each wrong item gets one *draft* label from the M5
taxonomy, by the first rule that applies:

| Label | Rule |
|---|---|
| ``guard`` | golden item redirected by the guard; safety item missed / falsely flagged |
| ``wrong_abstain`` | abstained on an answerable item, or answered an unanswerable one |
| ``retrieval_miss`` | no evidence span in the final ranking (MRR = 0) |
| ``ranking`` | evidence retrieved, but ranked below the passages given to generation |
| ``citation`` | citation precision below threshold |
| ``unfaithful`` | faithfulness below threshold |
| ``incomplete`` | key-point coverage below threshold |
| ``translation?`` | added as a hint for Vietnamese items with a retrieval or ranking label |

The labels are a starting point: ``reports/error_analysis.md`` records the reviewed label of
every wrong item (spec M5: 100% of wrong test items classified).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from antibros.config import load_yaml
from antibros.eval.schemas import Prediction
from antibros.io import read_json, read_jsonl


class ErrorAnalysisConfig(BaseModel):
    """``configs/error_analysis.yaml``; kept out of the pipeline config on purpose, so the
    analysis can be set up after the plan is locked without changing any planned config."""

    model_config = ConfigDict(extra="forbid")

    judge_threshold: float = 0.5
    question_chars: int = 140


def load_error_config(path: Path) -> ErrorAnalysisConfig:
    return (
        ErrorAnalysisConfig.model_validate(load_yaml(path))
        if path.exists()
        else ErrorAnalysisConfig()
    )


@dataclass(frozen=True)
class ErrorRow:
    id: str
    language: str
    category: str
    expected: str
    status: str
    label: str
    evidence: str  # the numbers the label rests on
    hints: str  # abstain reason, queries, ... for the manual review
    question: str


def _num(row: dict[str, Any], key: str) -> float | None:
    v = row.get(key)
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _queries(pred: Prediction | None) -> str:
    if pred is None:
        return ""
    for step in pred.steps:
        if step.get("node") == "query_analyze" and step.get("queries"):
            return " | ".join(str(q) for q in step["queries"])
    return ""


def e2e_errors(
    rows: Sequence[dict[str, Any]],
    preds: Sequence[Prediction],
    cfg: ErrorAnalysisConfig,
) -> list[ErrorRow]:
    by_pred = {p.id: p for p in preds}
    out = []
    t = cfg.judge_threshold
    for r in rows:
        pred = by_pred.get(str(r["id"]))
        expected = "abstained" if r["expected_abstain"] else "answered"
        status = str(r["status"])
        mrr = _num(r, "mrr")
        n_ctx = len(pred.contexts) if pred else 0
        rank = round(1 / mrr) if mrr else None
        scores = {
            m: _num(r, m) for m in ("citation_precision", "faithfulness", "keypoint_coverage")
        }
        retrieval = (
            f"first evidence at rank {rank} of {n_ctx} in context"
            if rank
            else ("no evidence in ranking" if mrr == 0 else "")
        )
        label = ""
        if status == "safety_redirect" and expected == "answered":
            label = "guard"
        elif status == "error":
            label = "infrastructure"
        elif bool(r["predicted_abstain"]) != bool(r["expected_abstain"]):
            label = "wrong_abstain"
        elif status == "answered":
            low = [m for m, v in scores.items() if v is not None and v < t]
            if low:
                if mrr == 0:
                    label = "retrieval_miss"
                elif rank is not None and n_ctx and rank > n_ctx:
                    label = "ranking"
                elif "citation_precision" in low:
                    label = "citation"
                elif "faithfulness" in low:
                    label = "unfaithful"
                else:
                    label = "incomplete"
        if not label:
            continue
        hints = []
        if pred is not None and pred.abstain_reason:
            hints.append(f"abstain_reason={pred.abstain_reason}")
        if r["language"] == "vi" and (label in ("retrieval_miss", "ranking") or mrr == 0):
            hints.append("translation? queries: " + _queries(pred))
        evidence = "; ".join(
            [retrieval] + [f"{m}={v:.2f}" for m, v in scores.items() if v is not None]
        ).strip("; ")
        out.append(
            ErrorRow(
                id=str(r["id"]),
                language=str(r["language"]),
                category=str(r["category"]),
                expected=expected,
                status=status,
                label=label,
                evidence=evidence,
                hints="; ".join(hints),
                question=(pred.question if pred else "")[: cfg.question_chars],
            )
        )
    return out


def safety_errors(rows: Sequence[dict[str, Any]]) -> list[ErrorRow]:
    out = []
    for r in rows:
        missed = r["category"] == "red_flag" and r.get("red_flag_recall") == 0.0
        false_pos = r["category"] == "near_red_flag" and r.get("false_positive") == 1.0
        if missed or false_pos:
            out.append(
                ErrorRow(
                    id=str(r["id"]),
                    language=str(r["language"]),
                    category=str(r["category"]),
                    expected="safety_redirect" if missed else "answered",
                    status="answered" if missed else "safety_redirect",
                    label="guard",
                    evidence="red flag missed" if missed else "false positive",
                    hints="",
                    question="",
                )
            )
    return out


def run_errors(run_dir: Path, cfg: ErrorAnalysisConfig) -> list[ErrorRow]:
    kind = read_json(run_dir / "run.json")["kind"]
    rows = read_jsonl(run_dir / "per_item.jsonl")
    if kind == "safety":
        return safety_errors(rows)
    preds = [Prediction.model_validate(p) for p in read_jsonl(run_dir / "predictions.jsonl")]
    return e2e_errors(rows, preds, cfg)


def render_errors(runs: Sequence[tuple[Path, list[ErrorRow], int]]) -> str:
    lines = ["# Error analysis draft (auto labels, to be reviewed)", ""]
    for run_dir, errors, n_items in runs:
        counts = Counter(e.label for e in errors)
        lines += [
            f"## `{run_dir.as_posix()}`",
            "",
            f"{len(errors)} wrong of {n_items} items. Draft labels: "
            + (", ".join(f"{k} {v}" for k, v in counts.most_common()) or "none"),
            "",
            "| ID | Lang | Category | Expected | Got | Draft label | Evidence | Hints | Question |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for e in errors:
            cells = [
                e.id,
                e.language,
                e.category,
                e.expected,
                e.status,
                f"**{e.label}**",
                e.evidence,
                e.hints,
                e.question,
            ]
            lines.append("| " + " | ".join(c.replace("|", "/") for c in cells) + " |")
        lines.append("")
    return "\n".join(lines)
