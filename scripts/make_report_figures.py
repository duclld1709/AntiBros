"""Render the M6 report figures (SVG) from stored run artifacts in ``experiments/``.

Every figure is an interval plot: a dot for the estimate and a 2px line for its 95%
bootstrap CI, one row per run or comparison, with the value printed beside it (so no number
depends on reading the axis). Difference plots draw a zero line; a filled dot means the CI
excludes 0, a hollow dot that it does not. Each mark carries an SVG ``<title>`` tooltip and
the same numbers are tabulated in ``reports/ablation.md``. Light and dark colours follow
``prefers-color-scheme``. No model is called; no dependency beyond the project itself.

    uv run python scripts/make_report_figures.py
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

from antibros.config import load_app_config
from antibros.eval.compare import compare_runs
from antibros.io import read_json, write_text

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments"
OUT = ROOT / "reports" / "figures"

STYLE = """
<style>
  .surface { fill: #fcfcfb; }
  .ink { fill: #0b0b0b; font: 600 13px system-ui, -apple-system, 'Segoe UI', sans-serif; }
  .label { fill: #0b0b0b; font: 12px system-ui, -apple-system, 'Segoe UI', sans-serif; }
  .muted { fill: #52514e; font: 11px system-ui, -apple-system, 'Segoe UI', sans-serif; }
  .grid { stroke: #e4e3df; stroke-width: 1; }
  .zero { stroke: #8a8984; stroke-width: 1; stroke-dasharray: 3 3; }
  .ci { stroke: #2a78d6; stroke-width: 2; stroke-linecap: round; }
  .dot { fill: #2a78d6; stroke: #fcfcfb; stroke-width: 2; }
  .hollow { fill: #fcfcfb; stroke: #2a78d6; stroke-width: 2; }
  @media (prefers-color-scheme: dark) {
    .surface { fill: #1a1a19; }
    .ink, .label { fill: #f0efec; }
    .muted { fill: #c3c2b7; }
    .grid { stroke: #34342f; }
    .zero { stroke: #8f8e86; }
    .ci { stroke: #3987e5; }
    .dot { fill: #3987e5; stroke: #1a1a19; }
    .hollow { fill: #1a1a19; stroke: #3987e5; }
  }
</style>
"""


@dataclass(frozen=True)
class Row:
    label: str
    value: float
    low: float
    high: float
    note: str = ""


@dataclass(frozen=True)
class Panel:
    title: str
    rows: Sequence[Row]
    lo: float
    hi: float
    difference: bool = False  # zero line + filled/hollow by significance


LABEL_W, PLOT_W, NOTE_W, ROW_H, HEAD_H, AXIS_H, PAD = 230, 330, 190, 26, 30, 26, 14


def _ticks(lo: float, hi: float) -> list[float]:
    span = hi - lo
    step = 10 ** math.floor(math.log10(span / 4))
    for mult in (1, 2, 2.5, 5, 10):
        if span / (step * mult) <= 5:
            step *= mult
            break
    first = math.ceil(lo / step - 1e-9) * step
    out, t = [], first
    while t <= hi + 1e-9:
        out.append(round(t, 6))
        t += step
    return out


def render(panels: Sequence[Panel], description: str) -> str:
    width = PAD * 2 + LABEL_W + PLOT_W + NOTE_W
    height = PAD + sum(HEAD_H + len(p.rows) * ROW_H + AXIS_H + 10 for p in panels) + PAD
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" aria-label="{escape(description)}">',
        f"<title>{escape(description)}</title>",
        STYLE,
        f'<rect class="surface" x="0" y="0" width="{width}" height="{height}" rx="8"/>',
    ]
    y = PAD
    x0 = PAD + LABEL_W
    for p in panels:

        def sx(v: float, p: Panel = p) -> float:
            v = min(max(v, p.lo), p.hi)
            return x0 + (v - p.lo) / (p.hi - p.lo) * PLOT_W

        parts.append(f'<text class="ink" x="{PAD}" y="{y + 18}">{escape(p.title)}</text>')
        top = y + HEAD_H
        bottom = top + len(p.rows) * ROW_H
        for t in _ticks(p.lo, p.hi):
            parts.append(
                f'<line class="grid" x1="{sx(t):.1f}" y1="{top}" x2="{sx(t):.1f}" y2="{bottom}"/>'
            )
            parts.append(
                f'<text class="muted" x="{sx(t):.1f}" y="{bottom + 16}" '
                f'text-anchor="middle">{t:g}</text>'
            )
        if p.difference and p.lo < 0 < p.hi:
            parts.append(
                f'<line class="zero" x1="{sx(0):.1f}" y1="{top - 4}" x2="{sx(0):.1f}" '
                f'y2="{bottom}"/>'
            )
        for i, r in enumerate(p.rows):
            cy = top + i * ROW_H + ROW_H / 2
            significant = not (r.low <= 0 <= r.high)
            dot = "dot" if (not p.difference or significant) else "hollow"
            text = f"{r.value:.3f} [{r.low:.3f}, {r.high:.3f}]"
            tip = f"{r.label}: {text}" + (f" ({r.note})" if r.note else "")
            parts += [
                f"<g><title>{escape(tip)}</title>",
                f'<text class="label" x="{PAD}" y="{cy + 4:.1f}">{escape(r.label)}</text>',
                f'<line class="ci" x1="{sx(r.low):.1f}" y1="{cy:.1f}" x2="{sx(r.high):.1f}" '
                f'y2="{cy:.1f}"/>',
                f'<circle class="{dot}" cx="{sx(r.value):.1f}" cy="{cy:.1f}" r="5"/>',
                f'<text class="muted" x="{x0 + PLOT_W + 12}" y="{cy + 4:.1f}">'
                f"{escape(text)}{escape(' ' + r.note) if r.note else ''}</text>",
                "</g>",
            ]
        y = bottom + AXIS_H + 10
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def _ci(block: dict[str, Any]) -> tuple[float, float, float]:
    return float(block["value"]), float(block["low"]), float(block["high"])


def run_dir(suffix: str) -> Path:
    matches = sorted(EXP.glob(f"*_{suffix}"))
    if not matches:
        raise SystemExit(f"no run directory ending in _{suffix}")
    return matches[-1]


RETRIEVAL = [
    ("hybrid (reference)", "retr_hybrid"),
    ("fixed chunks", "retr_fixed"),
    ("recursive chunks", "retr_recursive"),
    ("sparse only", "retr_sparse"),
    ("dense bge-m3", "retr_dense"),
    ("dense e5-large", "retr_dense_e5"),
    ("hybrid + reranker", "retr_rerank"),
]
LOO = [
    ("without reranker", "m4_loo_no_rerank"),
    ("without query rewrite", "m4_loo_no_rewrite"),
    ("without grade_context", "m4_loo_no_grade"),
    ("without verify_citations", "m4_loo_no_verify"),
]
GUARDS = [("rule", "guard_rule"), ("llm", "guard_llm"), ("rule_llm (selected)", "guard_rule_llm")]


def retrieval_figure() -> str:
    panels = []
    for set_name in ("golden", "synth"):
        rows = []
        for label, tag in RETRIEVAL:
            m = read_json(run_dir(f"{tag}_{set_name}") / "metrics.json")["overall"]
            v, lo, hi = _ci(m["evidence_recall@10"])
            rows.append(Row(label, v, lo, hi))
        n = read_json(run_dir(f"retr_hybrid_{set_name}") / "metrics.json")["n_queries"]
        panels.append(Panel(f"Evidence Recall@10, {set_name} dev (n = {n})", rows, 0.0, 1.0))
    return render(panels, "Retrieval ablation on dev: Evidence Recall@10 with 95% CIs")


def paired(a: Path, b: Path, metric: str) -> Row | None:
    cfg = load_app_config(ROOT / "configs" / "base.yaml").eval
    r = compare_runs(a, b, cfg).metrics.get(metric)
    if r is None:
        return None
    return Row("", float(r["diff"]), float(r["low"]), float(r["high"]), f"n = {r['n']}")


def loo_figure() -> str:
    full = run_dir("m4_full")
    panels = []
    for metric, lo, hi in (
        ("evidence_recall@10", -0.5, 0.5),
        ("faithfulness", -0.5, 0.5),
        ("abstention_f1", -0.6, 0.6),
        ("false_abstain", -0.6, 0.6),
    ):
        rows = []
        for label, tag in LOO:
            r = paired(full, run_dir(tag), metric)
            if r is not None:
                rows.append(Row(label, r.value, r.low, r.high, r.note))
        panels.append(Panel(f"{metric}: variant - full (dev)", rows, lo, hi, difference=True))
    return render(panels, "Leave-one-out on dev: paired differences to full with 95% CIs")


def final_figure() -> str:
    a = EXP / "final-c01d215e_baseline_test"
    b = EXP / "final-c01d215e_full_test_rerun1"
    cfg = load_app_config(ROOT / "configs" / "base.yaml").eval
    rows = []
    for label, metric, where in (
        ("faithfulness", "faithfulness", None),
        ("citation precision", "citation_precision", None),
        ("abstention F1", "abstention_f1", None),
        ("Evidence Recall@10", "evidence_recall@10", None),
        ("Evidence Recall@10, Vietnamese", "evidence_recall@10", {"language": "vi"}),
        ("false-abstain rate", "false_abstain", None),
    ):
        r = compare_runs(a, b, cfg, where).metrics[metric]
        rows.append(Row(label, r["diff"], r["low"], r["high"], f"n = {r['n']}"))
    return render(
        [Panel("full - baseline on golden test (H1)", rows, -0.4, 0.4, difference=True)],
        "Final test: paired differences full minus baseline with 95% CIs",
    )


def guard_figure() -> str:
    panels = []
    for metric, title in (("red_flag_recall", "red-flag recall"), ("fpr", "false-positive rate")):
        rows = []
        for label, tag in GUARDS:
            m = read_json(run_dir(tag) / "metrics.json")
            v, lo, hi = _ci(m[metric])
            rows.append(Row(f"dev: {label}", v, lo, hi, f"n = {m[metric]['n']}"))
        m = read_json(EXP / "final-c01d215e_guard_test" / "metrics.json")
        v, lo, hi = _ci(m[metric])
        rows.append(Row("test: rule_llm", v, lo, hi, f"n = {m[metric]['n']}"))
        panels.append(Panel(title, rows, 0.0, 1.0))
    return render(panels, "Guard variants: red-flag recall and FPR with 95% CIs")


def main() -> None:
    figures = {
        "retrieval_ablation.svg": retrieval_figure(),
        "loo_dev.svg": loo_figure(),
        "final_test.svg": final_figure(),
        "guard.svg": guard_figure(),
    }
    for name, svg in figures.items():
        write_text(OUT / name, svg)
        print(f"wrote {(OUT / name).relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
