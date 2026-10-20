"""``antibros compare``: paired bootstrap between two runs on the same questions (spec §6.3).

Refuses when hard fingerprint fields differ (different evalset, split, metric set, judge or
seed): those numbers are not comparable. Soft differences are what the experiment varies and
are only reported. Differences whose CI contains 0 are flagged as not significant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from antibros.config import EvalConfig
from antibros.eval.bootstrap import paired_bootstrap
from antibros.eval.fingerprint import compare_fingerprints
from antibros.eval.report import is_number
from antibros.io import read_json, read_jsonl

PER_ITEM = "per_item.jsonl"
NON_METRIC_KEYS = {"id", "language", "category", "status", "n_spans", "retrieved"}


class FingerprintMismatch(ValueError):
    def __init__(self, fields: list[str]):
        super().__init__(f"hard fingerprint fields differ: {', '.join(fields)}")
        self.fields = fields


@dataclass
class Comparison:
    n_common: int
    soft_differences: list[str]
    metrics: dict[str, dict[str, Any]] = field(default_factory=dict)


def compare_runs(run_a: Path, run_b: Path, cfg: EvalConfig) -> Comparison:
    fa, fb = read_json(run_a / "fingerprint.json"), read_json(run_b / "fingerprint.json")
    hard, soft = compare_fingerprints(fa, fb)
    if hard:
        raise FingerprintMismatch(hard)
    rows_a = {r["id"]: r for r in read_jsonl(run_a / PER_ITEM)}
    rows_b = {r["id"]: r for r in read_jsonl(run_b / PER_ITEM)}
    common = sorted(set(rows_a) & set(rows_b))
    metrics = sorted(
        k
        for k in set().union(*(r.keys() for r in rows_a.values()))
        if k not in NON_METRIC_KEYS and not k.startswith(("expected_", "predicted_"))
    )
    result = Comparison(n_common=len(common), soft_differences=soft)
    for m in metrics:
        pairs = [
            (rows_a[i][m], rows_b[i][m])
            for i in common
            if is_number(rows_a[i].get(m)) and is_number(rows_b[i].get(m))
        ]
        if not pairs:
            continue
        a, b = zip(*pairs, strict=True)
        res = paired_bootstrap(list(a), list(b), cfg.bootstrap_n, cfg.bootstrap_seed, cfg.ci_level)
        result.metrics[m] = {
            "mean_a": round(sum(a) / len(a), 6),
            "mean_b": round(sum(b) / len(b), 6),
            **res.to_dict(),
        }
    return result


def render_markdown(cmp: Comparison, name_a: str, name_b: str) -> str:
    lines = [
        f"# Paired comparison: `{name_a}` (A) vs `{name_b}` (B)",
        "",
        f"Common items: {cmp.n_common}. Soft fingerprint differences: "
        f"{', '.join(cmp.soft_differences) or 'none'}.",
        "",
        "| Metric | A | B | B - A | 95% CI | Significant |",
        "|---|---|---|---|---|---|",
    ]
    for m, r in cmp.metrics.items():
        sig = "yes" if r["significant"] else "no (CI contains 0)"
        lines.append(
            f"| {m} | {r['mean_a']:.3f} | {r['mean_b']:.3f} | {r['diff']:+.3f} | "
            f"[{r['low']:+.3f}, {r['high']:+.3f}] | {sig} |"
        )
    return "\n".join(lines) + "\n"
