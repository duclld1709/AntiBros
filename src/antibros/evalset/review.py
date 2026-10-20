"""``review_log.jsonl``: every verifier verdict and every keep/revise/drop decision (ADR-0010).

Verdicts are tied to an item *revision* (hash of its content), so editing an item makes its
old verdicts stale and the new text must be verified again. Rounds count verified revisions:
round 1 is the original draft, rounds 2..(1 + max_revisions) are revisions.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from antibros.evalset.schemas import AnyItem, SetName
from antibros.io import append_jsonl, read_jsonl

Decision = Literal["keep", "revise", "drop"]
Status = Literal["kept", "dropped", "pending", "rejected"]


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass
class ItemHistory:
    verdicts: list[dict[str, Any]] = field(default_factory=list)
    decisions: list[dict[str, Any]] = field(default_factory=list)

    def revisions(self) -> list[str]:
        """Distinct verified revisions in log order."""
        seen: list[str] = []
        for v in self.verdicts:
            if v["revision"] not in seen:
                seen.append(v["revision"])
        return seen


class ReviewLog:
    def __init__(self, path: Path):
        self.path = path
        self.history: dict[tuple[str, str], ItemHistory] = defaultdict(ItemHistory)
        for row in read_jsonl(path):
            key = (row["set"], row["item_id"])
            if row["kind"] == "verdict":
                self.history[key].verdicts.append(row)
            else:
                self.history[key].decisions.append(row)

    def get(self, name: SetName, item_id: str) -> ItemHistory:
        return self.history.get((name, item_id), ItemHistory())

    def verdicts_for(self, name: SetName, item: AnyItem) -> dict[str, dict[str, Any]]:
        """Latest verdict per verifier for the item's current revision."""
        rev = item.revision()
        out: dict[str, dict[str, Any]] = {}
        for v in self.get(name, item.id).verdicts:
            if v["revision"] == rev:
                out[v["verifier"]] = v
        return out

    def round_for(self, name: SetName, item: AnyItem) -> int:
        revs = self.get(name, item.id).revisions()
        rev = item.revision()
        return revs.index(rev) + 1 if rev in revs else len(revs) + 1

    def status(self, name: SetName, item: AnyItem, verifiers: list[str]) -> Status:
        hist = self.get(name, item.id)
        if hist.decisions and hist.decisions[-1]["decision"] == "drop":
            return "dropped"
        verdicts = self.verdicts_for(name, item)
        if not verifiers or any(v not in verdicts for v in verifiers):
            return "pending"
        if all(verdicts[v]["accepted"] for v in verifiers):
            return "kept"
        return "rejected"

    def add_verdict(
        self,
        name: SetName,
        item: AnyItem,
        verifier: str,
        verdict: dict[str, Any],
        accepted: bool,
        reasons: list[str],
    ) -> dict[str, Any]:
        row = {
            "kind": "verdict",
            "set": name,
            "item_id": item.id,
            "revision": item.revision(),
            "round": self.round_for(name, item),
            "verifier": verifier,
            "accepted": accepted,
            "reasons": reasons,
            "verdict": verdict,
            "ts": now(),
        }
        append_jsonl(self.path, row)
        self.history[(name, item.id)].verdicts.append(row)
        return row

    def add_decision(
        self, name: SetName, item: AnyItem, decision: Decision, by: str, note: str = ""
    ) -> dict[str, Any]:
        row = {
            "kind": "decision",
            "set": name,
            "item_id": item.id,
            "revision": item.revision(),
            "round": self.round_for(name, item),
            "decision": decision,
            "by": by,
            "note": note,
            "ts": now(),
        }
        append_jsonl(self.path, row)
        self.history[(name, item.id)].decisions.append(row)
        return row
