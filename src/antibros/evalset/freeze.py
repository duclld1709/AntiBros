"""``evalset freeze`` / ``evalset verify``: stratified splits, VERSION and byte checksums.

Checksums are over raw bytes of every file under the evalset directory, so any edit,
including an LF -> CRLF conversion, makes ``verify`` fail (``.gitattributes`` keeps Git from
converting line endings under ``evalsets/``).
"""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from antibros.evalset.checks import dev_quota
from antibros.evalset.schemas import AnyItem, SetName
from antibros.ingestion.schemas import CLEANING_VERSION
from antibros.io import read_text, write_text

CHECKSUMS = "CHECKSUMS.sha256"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _largest_remainder(
    sizes: dict[tuple[str, ...], int], quota: int, total: int, rng: random.Random
) -> dict[tuple[str, ...], int]:
    """Split ``quota`` over strata proportionally to their sizes (Hamilton method, seeded ties)."""
    if total == 0:
        return dict.fromkeys(sizes, 0)
    exact = {k: n * quota / total for k, n in sizes.items()}
    alloc = {k: int(v) for k, v in exact.items()}
    keys = sorted(sizes)
    rng.shuffle(keys)
    keys.sort(key=lambda k: exact[k] - alloc[k], reverse=True)  # stable: shuffled ties
    for k in keys[: quota - sum(alloc.values())]:
        alloc[k] += 1
    return alloc


def assign_splits(
    name: SetName, items: list[AnyItem], dev_total: int, total: int, seed: int
) -> list[AnyItem]:
    """Stratified dev/test split: exact dev quota per category, spread over (language, topic)."""
    if name == "synth":
        return [it.model_copy(update={"split": "dev"}) for it in items]
    rng = random.Random(seed)
    by_cat: dict[str, list[AnyItem]] = defaultdict(list)
    for it in sorted(items, key=lambda i: i.id):
        by_cat[getattr(it, "category", "")].append(it)
    dev_ids: set[str] = set()
    for cat in sorted(by_cat):
        group = by_cat[cat]
        strata: dict[tuple[str, ...], list[AnyItem]] = defaultdict(list)
        for it in group:
            strata[(it.language, it.topic)].append(it)
        quota = dev_quota(len(group), dev_total, total)
        alloc = _largest_remainder({k: len(v) for k, v in strata.items()}, quota, len(group), rng)
        for key in sorted(strata):
            dev_ids.update(it.id for it in rng.sample(strata[key], alloc[key]))
    return [it.model_copy(update={"split": "dev" if it.id in dev_ids else "test"}) for it in items]


def list_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.name != CHECKSUMS)


def write_checksums(root: Path) -> str:
    lines = [f"{sha256_file(p)}  {p.relative_to(root).as_posix()}\n" for p in list_files(root)]
    write_text(root / CHECKSUMS, "".join(lines))
    return sha256_file(root / CHECKSUMS)


def write_version(root: Path, version: str, manifest: Path, counts: dict[str, int]) -> None:
    fields = {
        "version": version,
        "frozen_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "cleaning_version": CLEANING_VERSION,
        "corpus_manifest_sha256": sha256_file(manifest),
        **{f"items_{k}": str(v) for k, v in sorted(counts.items())},
    }
    write_text(root / "VERSION", "".join(f"{k}={v}\n" for k, v in fields.items()))


def read_version(root: Path) -> dict[str, str]:
    path = root / "VERSION"
    if not path.exists():
        return {}
    pairs = (line.split("=", 1) for line in read_text(path).splitlines() if "=" in line)
    return {k: v for k, v in pairs}


@dataclass
class VerifyResult:
    ok: bool
    evalset_sha256: str | None = None
    problems: list[str] = field(default_factory=list)


def verify_checksums(root: Path) -> VerifyResult:
    path = root / CHECKSUMS
    if not path.exists():
        return VerifyResult(False, None, [f"missing {path}"])
    expected: dict[str, str] = {}
    for line in path.read_bytes().decode("utf-8").split("\n"):
        if line:
            digest, _, rel = line.partition("  ")
            expected[rel] = digest
    actual = {p.relative_to(root).as_posix(): sha256_file(p) for p in list_files(root)}
    problems = [
        f"modified: {rel}"
        for rel in sorted(expected)
        if rel in actual and actual[rel] != expected[rel]
    ]
    problems += [f"missing: {rel}" for rel in sorted(set(expected) - set(actual))]
    problems += [f"unexpected: {rel}" for rel in sorted(set(actual) - set(expected))]
    return VerifyResult(not problems, sha256_file(path), problems)
