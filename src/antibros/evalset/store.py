"""Evalset file layout and loading.

``<dir>/work/drafts/*.jsonl`` hold every drafted item (no split) while the set is being built;
``<dir>/*.jsonl`` hold the kept items with splits, written by ``evalset freeze``. Pooling and
verification artefacts live next to the drafts in ``work/`` and are frozen with the set.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from pydantic import ValidationError

from antibros.evalset.schemas import FILE_NAMES, ITEM_TYPES, AnyItem, SetName
from antibros.ingestion.manifest import read_manifest
from antibros.ingestion.pipeline import load_documents
from antibros.io import read_jsonl, write_jsonl

SET_NAMES: tuple[SetName, ...] = ("golden", "safety", "synth")
T = TypeVar("T")


@dataclass(frozen=True)
class Layout:
    root: Path

    @property
    def work(self) -> Path:
        return self.root / "work"

    def draft(self, name: SetName) -> Path:
        return self.work / "drafts" / FILE_NAMES[name]

    def final(self, name: SetName) -> Path:
        return self.root / FILE_NAMES[name]

    @property
    def pool(self) -> Path:
        return self.work / "pool.jsonl"

    @property
    def pool_grades(self) -> Path:
        return self.work / "pool_grades.jsonl"

    @property
    def unanswerable_top(self) -> Path:
        return self.work / "unanswerable_top.jsonl"

    @property
    def review_log(self) -> Path:
        return self.root / "review_log.jsonl"

    @property
    def manifest(self) -> Path:
        return self.root / "corpus_manifest.jsonl"

    @property
    def checksums(self) -> Path:
        return self.root / "CHECKSUMS.sha256"

    @property
    def version(self) -> Path:
        return self.root / "VERSION"


@dataclass
class LoadError:
    path: Path
    line: int
    message: str


def load_items(path: Path, name: SetName) -> tuple[list[AnyItem], list[LoadError]]:
    """Parse every line; schema errors are collected instead of raised."""
    model = ITEM_TYPES[name]
    items: list[AnyItem] = []
    errors: list[LoadError] = []
    for i, row in enumerate(read_jsonl(path), start=1):
        try:
            items.append(model.model_validate(row))
        except ValidationError as e:
            msg = "; ".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors())
            errors.append(LoadError(path, i, f"{row.get('id', '?')}: {msg}"))
    return items, errors


def load_items_strict(path: Path, name: SetName) -> list[AnyItem]:
    items, errors = load_items(path, name)
    if errors:
        raise ValueError(f"{path}: {len(errors)} invalid item(s), first: {errors[0].message}")
    return items


def save_items(path: Path, items: list[AnyItem]) -> None:
    write_jsonl(path, [i.model_dump(mode="json") for i in items])


def doc_texts(processed_dir: Path) -> dict[str, str]:
    return {d.doc_id: d.clean_text for d in load_documents(processed_dir)}


def manifest_ids(path: Path) -> set[str]:
    return {e.doc_id for e in read_manifest(path)}


def rows(path: Path) -> list[dict[str, Any]]:
    return [r for r in read_jsonl(path) if isinstance(r, dict)]
