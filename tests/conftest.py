from __future__ import annotations

from pathlib import Path

import pytest

from antibros.ingestion.schemas import Document
from antibros.io import read_text

FIXTURES = Path(__file__).resolve().parent / "fixtures"
MINI_CORPUS = FIXTURES / "mini_corpus"


def load_mini_docs() -> list[Document]:
    lines = read_text(MINI_CORPUS / "documents.jsonl").splitlines()
    return [Document.model_validate_json(line) for line in lines if line]


@pytest.fixture(scope="session")
def mini_docs() -> list[Document]:
    return load_mini_docs()
