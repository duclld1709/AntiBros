"""M6 packaging: offline fake provider, mini-corpus ingest, smoke config end to end, and
static checks of the Docker / compose setup."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import BaseModel
from qdrant_client import QdrantClient
from typer.testing import CliRunner

from antibros.config import deep_merge, load_app_config, load_model_profile
from antibros.llm import FakeLLMClient, Message
from antibros.llm.fake import schema_stub

MINI = Path("tests/fixtures/mini_corpus/documents.jsonl")


def _schemas() -> list[type[BaseModel]]:
    from antibros.eval.judge import JudgeScores
    from antibros.graph.pipeline import (
        GeneratedAnswer,
        GradeVerdict,
        QueryAnalysis,
        QueryRewrite,
        VerifyResult,
    )
    from antibros.guardrails.guards import GuardVerdict

    return [
        QueryAnalysis,
        QueryRewrite,
        GradeVerdict,
        GeneratedAnswer,
        VerifyResult,
        GuardVerdict,
        JudgeScores,
    ]


@pytest.mark.parametrize("schema", _schemas(), ids=lambda s: s.__name__)
def test_schema_stub_validates_every_structured_output(schema: type[BaseModel]) -> None:
    schema.model_validate(schema_stub(schema.model_json_schema()))


def test_unscripted_fake_client_stubs_but_scripted_one_still_runs_out() -> None:
    from antibros.graph.pipeline import GeneratedAnswer

    msg = [Message(role="user", content="q")]
    stub = FakeLLMClient().complete(msg, GeneratedAnswer)
    assert stub.parsed and stub.parsed["status"] == "answered"
    assert FakeLLMClient().complete(msg).text == "fake"
    scripted = FakeLLMClient(script=["one"])
    scripted.complete(msg)
    with pytest.raises(RuntimeError, match="exhausted"):
        scripted.complete(msg)


def test_ingest_from_jsonl_writes_processed_documents(tmp_path: Path) -> None:
    from antibros.cli import app
    from antibros.ingestion.pipeline import load_documents

    base = load_app_config(Path("configs/smoke.yaml")).model_dump(mode="json")
    base["ingestion"]["processed_dir"] = str(tmp_path / "processed")
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(base), encoding="utf-8")
    res = CliRunner().invoke(app, ["ingest", "--config", str(cfg_path), "--from-jsonl", str(MINI)])
    assert res.exit_code == 0, res.output
    docs = load_documents(tmp_path / "processed")
    assert len(docs) == 3 and all(d.license == "CC BY" for d in docs)


def test_smoke_config_answers_and_redirects_offline(mini_docs: list[Any]) -> None:
    """What the compose smoke test checks over HTTP, in-process with Qdrant :memory:."""
    from antibros.graph.pipeline import RagPipeline
    from antibros.indexing.pipeline import run_index

    cfg = load_app_config(Path("configs/smoke.yaml"))
    cfg = cfg.model_validate(
        deep_merge(cfg.model_dump(mode="json"), {"vector_store": {"mode": "memory"}})
    )
    client = QdrantClient(":memory:")
    from unittest.mock import patch

    with patch("antibros.indexing.pipeline.load_documents", return_value=mini_docs):
        run_index(cfg, client)
    profile = load_model_profile(Path("configs/models/fake.yaml"))
    pipe = RagPipeline(cfg, profile, client=client, documents={d.doc_id: d for d in mini_docs})
    answer = pipe.answer("s1", "How much protein per day builds muscle?", "en")
    assert answer.status == "answered" and answer.citations
    assert [s["node"] for s in answer.steps][-1] == "output_guard" and answer.disclaimer
    red = pipe.answer("s2", "I have kidney disease, can I take creatine?", "en")
    assert red.status == "safety_redirect" and not red.contexts


# --- static checks of the packaging files ---------------------------------------------------


def test_dockerfile_is_multistage_non_root_and_secret_free() -> None:
    text = Path("docker/Dockerfile.api").read_text(encoding="utf-8")
    assert text.count("FROM ") >= 3
    assert "USER antibros" in text and "HEALTHCHECK" in text
    for forbidden in ("API_KEY=", "SECRET", ".env", "COPY data", "huggingface-cli download"):
        assert forbidden not in text
    ignore = Path(".dockerignore").read_text(encoding="utf-8").splitlines()
    assert {".env", "data", ".llm_cache", ".git"} <= set(ignore)


def test_compose_services_and_smoke_override() -> None:
    compose = yaml.safe_load(Path("docker-compose.yml").read_text(encoding="utf-8"))
    services = compose["services"]
    assert set(services) == {"qdrant", "init", "api", "ollama"}
    assert services["ollama"]["profiles"] == ["ollama"]
    api_deps = services["api"]["depends_on"]
    assert api_deps["init"]["condition"] == "service_completed_successfully"
    assert "host.docker.internal:host-gateway" in services["api"]["extra_hosts"]
    smoke = yaml.safe_load(Path("docker-compose.smoke.yml").read_text(encoding="utf-8"))
    smoke_cmd = " ".join(smoke["services"]["init"]["command"])
    assert "configs/smoke.yaml" in smoke_cmd and str(MINI.as_posix()) in smoke_cmd
    assert "configs/models/fake.yaml" in smoke["services"]["api"]["command"]
    for path in ("configs/smoke.yaml", "configs/docker.yaml", "configs/models/fake.yaml"):
        assert Path(path).is_file()


def test_docker_config_serves_full_with_container_friendly_overrides() -> None:
    full = load_app_config(Path("configs/full.yaml"))
    docker = load_app_config(Path("configs/docker.yaml"))
    assert docker.retrieval == full.retrieval and docker.pipeline == full.pipeline
    assert docker.vector_store.mode == "server" and docker.rerank.device == "auto"
    assert docker.retrieval.embedder in docker.index.embedders


def test_env_example_documents_every_setting() -> None:
    from antibros.config import Settings

    env = Path(".env.example").read_text(encoding="utf-8")
    for name in Settings.model_fields:
        assert name.upper() in env, name
    assert "ANTIBROS_MODELS" in env and "ANTIBROS_CONFIG" in env


def test_ci_has_the_spec_jobs() -> None:
    ci = yaml.safe_load(Path(".github/workflows/ci.yml").read_text(encoding="utf-8"))
    assert {"lint", "typecheck", "test", "evalset-verify", "docker-build", "compose-smoke"} <= set(
        ci["jobs"]
    )
    assert json.dumps(ci["jobs"]["compose-smoke"]).count("fresh_clone_test.py") == 1
