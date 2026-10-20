from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from antibros.config import Settings, load_app_config, load_model_profile

ROOT = Path(__file__).resolve().parents[2]


def test_base_config_loads() -> None:
    cfg = load_app_config(ROOT / "configs" / "base.yaml")
    assert cfg.vector_store.mode in {"server", "local", "memory"}


@pytest.mark.parametrize("name", ["laptop", "laptop_light", "evalset_verifiers"])
def test_model_profiles_load(name: str) -> None:
    profile = load_model_profile(ROOT / "configs" / "models" / f"{name}.yaml")
    specs = [s for s in (profile.pipeline, profile.judge) if s] + profile.verifiers
    assert specs
    assert all(s.digest and not s.think and s.temperature == 0.0 for s in specs)
    if profile.pipeline and profile.judge:
        assert profile.pipeline.family != profile.judge.family
    if profile.verifiers:
        assert len({v.family for v in profile.verifiers}) == len(profile.verifiers)


def test_unknown_keys_rejected(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("name: x\ntypo_key: 1\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_app_config(p)


def test_settings_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)  # no .env here
    monkeypatch.setenv("LLM_BASE_URL", "http://host.docker.internal:11434/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    s = Settings()
    assert s.llm_base_url == "http://host.docker.internal:11434/v1"
    assert s.openai_api_key is not None
    assert "sk-test" not in repr(s)
