from __future__ import annotations

from typer.testing import CliRunner

from antibros.cli import app

runner = CliRunner()


def test_help_lists_all_subcommands() -> None:
    out = runner.invoke(app, ["--help"]).output
    for cmd in ["ingest", "index", "stats", "evalset", "eval", "compare", "serve"]:
        assert cmd in out
    evalset = runner.invoke(app, ["evalset", "--help"]).output
    for cmd in ["build", "check", "verify-models", "freeze", "verify", "leakcheck"]:
        assert cmd in evalset
    ev = runner.invoke(app, ["eval", "--help"]).output
    for cmd in ["retrieval", "safety", "predict", "judge", "report", "final"]:
        assert cmd in ev


def test_predict_refuses_test_split() -> None:
    res = runner.invoke(
        app,
        [
            "eval",
            "predict",
            "--config",
            "c.yaml",
            "--models",
            "m.yaml",
            "--tag",
            "x",
            "--split",
            "test",
        ],
    )
    assert res.exit_code == 1
    assert "eval final" in res.output


def test_predict_rejects_bad_tag() -> None:
    res = runner.invoke(
        app, ["eval", "predict", "--config", "c.yaml", "--models", "m.yaml", "--tag", "a:b"]
    )
    assert res.exit_code == 1


def test_stub_exit_code() -> None:
    assert runner.invoke(app, ["eval", "final"]).exit_code == 2
