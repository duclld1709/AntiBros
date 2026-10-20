"""``antibros`` command-line interface. Subcommands are stubs until their milestone lands."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated, NoReturn

import typer

from antibros.io import is_valid_run_tag

app = typer.Typer(help="AntiBros: evidence-based protein & creatine Q&A.", no_args_is_help=True)
evalset_app = typer.Typer(help="Build, verify and freeze evaluation sets.", no_args_is_help=True)
eval_app = typer.Typer(help="Run evaluations (dev split only; test via `eval final`).")
app.add_typer(evalset_app, name="evalset")
app.add_typer(eval_app, name="eval", no_args_is_help=True)


class Split(StrEnum):
    dev = "dev"
    test = "test"


class SetName(StrEnum):
    golden = "golden"
    synth = "synth"


def _todo(command: str, milestone: str) -> NoReturn:
    typer.echo(f"`antibros {command}` is not implemented yet (planned for {milestone}).", err=True)
    raise typer.Exit(code=2)


def _dev_only(split: Split) -> None:
    if split is Split.test:
        typer.echo(
            "Refusing to run on the test split. The test split only runs via "
            "`antibros eval final --plan configs/final_eval_plan.yaml` (spec §6.5).",
            err=True,
        )
        raise typer.Exit(code=1)


ConfigOpt = Annotated[Path, typer.Option("--config", help="Pipeline config YAML.")]
ModelsOpt = Annotated[Path, typer.Option("--models", help="Model profile YAML.")]
RunArg = Annotated[Path, typer.Option("--run", help="Experiment run directory.")]


@app.command()
def ingest(
    manifest: Annotated[Path, typer.Option(help="Corpus manifest JSONL.")] = Path(
        "evalsets/v1/corpus_manifest.jsonl"
    ),
) -> None:
    """Download and parse the corpus listed in the manifest (verifies sha256)."""
    _todo("ingest", "M1")


@app.command()
def index(config: ConfigOpt) -> None:
    """Chunk, embed and index the corpus into Qdrant."""
    _todo("index", "M1")


@app.command()
def stats() -> None:
    """Write corpus statistics to reports/corpus_stats.md."""
    _todo("stats", "M1")


@evalset_app.command("build")
def evalset_build() -> None:
    """Draft evalset items from the chunked corpus."""
    _todo("evalset build", "M2")


@evalset_app.command("check")
def evalset_check() -> None:
    """Deterministic checks: quotes exist, schema, duplicates, lexical overlap, splits."""
    _todo("evalset check", "M2")


@evalset_app.command("verify-models")
def evalset_verify_models(
    models: ModelsOpt = Path("configs/models/evalset_verifiers.yaml"),
) -> None:
    """Have two cross-family verifier models grade every item."""
    _todo("evalset verify-models", "M2")


@evalset_app.command("freeze")
def evalset_freeze() -> None:
    """Write CHECKSUMS.sha256 and VERSION."""
    _todo("evalset freeze", "M2")


@evalset_app.command("verify")
def evalset_verify() -> None:
    """Verify frozen evalset checksums."""
    _todo("evalset verify", "M2")


@evalset_app.command("leakcheck")
def evalset_leakcheck() -> None:
    """Fail if src/, configs/ or prompts/ share >=8-word n-grams with the evalset."""
    _todo("evalset leakcheck", "M2")


@eval_app.command("retrieval")
def eval_retrieval(
    config: ConfigOpt,
    set_: Annotated[SetName, typer.Option("--set")] = SetName.golden,
) -> None:
    """Retrieval metrics (no LLM)."""
    _todo("eval retrieval", "M2")


@eval_app.command("safety")
def eval_safety(config: ConfigOpt, split: Split = Split.dev) -> None:
    """Safety guard metrics."""
    _dev_only(split)
    _todo("eval safety", "M4")


@eval_app.command("predict")
def eval_predict(
    config: ConfigOpt,
    models: ModelsOpt,
    tag: Annotated[str, typer.Option(help="Run tag, [A-Za-z0-9_-] only.")],
    split: Split = Split.dev,
    limit: Annotated[int | None, typer.Option(help="Only the first N questions.")] = None,
) -> None:
    """Phase 1: run the pipeline LLM and store predictions."""
    _dev_only(split)
    if not is_valid_run_tag(tag):
        typer.echo(f"Invalid tag {tag!r}: use only [A-Za-z0-9_-].", err=True)
        raise typer.Exit(code=1)
    _todo("eval predict", "M2")


@eval_app.command("judge")
def eval_judge(run: RunArg, models: ModelsOpt) -> None:
    """Phase 2: score stored predictions with the judge LLM."""
    _todo("eval judge", "M2")


@eval_app.command("report")
def eval_report(run: RunArg) -> None:
    """Render metrics with bootstrap confidence intervals."""
    _todo("eval report", "M2")


@eval_app.command("final")
def eval_final(
    plan: Annotated[Path, typer.Option(help="Pre-registered plan.")] = Path(
        "configs/final_eval_plan.yaml"
    ),
) -> None:
    """The only way to run the test split (M5, pre-registered plan)."""
    _todo("eval final", "M5")


@app.command()
def compare(run_a: Path, run_b: Path) -> None:
    """Paired comparison of two runs (refuses on hard fingerprint mismatch)."""
    _todo("compare", "M2")


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Start the FastAPI server."""
    _todo("serve", "M3")


if __name__ == "__main__":
    app()
