"""``antibros`` command-line interface. Later-milestone subcommands are stubs until they land."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated, NoReturn, cast

import typer

from antibros.config import CHUNKER_NAMES, AppConfig, ChunkerName, load_app_config
from antibros.io import is_valid_run_tag, write_json
from antibros.logging import configure_logging

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


DEFAULT_CONFIG = Path("configs/base.yaml")


def _load(config: Path) -> AppConfig:
    cfg = load_app_config(config)
    configure_logging(cfg.log_level)
    return cfg


@app.command()
def manifest(
    config: ConfigOpt = DEFAULT_CONFIG,
    candidates: Annotated[Path | None, typer.Option(help="Candidate list YAML.")] = None,
    out: Annotated[Path | None, typer.Option(help="Manifest JSONL to write.")] = None,
) -> None:
    """Verify corpus candidates via Europe PMC (OA, licence, year) and write the manifest."""
    from antibros.ingestion.europepmc import EuropePMCClient
    from antibros.ingestion.manifest import build_manifest, load_candidates, write_manifest

    cfg = _load(config)
    cands = load_candidates(candidates or cfg.ingestion.candidates)
    with EuropePMCClient(cfg.ingestion.europepmc) as client:
        entries, verification = build_manifest(client, cands)
    write_json(
        cfg.reports_dir / "corpus_verification.json",
        [v.model_dump(mode="json", exclude={"entry"}) for v in verification],
    )
    counts = {t: sum(e.topic == t for e in entries) for t in cands.target}
    if counts != cands.target:
        typer.echo(f"Topic counts {counts} differ from target {cands.target}.", err=True)
        raise typer.Exit(code=1)
    path = out or cfg.ingestion.manifest
    write_manifest(path, entries)
    typer.echo(f"Wrote {len(entries)} entries to {path}")


@app.command()
def ingest(
    manifest: Annotated[Path | None, typer.Option(help="Corpus manifest JSONL.")] = None,
    config: ConfigOpt = DEFAULT_CONFIG,
    offline: Annotated[bool, typer.Option(help="Use cached data/raw only.")] = False,
) -> None:
    """Download and parse the corpus listed in the manifest (verifies sha256)."""
    from antibros.ingestion.europepmc import EuropePMCClient
    from antibros.ingestion.pipeline import IngestError
    from antibros.ingestion.pipeline import ingest as run_ingest

    cfg = _load(config)
    client = None if offline else EuropePMCClient(cfg.ingestion.europepmc)
    try:
        docs = run_ingest(
            manifest or cfg.ingestion.manifest,
            cfg.ingestion.raw_dir,
            cfg.ingestion.processed_dir,
            client,
        )
    except IngestError as exc:
        typer.echo(f"Ingest failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    finally:
        if client is not None:
            client.close()
    typer.echo(f"Ingested {len(docs)} documents into {cfg.ingestion.processed_dir}")


@app.command()
def index(
    config: ConfigOpt = DEFAULT_CONFIG,
    chunker: Annotated[list[str] | None, typer.Option(help="Only these chunkers.")] = None,
    embedder: Annotated[list[str] | None, typer.Option(help="Only these embedders.")] = None,
) -> None:
    """Chunk, embed and index the corpus into Qdrant (one collection per chunker x embedder)."""
    from antibros.indexing.pipeline import run_index
    from antibros.indexing.qdrant_store import make_client

    cfg = _load(config)
    unknown = set(chunker or []) - set(CHUNKER_NAMES)
    if unknown:
        typer.echo(f"Unknown chunker(s): {sorted(unknown)}", err=True)
        raise typer.Exit(code=1)
    client = make_client(cfg.vector_store)
    try:
        results = run_index(cfg, client, cast(list[ChunkerName], chunker) or None, embedder or None)
    finally:
        client.close()
    for name, r in results.items():
        typer.echo(f"{name}: {int(r['points'])} points")


@app.command()
def stats(config: ConfigOpt = DEFAULT_CONFIG) -> None:
    """Write corpus statistics to reports/corpus_stats.md."""
    from antibros.stats import write_stats

    typer.echo(f"Wrote {write_stats(_load(config))}")


@app.command()
def search(
    query: str,
    config: ConfigOpt = DEFAULT_CONFIG,
    k: Annotated[int, typer.Option(help="Number of results.")] = 5,
) -> None:
    """Ad-hoc retrieval with the config's `retrieval:` settings (for manual checks)."""
    from antibros.embedding.base import make_embedder
    from antibros.indexing.qdrant_store import collection_name, make_client
    from antibros.retrieval.base import make_retriever

    cfg = _load(config)
    r = cfg.retrieval
    client = make_client(cfg.vector_store)
    embedder = make_embedder(r.embedder, cfg.embedders[r.embedder])
    coll = collection_name(cfg.vector_store.collection_prefix, r.chunker, r.embedder)
    try:
        hits = make_retriever(r, client, coll, embedder).retrieve(query, k)
    finally:
        embedder.close()
        client.close()
    for i, h in enumerate(hits, start=1):
        snippet = str(h.payload.get("text", ""))[:160].replace("\n", " ")
        typer.echo(f"{i}. {h.score:.4f} {h.chunk_id}\n   {snippet}")


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
