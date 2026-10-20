"""``antibros`` command-line interface."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, cast

import typer

from antibros.config import CHUNKER_NAMES, AppConfig, ChunkerName, load_app_config

if TYPE_CHECKING:
    from antibros.config import ModelProfile as ModelProfileT
    from antibros.evalset.schemas import AnyItem as AnyItemT
    from antibros.evalset.schemas import SetName as SetNameLit
from antibros.io import is_valid_run_tag, write_json
from antibros.logging import configure_logging

app = typer.Typer(help="AntiBros: evidence-based protein & creatine Q&A.", no_args_is_help=True)
evalset_app = typer.Typer(help="Build, verify and freeze evaluation sets.", no_args_is_help=True)
eval_app = typer.Typer(help="Run evaluations (dev split only; test via `eval final`).")
app.add_typer(evalset_app, name="evalset")
plan_app = typer.Typer(help="Pre-registered final evaluation plan.", no_args_is_help=True)
app.add_typer(eval_app, name="eval", no_args_is_help=True)
app.add_typer(plan_app, name="plan")


class Split(StrEnum):
    dev = "dev"
    test = "test"


class SetName(StrEnum):
    golden = "golden"
    synth = "synth"


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
LAPTOP = Path("configs/models/laptop.yaml")


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
    from_jsonl: Annotated[
        Path | None,
        typer.Option(help="Load already parsed documents (e.g. the CI mini corpus) instead."),
    ] = None,
) -> None:
    """Download and parse the corpus listed in the manifest (verifies sha256)."""
    from antibros.ingestion.europepmc import EuropePMCClient
    from antibros.ingestion.pipeline import IngestError, docs_dir
    from antibros.ingestion.pipeline import ingest as run_ingest

    cfg = _load(config)
    if from_jsonl is not None:
        from antibros.ingestion.schemas import Document
        from antibros.io import read_text

        docs = [
            Document.model_validate_json(line)
            for line in read_text(from_jsonl).splitlines()
            if line
        ]
        for doc in docs:
            write_json(
                docs_dir(cfg.ingestion.processed_dir) / f"{doc.doc_id}.json",
                doc.model_dump(mode="json"),
            )
        typer.echo(
            f"Loaded {len(docs)} documents from {from_jsonl} into {cfg.ingestion.processed_dir}"
        )
        return
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


class Stage(StrEnum):
    draft = "draft"
    final = "final"


class DecisionOpt(StrEnum):
    revise = "revise"
    drop = "drop"


VERIFIERS = Path("configs/models/evalset_verifiers.yaml")


def _verifier_ids(models: Path) -> list[str]:
    from antibros.config import load_model_profile
    from antibros.evalset.verify_models import verifier_id

    return [verifier_id(s) for s in load_model_profile(models).verifiers]


def _evalset_items(cfg: AppConfig, stage: Stage) -> list[tuple[SetNameLit, AnyItemT]]:
    from antibros.evalset.store import SET_NAMES, Layout, load_items_strict

    layout = Layout(cfg.evalset.dir)
    out: list[tuple[SetNameLit, AnyItemT]] = []
    for name in SET_NAMES:
        path = layout.draft(name) if stage is Stage.draft else layout.final(name)
        out += [(name, it) for it in load_items_strict(path, name)]
    return out


@evalset_app.command("build")
def evalset_build(
    config: ConfigOpt = DEFAULT_CONFIG,
    merge: Annotated[bool, typer.Option(help="Merge work/pool_grades.jsonl into drafts.")] = False,
) -> None:
    """Pool evidence candidates (BM25 + bge-m3 + e5) and retrieve the unanswerable top-k."""
    from antibros.embedding.base import make_embedder
    from antibros.evalset.build import BM25, merge_pool_grades, pool
    from antibros.evalset.schemas import PoolGrade
    from antibros.evalset.store import Layout, doc_texts, load_items_strict, rows, save_items
    from antibros.indexing.pipeline import chunks_path, read_chunks
    from antibros.indexing.qdrant_store import collection_name, make_client
    from antibros.io import write_jsonl
    from antibros.retrieval.base import DenseRetriever, make_retriever

    cfg = _load(config)
    es = cfg.evalset
    layout = Layout(es.dir)
    texts = doc_texts(cfg.ingestion.processed_dir)
    chunks = read_chunks(chunks_path(cfg.ingestion.processed_dir, es.pool_chunker))
    if merge:
        grades = [PoolGrade.model_validate(r) for r in rows(layout.pool_grades)]
        total = 0
        for set_name in ("golden", "safety"):
            path = layout.draft(set_name)
            merged, added = merge_pool_grades(
                load_items_strict(path, set_name), grades, chunks, texts
            )
            save_items(path, merged)
            total += added
        typer.echo(f"Merged {total} pooled evidence quotes into drafts")
        return

    items = _evalset_items(cfg, Stage.draft)
    client = make_client(cfg.vector_store)
    prefix = cfg.vector_store.collection_prefix

    def dense_ranker(emb_name: str) -> Callable[[list[str], int], list[list[str]]]:
        def rank(questions: list[str], k: int) -> list[list[str]]:
            embedder = make_embedder(emb_name, cfg.embedders[emb_name])
            try:
                coll = collection_name(prefix, es.pool_chunker, emb_name)
                r = DenseRetriever(client, coll, embedder)
                return [[h.chunk_id for h in r.retrieve(q, k)] for q in questions]
            finally:
                embedder.close()

        return rank

    rankers = {"bm25": BM25(chunks).rank, **{e: dense_ranker(e) for e in es.pool_embedders}}
    try:
        pool_rows = pool(items, chunks, rankers, texts, es.pool_k, cfg.eval.span_min_coverage)
        write_jsonl(layout.pool, pool_rows)
        # Unanswerable items: the pipeline's own retrieval config, top-k passages for verifiers.
        r = cfg.retrieval
        embedder = make_embedder(r.embedder, cfg.embedders[r.embedder])
        try:
            retriever = make_retriever(
                r, client, collection_name(prefix, r.chunker, r.embedder), embedder
            )
            top = []
            for name, it in items:
                if getattr(it, "category", None) == "unanswerable":
                    hits = retriever.retrieve(it.question, es.unanswerable_top_k)
                    top += [
                        {
                            "set": name,
                            "item_id": it.id,
                            "rank": i,
                            "chunk_id": h.chunk_id,
                            "text": h.payload["text"],
                        }
                        for i, h in enumerate(hits, start=1)
                    ]
            write_jsonl(layout.unanswerable_top, top)
        finally:
            embedder.close()
    finally:
        client.close()
    todo = sum(1 for row in pool_rows if row["covered_grade"] == 0)
    typer.echo(f"Wrote {len(pool_rows)} pooled chunks ({todo} to grade) to {layout.pool}")


@evalset_app.command("check")
def evalset_check(
    config: ConfigOpt = DEFAULT_CONFIG,
    stage: Annotated[Stage, typer.Option(help="Check drafts or frozen files.")] = Stage.draft,
    embed: Annotated[bool, typer.Option(help="Embedding near-duplicate check (GPU).")] = True,
    require_verified: Annotated[bool, typer.Option(help="Require verifier verdicts.")] = False,
    models: ModelsOpt = VERIFIERS,
) -> None:
    """Deterministic checks: quotes exist, schema, duplicates, lexical overlap, counts, splits."""
    from antibros.embedding.base import make_embedder
    from antibros.evalset.checks import run_checks
    from antibros.evalset.review import ReviewLog
    from antibros.evalset.store import Layout, doc_texts, manifest_ids

    cfg = _load(config)
    es = cfg.evalset
    layout = Layout(es.dir)
    embed_fn = None
    if embed:
        embedder = make_embedder(es.dedup_embedder, cfg.embedders[es.dedup_embedder])

        def embed_fn(texts: list[str]) -> list[list[float]]:
            try:
                return embedder.embed_queries(texts).dense
            finally:
                embedder.close()

    report = run_checks(
        layout,
        es,
        doc_texts(cfg.ingestion.processed_dir),
        manifest_ids(layout.manifest),
        stage="final" if stage is Stage.final else "draft",
        embed=embed_fn,
        review=ReviewLog(layout.review_log),
        verifiers=_verifier_ids(models),
        require_verified=require_verified,
    )
    typer.echo(f"Checked {report.n_items} items; counts: {report.counts}")
    typer.echo(f"Lexical overlap (question vs evidence): {report.overlap}")
    for issue in report.issues:
        typer.echo(f"FAIL [{issue.check}] {issue.set}:{issue.item_id or '-'} {issue.message}")
    if not report.ok:
        typer.echo(f"{len(report.issues)} issue(s)", err=True)
        raise typer.Exit(code=1)
    typer.echo("All checks passed")


@evalset_app.command("verify-models")
def evalset_verify_models(
    config: ConfigOpt = DEFAULT_CONFIG,
    models: ModelsOpt = VERIFIERS,
    limit: Annotated[int | None, typer.Option(help="At most N items per verifier.")] = None,
    kind: Annotated[
        list[str] | None,
        typer.Option(help="Only these item kinds: supported, unanswerable, red_flag, synth."),
    ] = None,
) -> None:
    """Have two cross-family verifier models grade every item (resumable)."""
    from antibros.config import Settings, load_model_profile
    from antibros.evalset.review import ReviewLog
    from antibros.evalset.store import Layout, rows
    from antibros.evalset.verify_models import PROMPTS, verify
    from antibros.llm import build_client
    from antibros.prompts import load_prompt

    cfg = _load(config)
    layout = Layout(cfg.evalset.dir)
    profile = load_model_profile(models)
    if len({s.family for s in profile.verifiers}) < 2:
        typer.echo("evalset verifiers must come from two different model families", err=True)
        raise typer.Exit(code=1)
    top: dict[str, list[str]] = {}
    for row in sorted(rows(layout.unanswerable_top), key=lambda r: (r["item_id"], r["rank"])):
        top.setdefault(row["item_id"], []).append(row["text"])
    review = ReviewLog(layout.review_log)
    settings = Settings()
    stats = verify(
        _evalset_items(cfg, Stage.draft),
        review,
        profile.verifiers,
        lambda spec: build_client(
            spec, settings, cfg.llm_cache.dir if cfg.llm_cache.enabled else None
        ),
        {k: load_prompt(cfg.prompts_dir, v) for k, v in PROMPTS.items()},
        top,
        cfg.evalset,
        limit=limit,
        kinds=set(kind) if kind else None,
    )
    typer.echo(
        f"Judged {stats.judged} (accepted {stats.accepted}); auto-kept {stats.auto_kept}; "
        f"invalid outputs {len(stats.failed)}; over max rounds {len(stats.skipped_max_rounds)}"
    )
    if stats.failed:
        raise typer.Exit(code=1)


@evalset_app.command("decide")
def evalset_decide(
    item_id: Annotated[str, typer.Option("--id", help="Item ID.")],
    decision: Annotated[DecisionOpt, typer.Option(help="revise (before editing) or drop.")],
    note: Annotated[str, typer.Option(help="Why.")],
    config: ConfigOpt = DEFAULT_CONFIG,
) -> None:
    """Log Claude Code's decision for an item the verifiers disagreed on."""
    from antibros.evalset.review import ReviewLog
    from antibros.evalset.store import Layout

    cfg = _load(config)
    for name, it in _evalset_items(cfg, Stage.draft):
        if it.id == item_id:
            ReviewLog(Layout(cfg.evalset.dir).review_log).add_decision(
                name, it, decision.value, by="claude-code", note=note
            )
            typer.echo(f"Logged {decision.value} for {name}:{item_id}")
            return
    typer.echo(f"No item {item_id!r}", err=True)
    raise typer.Exit(code=1)


@evalset_app.command("freeze")
def evalset_freeze(config: ConfigOpt = DEFAULT_CONFIG, models: ModelsOpt = VERIFIERS) -> None:
    """Split kept items (stratified, seeded), write final files, VERSION and CHECKSUMS.sha256."""
    from antibros.evalset.checks import run_checks, target_dev
    from antibros.evalset.freeze import assign_splits, write_checksums, write_version
    from antibros.evalset.review import ReviewLog
    from antibros.evalset.store import SET_NAMES, Layout, doc_texts, manifest_ids, save_items

    cfg = _load(config)
    es = cfg.evalset
    layout = Layout(es.dir)
    if layout.checksums.exists():
        typer.echo(f"{layout.root} is already frozen; changes require a new version.", err=True)
        raise typer.Exit(code=1)
    review = ReviewLog(layout.review_log)
    ids = _verifier_ids(models)
    texts, mids = doc_texts(cfg.ingestion.processed_dir), manifest_ids(layout.manifest)
    pre = run_checks(layout, es, texts, mids, review=review, verifiers=ids, require_verified=True)
    if not pre.ok:
        for issue in pre.issues:
            typer.echo(f"FAIL [{issue.check}] {issue.set}:{issue.item_id or '-'} {issue.message}")
        raise typer.Exit(code=1)
    counts: dict[str, int] = {}
    by_set: dict[SetNameLit, list[AnyItemT]] = {n: [] for n in SET_NAMES}
    for name, it in _evalset_items(cfg, Stage.draft):
        if review.status(name, it, ids) == "kept":
            by_set[name].append(it)
    for name in SET_NAMES:
        dev_total, total = target_dev(es, name)
        kept = [
            it.model_copy(
                update={"provenance": it.provenance.model_copy(update={"verified_by": ids})}
            )
            for it in by_set[name]
        ]
        save_items(layout.final(name), assign_splits(name, kept, dev_total, total, es.split_seed))
        counts[name] = len(kept)
    post = run_checks(
        layout, es, texts, mids, stage="final", review=review, verifiers=ids, require_verified=True
    )
    if not post.ok:
        for issue in post.issues:
            typer.echo(f"FAIL [{issue.check}] {issue.set}:{issue.item_id or '-'} {issue.message}")
        raise typer.Exit(code=1)
    write_version(layout.root, es.version, layout.manifest, counts)
    digest = write_checksums(layout.root)
    typer.echo(f"Frozen {layout.root}: {counts}; evalset sha256 {digest}")


@evalset_app.command("verify")
def evalset_verify(config: ConfigOpt = DEFAULT_CONFIG) -> None:
    """Verify frozen evalset checksums (byte-exact, so CRLF conversion fails)."""
    from antibros.evalset.freeze import verify_checksums

    cfg = load_app_config(config)
    result = verify_checksums(cfg.evalset.dir)
    for p in result.problems:
        typer.echo(f"FAIL {p}", err=True)
    if not result.ok:
        raise typer.Exit(code=1)
    typer.echo(f"OK {cfg.evalset.dir} sha256 {result.evalset_sha256}")


@evalset_app.command("leakcheck")
def evalset_leakcheck(config: ConfigOpt = DEFAULT_CONFIG) -> None:
    """Fail if src/, configs/ or prompts/ share >=8-word n-grams with the evalset."""
    from antibros.evalset.leakcheck import leakcheck
    from antibros.evalset.store import Layout

    cfg = load_app_config(config)
    es = cfg.evalset
    stage = Stage.final if Layout(es.dir).checksums.exists() else Stage.draft
    items = [it for _, it in _evalset_items(cfg, stage)]
    leaks = leakcheck(items, es.leakcheck_paths, es.leakcheck_ngram)
    for leak in leaks:
        typer.echo(f"LEAK {leak.path}:{leak.line} [{leak.item_id}] {leak.ngram!r}", err=True)
    if leaks:
        raise typer.Exit(code=1)
    typer.echo(
        f"No leaks: {len(items)} {stage.value} items vs {[str(p) for p in es.leakcheck_paths]}"
    )


def _frozen_items(cfg: AppConfig, name: SetNameLit) -> list[AnyItemT]:
    from antibros.evalset.store import Layout, load_items_strict

    layout = Layout(cfg.evalset.dir)
    if not layout.final(name).exists():
        typer.echo(f"{layout.final(name)} not found: freeze the evalset first.", err=True)
        raise typer.Exit(code=1)
    return load_items_strict(layout.final(name), name)


def _start_run(
    cfg: AppConfig, tag: str, meta: Mapping[str, Any], snapshot: Mapping[str, Any]
) -> Path:
    from antibros.eval.runs import new_run_dir, write_config_snapshot

    run_dir = new_run_dir(cfg.eval.runs_dir, tag)
    write_config_snapshot(run_dir, snapshot)
    write_json(run_dir / "run.json", meta)
    return run_dir


@eval_app.command("retrieval")
def eval_retrieval(
    config: ConfigOpt,
    set_: Annotated[SetName, typer.Option("--set")] = SetName.golden,
    tag: Annotated[str | None, typer.Option(help="Run tag (default: retrieval_<set>).")] = None,
) -> None:
    """Retrieval metrics on the dev split, no LLM: Evidence Recall@k, MRR, nDCG, Doc Recall."""
    from antibros.embedding.base import make_embedder
    from antibros.eval.fingerprint import build_fingerprint
    from antibros.eval.retrieval_eval import chunk_ref, queries_for, run_retrieval_eval
    from antibros.eval.runs import timed_phase
    from antibros.evalset.store import doc_texts
    from antibros.indexing.pipeline import chunks_path, read_chunks
    from antibros.indexing.qdrant_store import collection_name, make_client
    from antibros.io import write_jsonl
    from antibros.reranking.base import make_reranker
    from antibros.retrieval.base import make_retriever

    cfg = _load(config)
    name: SetNameLit = "golden" if set_ is SetName.golden else "synth"
    queries = queries_for(_frozen_items(cfg, name), "dev")
    r = cfg.retrieval
    snapshot = {"app": cfg.model_dump(mode="json"), "config_path": config.as_posix()}
    meta = {"kind": "retrieval", "eval_set": name, "split": "dev"}
    run_dir = _start_run(cfg, tag or f"retrieval_{name}", meta, snapshot)
    fp = build_fingerprint(
        evalset_dir=cfg.evalset.dir,
        eval_set=name,
        split="dev",
        metric_set="retrieval-v1",
        bootstrap_seed=cfg.eval.bootstrap_seed,
        config=snapshot,
        pipeline={
            "model": f"{r.mode}:{r.chunker}:{r.embedder}"
            + (f"+rerank:{cfg.rerank.model}" if cfg.rerank.enabled else "")
        },
    )
    write_json(run_dir / "fingerprint.json", fp)
    all_chunks = [
        chunk_ref(c.model_dump())
        for c in read_chunks(chunks_path(cfg.ingestion.processed_dir, r.chunker))
    ]
    texts = doc_texts(cfg.ingestion.processed_dir)
    client = make_client(cfg.vector_store)
    embedder = make_embedder(r.embedder, cfg.embedders[r.embedder])
    reranker = make_reranker(cfg.rerank) if cfg.rerank.enabled else None
    try:
        retriever = make_retriever(
            r,
            client,
            collection_name(cfg.vector_store.collection_prefix, r.chunker, r.embedder),
            embedder,
        )

        def ranked(q: str, k: int) -> list[Any]:
            # Reranking reorders the retrieved top-k; metrics @5/@10 read the new order.
            hits = retriever.retrieve(q, k)
            if reranker is not None:
                hits = reranker.rerank(q, hits)
            return [chunk_ref(h.payload) for h in hits]

        with timed_phase(run_dir, "retrieval", "retrieval", cfg.eval.budgets_s, len(queries)):
            rows, metrics = run_retrieval_eval(
                queries, ranked, all_chunks, texts, cfg.eval, r.top_k
            )
    finally:
        embedder.close()
        if reranker is not None:
            reranker.close()
        client.close()
    write_jsonl(run_dir / "per_item.jsonl", rows)
    write_json(run_dir / "metrics.json", metrics)
    ov = metrics["overall"]
    typer.echo(f"{run_dir}: {len(queries)} queries")
    for m, v in ov.items():
        typer.echo(f"  {m:22s} {v['value']:.3f}  [{v['low']:.3f}, {v['high']:.3f}]")


@eval_app.command("safety")
def eval_safety(
    config: ConfigOpt,
    models: ModelsOpt = LAPTOP,
    split: Split = Split.dev,
    tag: Annotated[str | None, typer.Option(help="Run tag (default: safety_<mode>).")] = None,
) -> None:
    """Run only the input guard over the safety set: red-flag recall and FPR."""
    from antibros.config import load_model_profile
    from antibros.eval.fingerprint import build_fingerprint
    from antibros.eval.runs import timed_phase
    from antibros.eval.safety import aggregate, make_guard, per_item_rows, run_safety
    from antibros.evalset.schemas import SafetyItem
    from antibros.io import write_jsonl

    _dev_only(split)
    cfg = _load(config)
    items = [it for it in _frozen_items(cfg, "safety") if it.split == split.value]
    profile = load_model_profile(models)
    guard = make_guard(cfg, profile)
    uses_llm = cfg.guard.mode in ("llm", "rule_llm")
    snapshot = {
        "app": cfg.model_dump(mode="json"),
        "models": profile.model_dump(mode="json") if uses_llm else None,
        "config_path": config.as_posix(),
    }
    run_dir = _start_run(
        cfg,
        tag or f"safety_{cfg.guard.mode}",
        {"kind": "safety", "eval_set": "safety", "split": split.value},
        snapshot,
    )
    write_json(
        run_dir / "fingerprint.json",
        build_fingerprint(
            evalset_dir=cfg.evalset.dir,
            eval_set="safety",
            split=split.value,
            metric_set="safety-v1",
            bootstrap_seed=cfg.eval.bootstrap_seed,
            config=snapshot,
            pipeline={
                "model": f"{cfg.guard.mode}:{profile.pipeline.model}"
                if uses_llm and profile.pipeline
                else cfg.guard.mode,
                "digest": profile.pipeline.digest if uses_llm and profile.pipeline else None,
            },
        ),
    )
    with timed_phase(run_dir, "safety", f"safety_{split.value}", cfg.eval.budgets_s, len(items)):
        records = run_safety(cast(list[SafetyItem], items), guard)
    rows = per_item_rows(records)
    write_jsonl(run_dir / "per_item.jsonl", rows)
    metrics = aggregate(rows, cfg.eval)
    write_json(run_dir / "metrics.json", metrics)
    typer.echo(f"{run_dir}: recall {metrics['red_flag_recall']}, fpr {metrics['fpr']}")


def _new_e2e_run(
    cfg: AppConfig, profile: ModelProfileT, config: Path, tag: str, split: str
) -> Path:
    from antibros.eval.fingerprint import build_fingerprint

    snapshot = {
        "app": cfg.model_dump(mode="json"),
        "models": profile.model_dump(mode="json"),
        "config_path": config.as_posix(),
    }
    run_dir = _start_run(cfg, tag, {"kind": "e2e", "eval_set": "golden", "split": split}, snapshot)
    p, j = profile.pipeline, profile.judge
    write_json(
        run_dir / "fingerprint.json",
        build_fingerprint(
            evalset_dir=cfg.evalset.dir,
            eval_set="golden",
            split=split,
            metric_set="e2e-v1",
            bootstrap_seed=cfg.eval.bootstrap_seed,
            config=snapshot,
            judge=j.model_dump() if j else None,
            pipeline=p.model_dump() if p else None,
        ),
    )
    return run_dir


@eval_app.command("predict")
def eval_predict(
    config: ConfigOpt,
    models: ModelsOpt,
    tag: Annotated[str, typer.Option(help="Run tag, [A-Za-z0-9_-] only.")],
    split: Split = Split.dev,
    limit: Annotated[int | None, typer.Option(help="Only the first N questions.")] = None,
    run: Annotated[Path | None, typer.Option(help="Resume this run directory.")] = None,
) -> None:
    """Phase 1: run the pipeline on the golden split and store predictions (resumable)."""
    from antibros.config import load_model_profile
    from antibros.eval.predict import make_pipeline, run_predict
    from antibros.eval.runs import timed_phase
    from antibros.evalset.schemas import GoldenItem
    from antibros.io import read_json

    _dev_only(split)
    if not is_valid_run_tag(tag):
        typer.echo(f"Invalid tag {tag!r}: use only [A-Za-z0-9_-].", err=True)
        raise typer.Exit(code=1)
    cfg = _load(config)
    profile = load_model_profile(models)
    items = [cast(GoldenItem, it) for it in _frozen_items(cfg, "golden") if it.split == split.value]
    if run is not None:
        meta = read_json(run / "run.json")
        if meta.get("split") != split.value or meta.get("kind") != "e2e":
            typer.echo(f"{run} is not an e2e run on split {split.value}", err=True)
            raise typer.Exit(code=1)
        run_dir = run
    else:
        run_dir = _new_e2e_run(cfg, profile, config, tag, split.value)
    pipeline = make_pipeline(cfg, profile)
    try:
        with timed_phase(run_dir, "predict", f"e2e_{split.value}", cfg.eval.budgets_s, len(items)):
            stats = run_predict(items, pipeline, run_dir, limit)
    finally:
        close = getattr(pipeline, "close", None)
        if callable(close):
            close()
    typer.echo(
        f"{run_dir}: predicted {stats.predicted} "
        f"(resumed after {stats.done_before}), errors {stats.errors}"
    )


class TuneParam(StrEnum):
    grade_min_rerank_score = "grade_min_rerank_score"
    verify_max_removed_ratio = "verify_max_removed_ratio"


@eval_app.command("tune")
def eval_tune(
    config: ConfigOpt,
    models: ModelsOpt,
    param: Annotated[TuneParam, typer.Option(help="Threshold to sweep.")],
    split: Split = Split.dev,
    out: Annotated[Path | None, typer.Option(help="Also write the table to this file.")] = None,
) -> None:
    """Try every pre-declared grid value (`tuning:` in the config) of one threshold on dev.

    Each value is its own run (predict + report, no judge): abstention F1 and the
    false-abstain rate need no judge. Earlier steps replay from the LLM cache."""
    from antibros.config import load_model_profile
    from antibros.eval.predict import run_predict
    from antibros.eval.runs import timed_phase
    from antibros.evalset.schemas import GoldenItem
    from antibros.graph.pipeline import RagPipeline
    from antibros.io import read_json, write_text

    _dev_only(split)
    cfg = _load(config)
    grid = getattr(cfg.tuning, param.value)
    p = cfg.pipeline
    needs = {
        TuneParam.grade_min_rerank_score: p.grade.enabled and cfg.rerank.enabled,
        TuneParam.verify_max_removed_ratio: p.verify.enabled,
    }
    if not grid or not needs[param]:
        typer.echo(f"{param.value}: empty grid or the step it tunes is disabled", err=True)
        raise typer.Exit(code=1)
    profile = load_model_profile(models)
    items = [cast(GoldenItem, it) for it in _frozen_items(cfg, "golden") if it.split == split.value]
    pipeline = RagPipeline(cfg, profile)
    rows = []
    try:
        for value in grid:
            data = cfg.model_dump(mode="json")
            if param is TuneParam.grade_min_rerank_score:
                data["pipeline"]["grade"]["min_rerank_score"] = value
            else:
                data["pipeline"]["verify"]["max_removed_ratio"] = value
            variant = AppConfig.model_validate(data)
            # Thresholds are read at run time and never change the graph shape, so the
            # loaded models are reused across grid values.
            pipeline.cfg = variant
            tag = f"tune_{param.value}_{str(value).replace('.', 'p')}"
            run_dir = _new_e2e_run(variant, profile, config, tag, split.value)
            with timed_phase(
                run_dir, "predict", f"e2e_{split.value}", cfg.eval.budgets_s, len(items)
            ):
                run_predict(items, pipeline, run_dir)
            eval_report(run_dir)
            ov = read_json(run_dir / "metrics.json")["overall"]
            rows.append((value, run_dir, ov["abstention_f1"], ov["false_abstain"]))
    finally:
        pipeline.close()
    lines = [
        f"# Threshold sweep: `{param.value}` ({split.value}, n={len(items)})",
        "",
        "| Value | Abstention F1 | False-abstain rate | Run |",
        "|---|---|---|---|",
    ]
    for value, run_dir, f1, fa in rows:
        lines.append(
            f"| {value} | {f1['value']:.3f} [{f1['low']:.3f}, {f1['high']:.3f}] | "
            f"{fa['value']:.3f} [{fa['low']:.3f}, {fa['high']:.3f}] | `{run_dir.as_posix()}` |"
        )
    md = "\n".join(lines) + "\n"
    if out is not None:
        write_text(out, md)
    typer.echo(md)


@eval_app.command("judge")
def eval_judge(run: RunArg, models: ModelsOpt) -> None:
    """Phase 2: score stored predictions with the judge (resumable), then write the report."""
    import yaml

    from antibros.config import AppConfig, load_model_profile
    from antibros.eval.judge import make_judge, run_judge
    from antibros.eval.predict import load_predictions
    from antibros.eval.runs import timed_phase
    from antibros.evalset.schemas import GoldenItem
    from antibros.io import read_json

    meta = read_json(run / "run.json")
    cfg = AppConfig.model_validate(
        yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))["app"]
    )
    configure_logging(cfg.log_level)
    profile = load_model_profile(models)
    fp = read_json(run / "fingerprint.json")
    if profile.judge and fp.get("judge_model") not in (None, profile.judge.model):
        typer.echo(
            f"Judge {profile.judge.model} differs from the run's {fp['judge_model']}", err=True
        )
        raise typer.Exit(code=1)
    items = [
        cast(GoldenItem, it) for it in _frozen_items(cfg, "golden") if it.split == meta["split"]
    ]
    preds = load_predictions(run)
    judge = make_judge(cfg, profile)
    try:
        with timed_phase(run, "judge", f"e2e_{meta['split']}", cfg.eval.budgets_s, len(preds)):
            stats = run_judge(items, preds, judge, run)
    finally:
        close = getattr(judge, "close", None)
        if callable(close):
            close()
    typer.echo(
        f"{run}: judged {stats.judged} (resumed after {stats.done_before}), errors {stats.errors}"
    )
    eval_report(run)


@eval_app.command("report")
def eval_report(run: RunArg) -> None:
    """Compute every e2e metric with bootstrap CIs into per_item.jsonl and metrics.json."""
    import yaml

    from antibros.config import AppConfig
    from antibros.eval.report import write_report
    from antibros.evalset.schemas import GoldenItem
    from antibros.io import read_json

    meta = read_json(run / "run.json")
    cfg = AppConfig.model_validate(
        yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))["app"]
    )
    items = [
        cast(GoldenItem, it) for it in _frozen_items(cfg, "golden") if it.split == meta["split"]
    ]
    metrics, with_retrieval = write_report(run, cfg, items)
    if not with_retrieval:
        typer.echo("Chunk file not found: skipped Evidence Recall@k (run `antibros index`)")
    typer.echo(f"{run}: {metrics['n_items']} items, status {metrics['status_counts']}")


@eval_app.command("final")
def eval_final(
    plan: Annotated[Path, typer.Option(help="Pre-registered plan.")] = Path(
        "configs/final_eval_plan.yaml"
    ),
    rehearsal: Annotated[
        bool, typer.Option(help="Dry run on dev with --limit questions; no test_runs.log entry.")
    ] = False,
    limit: Annotated[int, typer.Option(help="Questions per run in a rehearsal.")] = 3,
    rerun: Annotated[
        list[str] | None,
        typer.Option(help="Re-run this failed test run once (infrastructure faults only)."),
    ] = None,
    reason: Annotated[str, typer.Option(help="Why the re-run is needed (deviations.md).")] = "",
) -> None:
    """The only way to run the test split (M5): checks the locked plan, then runs it once."""
    from antibros.eval.final import FinalError, InfraFailure, default_deps, read_git_state, run_plan

    configure_logging("INFO")
    try:
        result = run_plan(
            plan,
            runs_dir=Path("experiments"),
            deps=default_deps(typer.echo),
            git=read_git_state(plan),
            rehearsal=rehearsal,
            limit=limit,
            rerun={name: reason for name in rerun or []},
        )
    except FinalError as e:
        typer.echo(f"Refusing to run: {e}", err=True)
        raise typer.Exit(code=1) from e
    except InfraFailure as e:
        typer.echo(
            f"{e}\nLogged as failed in experiments/test_runs.log. Fix the infrastructure, then "
            "re-run that run once with --rerun <name> --reason '...' (recorded in "
            "experiments/deviations.md). Never re-run because of results.",
            err=True,
        )
        raise typer.Exit(code=1) from e
    for name, run_dir in result.run_dirs.items():
        typer.echo(f"{name}: {run_dir.as_posix()}")
    if result.comparison is not None:
        typer.echo(f"Comparison: {result.comparison.as_posix()}")


@eval_app.command("errors")
def eval_errors(
    run: Annotated[list[Path], typer.Option("--run", help="Run directory (repeatable).")],
    config: Annotated[Path, typer.Option(help="Error-analysis settings.")] = Path(
        "configs/error_analysis.yaml"
    ),
    out: Annotated[Path | None, typer.Option(help="Also write the draft to this file.")] = None,
) -> None:
    """Draft labels for every wrong item of e2e/safety runs (input to error_analysis.md)."""
    from antibros.eval.errors import load_error_config, render_errors, run_errors
    from antibros.io import read_jsonl, write_text

    cfg = load_error_config(config)
    md = render_errors(
        [(r, run_errors(r, cfg), len(read_jsonl(r / "per_item.jsonl"))) for r in run]
    )
    if out is not None:
        write_text(out, md)
    typer.echo(md)


@app.command()
def compare(
    run_a: Path,
    run_b: Path,
    config: ConfigOpt = DEFAULT_CONFIG,
    out: Annotated[Path | None, typer.Option(help="Also write the table to this file.")] = None,
) -> None:
    """Paired bootstrap comparison of two runs (refuses on hard fingerprint mismatch)."""
    from antibros.eval.compare import FingerprintMismatch, compare_runs, render_markdown
    from antibros.io import write_text

    cfg = load_app_config(config)
    try:
        cmp = compare_runs(run_a, run_b, cfg.eval)
    except FingerprintMismatch as e:
        typer.echo(f"Refusing to compare: {e}. These runs are not comparable.", err=True)
        raise typer.Exit(code=1) from e
    md = render_markdown(cmp, run_a.name, run_b.name)
    if out is not None:
        write_text(out, md)
    typer.echo(md)


@app.command()
def summarize(
    runs: Annotated[list[Path], typer.Argument(help="Run directories (same kind).")],
    out: Annotated[Path | None, typer.Option(help="Also write the tables to this file.")] = None,
) -> None:
    """Markdown tables of metrics with 95% CIs for several runs (for reports)."""
    from antibros.eval.summary import render_summary
    from antibros.io import write_text

    md = render_summary(runs)
    if out is not None:
        write_text(out, md)
    typer.echo(md)


@plan_app.command("lock")
def plan_lock(
    guard_run: Annotated[
        list[Path], typer.Option("--guard-run", help="Dev safety run of each guard variant.")
    ],
    plan: Annotated[Path, typer.Option(help="Plan YAML.")] = Path("configs/final_eval_plan.yaml"),
    config: ConfigOpt = DEFAULT_CONFIG,
) -> None:
    """Select the guard by the declared rule and pin config hashes and model digests."""
    from antibros.eval.plan import PlanError, load_plan, lock_plan, save_plan

    cfg = load_app_config(config)
    try:
        locked = lock_plan(load_plan(plan), guard_run, cfg.evalset.dir)
    except PlanError as e:
        typer.echo(f"Cannot lock plan: {e}", err=True)
        raise typer.Exit(code=1) from e
    save_plan(plan, locked)
    typer.echo(f"Locked {plan}; selected guard {locked.guard_selection.selected}")


@app.command()
def serve(
    host: str = "127.0.0.1",
    port: int = 8000,
    config: ConfigOpt = Path("configs/baseline.yaml"),
    models: ModelsOpt = Path("configs/models/laptop.yaml"),
    ui_dir: Annotated[
        Path, typer.Option(help="Built frontend to serve at / (skipped if not built).")
    ] = Path("frontend/dist"),
) -> None:
    """Start the FastAPI application (and the built web UI, if present)."""
    import uvicorn

    from antibros.api import create_app

    if not (ui_dir / "index.html").is_file():
        typer.echo(f"No built UI in {ui_dir}: API only (build it with `uv run poe ui-build`).")
    uvicorn.run(create_app(config, models, ui_dir=ui_dir), host=host, port=port)


if __name__ == "__main__":
    app()
