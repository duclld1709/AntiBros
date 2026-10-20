"""LangGraph RAG pipeline (spec §5.1-5.2).

``input_guard → query_analyze → retrieve → rerank → grade_context → generate →
verify_citations → finalize → output_guard``. Every optional step is switched on or off in
config, and a disabled step is simply not a node of the compiled graph, so ablation
configs change the graph shape and nothing else. ``baseline`` is ``query_analyze →
retrieve → generate → finalize`` (M3).

Each node reads the state and returns a partial update; LLM accounting (``stats``) and the
per-node trace (``steps``) are merged by LangGraph reducers. Nothing is kept on the pipeline
object between requests, so concurrent API requests cannot see each other's results.
"""

from __future__ import annotations

import operator
import re
import time
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

from antibros.config import AppConfig, ModelProfile, Settings
from antibros.embedding.base import Embedder, make_embedder
from antibros.eval.schemas import Citation, Claim, Context, GuardLabel, Prediction, PredStatus
from antibros.generation.citations import clean_claims, format_passages, number_citations
from antibros.generation.responses import Responses, load_responses
from antibros.guardrails.guards import Guard, OutputGuard, load_rules, make_input_guard
from antibros.indexing.qdrant_store import collection_name, make_client
from antibros.ingestion.pipeline import load_documents
from antibros.ingestion.schemas import Document
from antibros.llm import LLMClient, LLMParams, build_client
from antibros.llm.structured import CallStats, T, call_structured
from antibros.prompts import load_prompt
from antibros.reranking.base import RERANK_SCORE, Reranker, make_reranker
from antibros.retrieval.base import RetrievalFilters, ScoredChunk, make_retriever, rrf


class QueryAnalysis(BaseModel):
    language: Literal["vi", "en"]
    query_en: str = Field(min_length=1)


class QueryRewrite(BaseModel):
    queries: list[str] = Field(min_length=1)


class GradeVerdict(BaseModel):
    sufficient: bool
    reason: str = ""


class GeneratedClaim(BaseModel):
    text: str = Field(min_length=1)
    citation_ids: list[int] = Field(min_length=1)


class GeneratedAnswer(BaseModel):
    answer: str
    claims: list[GeneratedClaim]
    status: Literal["answered", "abstained"]


class ClaimVerdict(BaseModel):
    claim: int
    supported: bool


class VerifyResult(BaseModel):
    verdicts: list[ClaimVerdict]


class GraphState(TypedDict, total=False):
    question: str
    language: str
    filters: RetrievalFilters | None
    guard_label: GuardLabel | None
    queries: list[str]
    tried_queries: list[str]
    attempt: int
    retry: bool
    chunks: list[ScoredChunk]
    claims: list[Claim]  # citation IDs are context positions until ``finalize``
    model_answer: str
    removed: int
    abstain_reason: str | None
    # outputs
    status: PredStatus
    answer: str
    citations: list[Citation]
    disclaimer: str | None
    # merged by reducers
    stats: Annotated[CallStats, operator.add]
    steps: Annotated[list[dict[str, Any]], operator.add]


@dataclass
class PipelineResult:
    prediction: Prediction
    chunks: list[ScoredChunk] = field(default_factory=list)


def detect_language(text: str) -> Literal["vi", "en"]:
    vietnamese = r"[ăâđêôơưĂÂĐÊÔƠƯ]|[àáạảãằắặẳẵầấậẩẫèéẹẻẽềếệểễìíịỉĩòóọỏõồốộổỗờớợởỡùúụủũừứựửữỳýỵỷỹ]"
    if re.search(vietnamese, text):
        return "vi"
    return "en"


def _step(node: str, started: float, **details: Any) -> list[dict[str, Any]]:
    return [{"node": node, "ms": round((time.perf_counter() - started) * 1000, 2), **details}]


def to_context(chunk: ScoredChunk) -> Context:
    return Context(
        chunk_id=chunk.chunk_id,
        doc_id=str(chunk.payload.get("doc_id", "")),
        text=str(chunk.payload.get("text", "")),
    )


class RagPipeline:
    def __init__(
        self,
        cfg: AppConfig,
        profile: ModelProfile,
        *,
        llm: LLMClient | None = None,
        client: QdrantClient | None = None,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        documents: dict[str, Document] | None = None,
    ):
        if profile.pipeline is None:
            raise ValueError("model profile needs a pipeline model")
        self.cfg = cfg
        self.spec = profile.pipeline
        self.llm = llm or build_client(
            profile.pipeline, Settings(), cfg.llm_cache.dir if cfg.llm_cache.enabled else None
        )
        self.client = client or make_client(cfg.vector_store)
        r = cfg.retrieval
        self.embedder = embedder or make_embedder(r.embedder, cfg.embedders[r.embedder])
        self.retriever = make_retriever(
            r,
            self.client,
            collection_name(cfg.vector_store.collection_prefix, r.chunker, r.embedder),
            self.embedder,
        )
        self.reranker: Reranker | None = None
        if cfg.rerank.enabled:
            self.reranker = reranker or make_reranker(cfg.rerank)
        self.documents = (
            documents
            if documents is not None
            else {doc.doc_id: doc for doc in load_documents(cfg.ingestion.processed_dir)}
        )
        self.responses: Responses = load_responses(cfg.prompts_dir)
        self.guard: Guard | None = None
        self.output_guard: OutputGuard | None = None
        if cfg.guard.input_enabled:
            self.guard = make_input_guard(cfg, self.llm, self._params(cfg.guard.llm_max_tokens))
        if cfg.guard.output_enabled:
            self.output_guard = OutputGuard(load_rules(cfg.guard.rules_path))
        self.graph = self._build_graph()

    # --- helpers -------------------------------------------------------------------------

    def _params(self, max_tokens: int) -> LLMParams:
        return LLMParams(
            temperature=self.spec.temperature,
            max_tokens=max_tokens,
            num_ctx=self.spec.num_ctx,
            think=self.spec.think,
        )

    def _call(
        self, prompt: str, schema: type[T], max_tokens: int, **values: str
    ) -> tuple[T | None, CallStats]:
        messages = load_prompt(self.cfg.prompts_dir, prompt).render(**values)
        return call_structured(self.llm, messages, schema, self._params(max_tokens))

    def _context(self, state: GraphState) -> list[ScoredChunk]:
        chunks = state.get("chunks", [])
        k = self.cfg.pipeline.context_k
        return chunks if k is None else chunks[:k]

    # --- graph ---------------------------------------------------------------------------

    def _build_graph(self) -> Any:
        from langgraph.graph import END, START, StateGraph

        p, g = self.cfg.pipeline, StateGraph(GraphState)
        nodes: dict[str, Any] = {
            "query_analyze": self.query_analyze,
            "retrieve": self.retrieve,
            "generate": self.generate,
            "finalize": self.finalize,
            "abstain": self.abstain,
        }
        if self.guard is not None:
            nodes["input_guard"] = self.input_guard
        if self.reranker is not None:
            nodes["rerank"] = self.rerank
        if p.grade.enabled:
            nodes["grade_context"] = self.grade_context
        if p.verify.enabled:
            nodes["verify_citations"] = self.verify_citations
        if self.output_guard is not None:
            nodes["output_guard"] = self.apply_output_guard
        for name, fn in nodes.items():
            g.add_node(name, fn)

        if "input_guard" in nodes:
            g.add_edge(START, "input_guard")
            g.add_conditional_edges(
                "input_guard",
                lambda s: END if s.get("status") else "query_analyze",
                {END: END, "query_analyze": "query_analyze"},
            )
        else:
            g.add_edge(START, "query_analyze")
        g.add_edge("query_analyze", "retrieve")
        before_grade = "rerank" if "rerank" in nodes else "retrieve"
        if before_grade == "rerank":
            g.add_edge("retrieve", "rerank")
        if "grade_context" in nodes:
            g.add_edge(before_grade, "grade_context")
            g.add_conditional_edges(
                "grade_context",
                lambda s: (
                    "query_analyze"
                    if s.get("retry")
                    else ("abstain" if s.get("abstain_reason") else "generate")
                ),
                ["query_analyze", "abstain", "generate"],
            )
        else:
            g.add_edge(before_grade, "generate")
        after_generate = "verify_citations" if "verify_citations" in nodes else "finalize"
        g.add_conditional_edges(
            "generate",
            lambda s: "abstain" if s.get("abstain_reason") else after_generate,
            ["abstain", after_generate],
        )
        if "verify_citations" in nodes:
            g.add_conditional_edges(
                "verify_citations",
                lambda s: "abstain" if s.get("abstain_reason") else "finalize",
                ["abstain", "finalize"],
            )
        if "output_guard" in nodes:
            g.add_edge("finalize", "output_guard")
            g.add_edge("output_guard", END)
        else:
            g.add_edge("finalize", END)
        g.add_edge("abstain", END)
        return g.compile()

    # --- nodes ---------------------------------------------------------------------------

    def input_guard(self, state: GraphState) -> GraphState:
        assert self.guard is not None
        started = time.perf_counter()
        d = self.guard.classify(state["question"])
        out: GraphState = {
            "guard_label": d.label,
            "stats": d.stats,
            "steps": _step("input_guard", started, label=d.label, layer=d.layer, reason=d.reason),
        }
        if d.label == "medical_red_flag":
            out |= {
                "status": "safety_redirect",
                "answer": self.responses.get("safety_redirect", state["language"]),
            }
        elif d.label == "off_topic":
            out |= {
                "status": "abstained",
                "abstain_reason": "off_topic",
                "answer": self.responses.get("off_topic", state["language"]),
            }
        return out

    def query_analyze(self, state: GraphState) -> GraphState:
        """First pass: rewrite into English variants (``query_rewrite``) or, as in M3, only
        translate Vietnamese. A ``grade_context`` retry asks for new formulations."""
        started = time.perf_counter()
        question, language = state["question"], state["language"]
        attempt = state.get("attempt", 0)
        p = self.cfg.pipeline
        stats = CallStats()
        if attempt > 0 or p.query_rewrite.enabled:
            tried = state.get("tried_queries", [])
            rewritten, stats = self._call(
                "query_rewrite",
                QueryRewrite,
                p.query_rewrite.max_tokens,
                question=question,
                max_variants=str(p.query_rewrite.max_variants),
                previous="; ".join(tried) if tried else "(none)",
            )
            variants = [q.strip() for q in (rewritten.queries if rewritten else []) if q.strip()]
            variants = variants[: p.query_rewrite.max_variants]
            # Keep the user's wording too: bge-m3 retrieves cross-lingually.
            queries = [*variants, question] if attempt == 0 else variants or [question]
        elif language == "vi":
            analysis, stats = self._call(
                "query_analyze", QueryAnalysis, p.translate_max_tokens, question=question
            )
            # On invalid output, retrieval on the Vietnamese wording is still cross-lingual.
            queries = [analysis.query_en] if analysis else [question]
        else:
            queries = [question]
        queries = list(dict.fromkeys(queries))
        return {
            "queries": queries,
            "stats": stats,
            "steps": _step("query_analyze", started, attempt=attempt, queries=queries),
        }

    def retrieve(self, state: GraphState) -> GraphState:
        """One ranked list per query, fused with RRF when there are several."""
        started = time.perf_counter()
        k, filters = self.cfg.retrieval.top_k, state.get("filters")
        lists = [self.retriever.retrieve(q, k, filters) for q in state["queries"]]
        if len(lists) == 1:
            chunks = lists[0]
        else:
            payloads = {c.chunk_id: c.payload for hits in lists for c in hits}
            fused = rrf([[c.chunk_id for c in hits] for hits in lists], self.cfg.retrieval.rrf_k)
            chunks = [ScoredChunk(cid, score, payloads[cid]) for cid, score in fused[:k]]
        return {
            "chunks": chunks,
            "steps": _step(
                "retrieve",
                started,
                hits=[[c.chunk_id, round(c.score, 6)] for c in chunks],
            ),
        }

    def rerank(self, state: GraphState) -> GraphState:
        assert self.reranker is not None
        started = time.perf_counter()
        chunks = self.reranker.rerank(state["question"], state.get("chunks", []))
        return {
            "chunks": chunks,
            "steps": _step(
                "rerank", started, scores=[[c.chunk_id, round(c.score, 6)] for c in chunks]
            ),
        }

    def grade_context(self, state: GraphState) -> GraphState:
        """Sufficient = reranker score passes (when reranking is on) AND the LLM says so.
        The LLM is skipped when the score already fails. Invalid LLM output counts as
        insufficient (abstaining is the safe side, ADR-0007)."""
        started = time.perf_counter()
        g = self.cfg.pipeline.grade
        context = self._context(state)
        top = max((float(c.payload.get(RERANK_SCORE, 0.0)) for c in context), default=0.0)
        score_ok = bool(context)
        if score_ok and self.reranker is not None and g.min_rerank_score is not None:
            score_ok = top >= g.min_rerank_score
        llm_ok: bool | None = None
        stats = CallStats()
        if score_ok and g.use_llm:
            verdict, stats = self._call(
                "grade",
                GradeVerdict,
                g.max_tokens,
                question=state["question"],
                evidence=format_passages([to_context(c).text for c in context]),
            )
            llm_ok = bool(verdict and verdict.sufficient)
        sufficient = score_ok and llm_ok is not False
        attempt = state.get("attempt", 0)
        retry = not sufficient and attempt < g.max_retries
        out: GraphState = {
            "retry": retry,
            "stats": stats,
            "steps": _step(
                "grade_context",
                started,
                attempt=attempt,
                top_rerank_score=round(top, 6) if self.reranker is not None else None,
                score_ok=score_ok,
                llm_sufficient=llm_ok,
                sufficient=sufficient,
            ),
        }
        if retry:
            out |= {
                "attempt": attempt + 1,
                "tried_queries": state.get("tried_queries", []) + state.get("queries", []),
            }
        elif not sufficient:
            out["abstain_reason"] = "insufficient_context"
        return out

    def generate(self, state: GraphState) -> GraphState:
        started = time.perf_counter()
        context = [to_context(c) for c in self._context(state)]
        if not context:
            return {
                "abstain_reason": "no_context",
                "steps": _step("generate", started, skipped="no_context"),
            }
        generated, stats = self._call(
            "generate",
            GeneratedAnswer,
            self.cfg.pipeline.generate_max_tokens,
            question=state["question"],
            language="Vietnamese" if state["language"] == "vi" else "English",
            evidence=format_passages([c.text for c in context]),
        )
        out: GraphState = {"stats": stats}
        if generated is None:
            out |= {"abstain_reason": "generator_invalid_output"}
            claims: list[Claim] = []
        else:
            claims = clean_claims(
                [(c.text, c.citation_ids) for c in generated.claims], len(context)
            )
            out |= {
                "claims": claims,
                "removed": len(generated.claims) - len(claims),
                "model_answer": generated.answer,
            }
            if generated.status != "answered" or not claims:
                out["abstain_reason"] = "generator_abstained"
        out["steps"] = _step(
            "generate",
            started,
            valid=generated is not None,
            model_status=generated.status if generated else None,
            n_claims=len(claims),
        )
        return out

    def verify_citations(self, state: GraphState) -> GraphState:
        """One call checks every claim against the passages it cites (ADR-0006)."""
        started = time.perf_counter()
        v = self.cfg.pipeline.verify
        context = [to_context(c) for c in self._context(state)]
        claims = state.get("claims", [])
        blocks = [
            f"Claim {i}: {c.text}\n"
            + format_passages([context[n - 1].text for n in c.citation_ids], c.citation_ids)
            for i, c in enumerate(claims, 1)
        ]
        result, stats = self._call("verify", VerifyResult, v.max_tokens, claims="\n\n".join(blocks))
        supported = {vd.claim for vd in result.verdicts if vd.supported} if result else set()
        kept = [c for i, c in enumerate(claims, 1) if i in supported]
        removed_now = len(claims) - len(kept)
        out: GraphState = {
            "claims": kept,
            "removed": state.get("removed", 0) + removed_now,
            "stats": stats,
            "steps": _step(
                "verify_citations",
                started,
                valid=result is not None,
                n_claims=len(claims),
                removed=removed_now,
            ),
        }
        if result is None:
            out["abstain_reason"] = "verify_invalid_output"
        elif not kept or removed_now / len(claims) > v.max_removed_ratio:
            out["abstain_reason"] = "unsupported_claims"
        return out

    def finalize(self, state: GraphState) -> GraphState:
        context = [to_context(c) for c in self._context(state)]
        numbered = number_citations(state.get("claims", []), context)
        return {
            "status": "answered",
            "answer": numbered.answer,
            "claims": numbered.claims,
            "citations": numbered.citations,
        }

    def abstain(self, state: GraphState) -> GraphState:
        reason = state.get("abstain_reason")
        model_answer = state.get("model_answer", "").strip()
        answer = (
            model_answer
            if reason == "generator_abstained" and model_answer
            else self.responses.get("abstain", state["language"])
        )
        return {"status": "abstained", "answer": answer, "claims": [], "citations": []}

    def apply_output_guard(self, state: GraphState) -> GraphState:
        assert self.output_guard is not None
        started = time.perf_counter()
        language = state["language"]
        reason = self.output_guard.blocked_reason(state.get("answer", ""))
        if reason is not None:
            return {
                "status": "safety_redirect",
                "answer": self.responses.get("output_blocked", language),
                "claims": [],
                "citations": [],
                "steps": _step("output_guard", started, blocked=reason),
            }
        return {
            "disclaimer": self.responses.get("disclaimer", language),
            "steps": _step("output_guard", started, blocked=None),
        }

    # --- entry points --------------------------------------------------------------------

    def run(
        self,
        item_id: str,
        question: str,
        language: str | None = None,
        filters: RetrievalFilters | None = None,
    ) -> PipelineResult:
        started = time.perf_counter()
        state: GraphState = self.graph.invoke(
            {
                "question": question,
                "language": language or detect_language(question),
                "filters": filters,
                "stats": CallStats(),
                "steps": [],
            }
        )
        stats: CallStats = state["stats"]
        status = state.get("status", "abstained")
        chunks = state.get("chunks", [])
        prediction = Prediction(
            id=item_id,
            question=question,
            language=state["language"],
            status=status,
            answer=state.get("answer", ""),
            claims=state.get("claims", []) if status == "answered" else [],
            citations=state.get("citations", []) if status == "answered" else [],
            contexts=[to_context(c) for c in self._context(state)],
            latency_ms=(time.perf_counter() - started) * 1000,
            prompt_tokens=stats.prompt_tokens,
            completion_tokens=stats.completion_tokens,
            cost_usd=stats.cost_usd or None,
            llm_cache_hits=stats.llm_cache_hits,
            llm_calls=stats.llm_calls,
            structured_outputs=stats.structured_outputs,
            valid_structured_outputs=stats.valid_structured_outputs,
            guard_label=state.get("guard_label"),
            retries=state.get("attempt", 0),
            unsupported_claims_removed=state.get("removed", 0),
            abstain_reason=state.get("abstain_reason") if status == "abstained" else None,
            disclaimer=state.get("disclaimer"),
            steps=state.get("steps", []),
        )
        return PipelineResult(prediction, chunks)

    def answer(self, item_id: str, question: str, language: str) -> Prediction:
        return self.run(item_id, question, language).prediction

    def close(self) -> None:
        self.embedder.close()
        if self.reranker is not None:
            self.reranker.close()
        self.client.close()
        for hook in ("unload", "close"):
            fn = getattr(self.llm, hook, None)
            if callable(fn):
                fn()
