"""Minimal three-node LangGraph RAG pipeline used by M3."""

from __future__ import annotations

import re
import time
from typing import Any, Literal, TypedDict, cast

from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

from antibros.config import AppConfig, ModelProfile, Settings
from antibros.embedding.base import Embedder, make_embedder
from antibros.eval.schemas import Citation, Claim, Context, Prediction
from antibros.indexing.qdrant_store import collection_name, make_client
from antibros.ingestion.pipeline import load_documents
from antibros.llm import LLMClient, LLMParams, StructuredOutputError, build_client
from antibros.prompts import load_prompt
from antibros.retrieval.base import RetrievalFilters, ScoredChunk, make_retriever


class QueryAnalysis(BaseModel):
    language: Literal["vi", "en"]
    query_en: str = Field(min_length=1)


class GeneratedClaim(BaseModel):
    text: str = Field(min_length=1)
    citation_ids: list[int] = Field(min_length=1)


class GeneratedAnswer(BaseModel):
    answer: str
    claims: list[GeneratedClaim]
    status: Literal["answered", "abstained"]


class GraphState(TypedDict, total=False):
    item_id: str
    question: str
    language: str
    query: str
    chunks: list[ScoredChunk]
    filters: RetrievalFilters | None
    prediction: Prediction
    timings: dict[str, float]
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    llm_cache_hits: int
    llm_calls: int
    structured_outputs: int
    valid_structured_outputs: int


class RagPipeline:
    def __init__(
        self,
        cfg: AppConfig,
        profile: ModelProfile,
        *,
        llm: LLMClient | None = None,
        client: QdrantClient | None = None,
        embedder: Embedder | None = None,
    ):
        if profile.pipeline is None:
            raise ValueError("model profile needs a pipeline model")
        self.cfg = cfg
        self.num_ctx = profile.pipeline.num_ctx
        self.llm = llm or build_client(
            profile.pipeline, Settings(), cfg.llm_cache.dir if cfg.llm_cache.enabled else None
        )
        self.client = client or make_client(cfg.vector_store)
        self.embedder = embedder or make_embedder(
            cfg.retrieval.embedder, cfg.embedders[cfg.retrieval.embedder]
        )
        self.retriever = make_retriever(
            cfg.retrieval,
            self.client,
            collection_name(
                cfg.vector_store.collection_prefix,
                cfg.retrieval.chunker,
                cfg.retrieval.embedder,
            ),
            self.embedder,
        )
        self.documents = {doc.doc_id: doc for doc in load_documents(cfg.ingestion.processed_dir)}
        self.last_trace: dict[str, object] = {}
        self.last_chunks: list[ScoredChunk] = []
        self.graph = self._build_graph()

    def _build_graph(self) -> Any:
        from langgraph.graph import END, START, StateGraph

        graph = StateGraph(GraphState)
        graph.add_node("query_analyze", self._query_analyze)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("generate", self._generate)
        graph.add_edge(START, "query_analyze")
        graph.add_edge("query_analyze", "retrieve")
        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", END)
        return graph.compile()

    def _query_analyze(self, state: GraphState) -> GraphState:
        started = time.perf_counter()
        language = state.get("language") or detect_language(state["question"])
        query = state["question"]
        tokens = completion = 0
        cost = 0.0
        cache_hits = state.get("llm_cache_hits", 0)
        calls = state.get("llm_calls", 0)
        structured = state.get("structured_outputs", 0)
        valid_structured = state.get("valid_structured_outputs", 0)
        if language == "vi":
            prompt = load_prompt(self.cfg.prompts_dir, "query_analyze")
            calls += 1
            structured += 1
            try:
                result = self.llm.complete(
                    prompt.render(question=state["question"]),
                    schema=QueryAnalysis,
                    params=LLMParams(temperature=0, max_tokens=128, num_ctx=self.num_ctx),
                )
                parsed = QueryAnalysis.model_validate(result.parsed)
                query = parsed.query_en
                language = "vi"
                valid_structured += 1
                cache_hits += int(result.cached)
                tokens += result.usage.prompt_tokens
                completion += result.usage.completion_tokens
                cost += result.cost_usd or 0.0
            except StructuredOutputError as e:
                if e.cached:
                    calls -= 1
                    cache_hits += 1
                # Retrieval on the original Vietnamese wording remains cross-lingual.
                query = state["question"]
            except Exception:
                # Retrieval on the original Vietnamese wording remains cross-lingual.
                query = state["question"]
        timings = dict(state.get("timings", {}))
        timings["query_analyze"] = time.perf_counter() - started
        return {
            "language": language,
            "query": query,
            "timings": timings,
            "prompt_tokens": state.get("prompt_tokens", 0) + tokens,
            "completion_tokens": state.get("completion_tokens", 0) + completion,
            "cost_usd": state.get("cost_usd", 0.0) + cost,
            "llm_cache_hits": cache_hits,
            "llm_calls": calls,
            "structured_outputs": structured,
            "valid_structured_outputs": valid_structured,
        }

    def _retrieve(self, state: GraphState) -> GraphState:
        started = time.perf_counter()
        chunks = self.retriever.retrieve(
            state["query"], self.cfg.retrieval.top_k, state.get("filters")
        )
        self.last_chunks = chunks
        timings = dict(state.get("timings", {}))
        timings["retrieve"] = time.perf_counter() - started
        return {"chunks": chunks, "timings": timings}

    def _generate(self, state: GraphState) -> GraphState:
        started = time.perf_counter()
        chunks = state.get("chunks", [])
        contexts = [
            Context(
                chunk_id=c.chunk_id,
                doc_id=str(c.payload.get("doc_id", "")),
                text=str(c.payload.get("text", "")),
            )
            for c in chunks
        ]
        evidence = "\n\n".join(f"[{i}] {c.text}" for i, c in enumerate(contexts, 1))
        prompt = load_prompt(self.cfg.prompts_dir, "generate")
        try:
            result = self.llm.complete(
                prompt.render(
                    question=state["question"],
                    language="Vietnamese" if state["language"] == "vi" else "English",
                    evidence=evidence or "(no evidence retrieved)",
                ),
                schema=GeneratedAnswer,
                params=LLMParams(temperature=0, max_tokens=512, num_ctx=self.num_ctx),
            )
            generated = GeneratedAnswer.model_validate(result.parsed)
            tokens_in = state.get("prompt_tokens", 0) + result.usage.prompt_tokens
            tokens_out = state.get("completion_tokens", 0) + result.usage.completion_tokens
            cost = state.get("cost_usd", 0.0) + (result.cost_usd or 0.0)
            llm_calls = state.get("llm_calls", 0) + 1
            structured_outputs = state.get("structured_outputs", 0) + 1
            valid_structured_outputs = state.get("valid_structured_outputs", 0) + 1
            llm_cache_hits = state.get("llm_cache_hits", 0) + int(result.cached)
        except StructuredOutputError as e:
            generated = GeneratedAnswer(answer="", claims=[], status="abstained")
            tokens_in = state.get("prompt_tokens", 0)
            tokens_out = state.get("completion_tokens", 0)
            cost = state.get("cost_usd", 0.0)
            llm_calls = state.get("llm_calls", 0) + int(not e.cached)
            structured_outputs = state.get("structured_outputs", 0) + 1
            valid_structured_outputs = state.get("valid_structured_outputs", 0)
            llm_cache_hits = state.get("llm_cache_hits", 0) + int(e.cached)
        except Exception:
            generated = GeneratedAnswer(answer="", claims=[], status="abstained")
            tokens_in = state.get("prompt_tokens", 0)
            tokens_out = state.get("completion_tokens", 0)
            cost = state.get("cost_usd", 0.0)
            llm_calls = state.get("llm_calls", 0) + 1
            structured_outputs = state.get("structured_outputs", 0) + 1
            valid_structured_outputs = state.get("valid_structured_outputs", 0)
            llm_cache_hits = state.get("llm_cache_hits", 0)

        claims: list[Claim] = []
        citation_map: dict[int, int] = {}
        for raw in generated.claims:
            claim_text = raw.text
            valid = sorted({n for n in raw.citation_ids if 1 <= n <= len(contexts)})
            if not valid:
                continue
            claim_text = re.sub(r"\s*\[\d+\]", "", claim_text).strip()
            claims.append(Claim(text=claim_text, citation_ids=valid))
            for n in valid:
                citation_map.setdefault(n, n)
        citations = [
            Citation(id=n, chunk_id=contexts[n - 1].chunk_id) for n in sorted(citation_map)
        ]
        if generated.status == "answered" and claims:
            answer = "\n".join(
                f"{claim.text} " + "".join(f"[{n}]" for n in claim.citation_ids) for claim in claims
            )
            status: Literal["answered", "abstained"] = "answered"
        else:
            answer = (
                generated.answer
                or "The available sources do not provide enough evidence to answer."
            )
            status = "abstained"
            claims, citations = [], []
        timings = dict(state.get("timings", {}))
        timings["generate"] = time.perf_counter() - started
        self.last_trace = {
            "timings": timings,
            "retrieved": [c.chunk_id for c in chunks],
            "unsupported_claims_removed": len(generated.claims) - len(claims),
            "llm_cache_hits": llm_cache_hits,
            "llm_calls": llm_calls,
            "structured_outputs": structured_outputs,
            "valid_structured_outputs": valid_structured_outputs,
        }
        return {
            "prediction": Prediction(
                id=state["item_id"],
                question=state["question"],
                language=state["language"],
                status=status,
                answer=answer,
                claims=claims,
                citations=citations,
                contexts=contexts,
                prompt_tokens=tokens_in,
                completion_tokens=tokens_out,
                cost_usd=cost if cost else None,
                llm_cache_hits=llm_cache_hits,
                llm_calls=llm_calls,
                structured_outputs=structured_outputs,
                valid_structured_outputs=valid_structured_outputs,
            )
        }

    def answer(
        self,
        item_id: str,
        question: str,
        language: str,
        filters: RetrievalFilters | None = None,
    ) -> Prediction:
        state = self.graph.invoke(
            {
                "item_id": item_id,
                "question": question,
                "language": language,
                "filters": filters,
                "timings": {},
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cost_usd": 0.0,
                "llm_cache_hits": 0,
                "llm_calls": 0,
                "structured_outputs": 0,
                "valid_structured_outputs": 0,
            }
        )
        return cast(Prediction, state["prediction"])

    def ask(
        self, question: str, filters: RetrievalFilters | None = None
    ) -> tuple[Prediction, list[ScoredChunk]]:
        language = detect_language(question)
        prediction = self.answer("api", question, language, filters)
        return prediction, self.last_chunks

    def close(self) -> None:
        self.embedder.close()
        self.client.close()
        unload = getattr(self.llm, "unload", None)
        if callable(unload):
            unload()
        close = getattr(self.llm, "close", None)
        if callable(close):
            close()


def detect_language(text: str) -> Literal["vi", "en"]:
    vietnamese = r"[ăâđêôơưĂÂĐÊÔƠƯ]|[àáạảãằắặẳẵầấậẩẫèéẹẻẽềếệểễìíịỉĩòóọỏõồốộổỗờớợởỡùúụủũừứựửữỳýỵỷỹ]"
    if re.search(vietnamese, text):
        return "vi"
    return "en"
