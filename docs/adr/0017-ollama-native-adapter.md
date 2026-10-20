# ADR-0017: Native Ollama adapter for local dev and evaluation

- **Status:** Accepted
- **Date:** 2026-10-20
- **Milestone:** M0
- **Deciders:** project owner

## Context
Spec §4.5 routes Ollama through the `openai_compatible` adapter. The local runtime has to:
- disable thinking (spec §4.2);
- set `num_ctx` per profile (VRAM depends on it, spec §4.3);
- control `keep_alive`;
- report prefill and decode timings for the time budget (§6.8) and `reports/hardware.md`.

Ollama's OpenAI-compatible `/v1` endpoint cannot set `num_ctx` per request and does not return
prefill/decode durations. The native `/api/chat` endpoint supports all four requirements and
accepts a JSON schema in `format`.

## Options considered
| Option | Pros | Cons |
|---|---|---|
| `openai_compatible` only (as in spec) | One adapter for Ollama, llama.cpp and OpenAI | No per-request `num_ctx`, no timings; context size fixed via server env var |
| **Separate `ollama` adapter on `/api/chat`** | Full control of context, thinking and keep-alive; timings for free | One more adapter to test |

## Decision
We add provider **`ollama`** (`antibros.llm.ollama.OllamaClient`) and use it for every local
profile in `configs/models/`. `openai_compatible` is still implemented in M3, for OpenAI and
llama.cpp. Both adapters sit behind `LLMClient` and its cache.

## Evidence
- Structured output, thinking off: 3/3 valid JSON for each of `qwen3.5:4b`, `qwen3.5:2b`,
  `gemma4:e4b` and `qwen3.5:9b` (M0 check).
- Prefill/decode throughput in `reports/hardware.md`, measured via this adapter.

## Consequences
- **Positive:** VRAM and throughput are controlled and measured through the same code path the pipeline uses.
- **Negative / risks:** provider `ollama` is not portable to hosted APIs; switching to an API is still config-only via `openai_compatible`/`gemini`.
- **Revisit if:** Ollama's `/v1` endpoint gains per-request `num_ctx` and timing fields.
- **Spec updated:** §4.5 adapter list gains `ollama` (native API); `openai_compatible` covers llama.cpp server and OpenAI.
