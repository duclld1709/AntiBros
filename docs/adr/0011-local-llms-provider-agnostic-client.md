# ADR-0011: Local LLMs for development, provider-agnostic LLM client

- **Status:** Accepted
- **Date:** 2026-10-20
- **Milestone:** M0, M3

## Context
Production is expected to use a hosted API (OpenAI or Gemini), but the project must avoid API costs during development and evaluation. The only hardware is a laptop with an RTX 3050 (6GB) (see ADR-0013). Hosting and production are out of scope for now, but the code must be able to switch providers later.

## Options considered
| Option | Pros | Cons |
|---|---|---|
| API for everything | Best model quality; simplest | Ongoing cost; judge models can be deprecated or silently updated |
| Local only, provider SDK calls spread across the code | No cost | Hard to switch to an API later |
| Local for dev and eval behind one `LLMClient` interface; API adapters tested offline | No cost; switch by config; frozen local judge is reproducible | Small local models are weaker; adapters for paid APIs are not exercised online |

## Decision
- One interface, `LLMClient`, with adapters `openai_compatible` (Ollama / llama.cpp / OpenAI), `gemini` and `fake`. The provider is selected in config.
- Development and evaluation use Ollama with small GGUF models that fit in 6GB.
- **Online API tests are out of scope.** Adapters are verified with mocked HTTP and recorded fixtures, covering payload mapping for structured output, retries, timeouts, usage parsing and cost calculation. Tests marked `integration` are skipped by default.
- All calls go through a disk cache keyed by provider, model, digest, prompt hash and parameters. Evaluation libraries such as RAGAS are wrapped so they also go through this client.

## Evidence
Adapter test suite in `tests/adapters/`; fingerprints record provider, model and digest.

## Consequences
- **Positive:** zero API cost; reproducible judge (fixed GGUF digest); provider switch without code changes.
- **Negative:** reported quality reflects small local models; API adapters may need small fixes the first time they run online.
- **Revisit if:** the project moves to production. Then run a small online smoke test and re-run the evaluation with the production model (new fingerprint).
