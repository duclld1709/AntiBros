# ADR-0021: M4 pipeline composition, config inheritance and fallbacks

- **Status:** Accepted
- **Date:** 2026-10-20
- **Milestone:** M4

## Context
M4 adds five optional steps (rerank, query rewrite, grade_context, verify_citations, guards)
that must be switchable by config alone for leave-one-out and guard ablations (spec §5.2,
§6.4). Several details are not fixed by the spec: what a disabled step means for the graph,
how ablation configs avoid copy-paste drift, what happens when the LLM returns invalid JSON
at each step, and how a golden question answered with a safety redirect is scored.

## Options considered
| Option | Pros | Cons |
|---|---|---|
| Fixed graph, disabled nodes pass through | One graph shape | Pass-through nodes still appear in traces and timings; easy to leave hidden work in them |
| **Graph built from config; disabled steps are absent** | The trace shows exactly what ran; LOO configs differ in one line | Graph construction has a few branches |
| Full copies of every ablation config | No new mechanism | 15+ near-identical files drift apart silently |
| **`extends:` inheritance with deep merge** | Each ablation file shows only its one change | Hash of the child file alone does not cover its parents |

## Decision
1. **Graph shape follows config.** `RagPipeline._build_graph` adds a node only when its step
   is enabled. Defaults are all off, so `baseline.yaml` is unchanged in meaning.
2. **Config inheritance.** A config may say `extends: <relative path>`; mappings are merged
   recursively. Run snapshots store the resolved config. The final plan pins both the
   sha256 of the resolved config and the byte sha256 of every file in the chain.
3. **Context size.** `pipeline.context_k` is the number of passages handed to generation
   (top-n after reranking). `full` uses 5; `loo_no_rerank` keeps 5 so only the reordering is
   removed; `baseline` keeps all top-20 as in M3.
4. **Grade.** Sufficient = (max reranker score ≥ `min_rerank_score`, only when reranking is
   on, since RRF scores are rank-based) AND (LLM says sufficient). The LLM is skipped when the
   score already fails. A retry asks `query_rewrite` for formulations different from the
   queries already tried, even in `loo_no_rewrite`.
5. **Fallbacks on invalid structured output** (counted in `structured_output_validity`):
   translation/rewrite → retrieve with the original question; grade → insufficient;
   generate → abstain; verify → abstain; LLM input guard → `safe` (marked
   `layer=llm_invalid`). Transport errors are not caught: the eval run records
   `status=error`.
6. **Output guard** is rule-only (drug/PED names + a dose unit) and attaches a disclaimer
   in a separate `disclaimer` field, so judge scores of the answer text are unaffected.
7. **Scoring.** On golden, `safety_redirect` counts as abstaining (a refusal). Evidence
   Recall@k of an e2e run is computed from the last ranking recorded in the prediction's
   trace (the reranked list when reranking is on), so H1 can use it.

## Evidence
Unit tests: `tests/unit/test_m4_pipeline.py`, `tests/unit/test_m4_tooling.py`. Dev numbers
for each step are TBD (M4 runs on the laptop; see `docs/runbooks/m4_dev_runs.md`).

## Consequences
- **Positive:** ablations are one-line diffs; traces show exactly which steps ran; outages
  cannot masquerade as abstentions.
- **Negative / risks:** conservative fallbacks (grade/verify → abstain) can raise the
  false-abstain rate; it is measured per run. Prompt changes for injection defence also
  apply to `baseline`, so M3 numbers are not reproduced bit-for-bit by the M4 code.
- **Revisit if:** structured-output validity on dev falls below ~90% for any step.
