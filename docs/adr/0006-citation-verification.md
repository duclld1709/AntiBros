# ADR-0006: Post-hoc citation verification

- **Status:** Proposed
- **Date:** TBD
- **Milestone:** M4

## Context
Even with strict prompts, LLMs can attach a citation to a claim the chunk does not support. Citation correctness is the core promise of AntiBros.

## Options considered
| Option | Pros | Cons |
|---|---|---|
| Trust the prompt only | Zero extra latency/cost | No guarantee; unsupported claims slip through |
| LLM-as-judge per claim | Handles paraphrase and nuance; easy to implement | Extra cost/latency; self-evaluation bias if the same model is used |
| NLI cross-encoder (entailment) | Cheap; local; deterministic | Weaker on long scientific chunks and numbers; another model competing for 6GB VRAM |

## Decision
Proposed: **verification by the pipeline LLM itself**, with all claims checked in one batched call.

Rationale:
- On a 6GB GPU only one LLM can stay resident next to the embedder and reranker (see ADR-0012).
- Production will use a single API provider.

Policy: drop unsupported claims; abstain when the share of dropped claims exceeds a configurable threshold (tuned on dev within a pre-declared grid).

The independent judge (a different model family) measures citation precision/recall during evaluation, so the self-check bias is visible rather than hidden.

## Evidence
TBD. Leave-one-out `loo_no_verify` vs `full` on dev: citation precision/recall, faithfulness, added latency p95. Final effect on test via the pre-registered `full` vs `baseline` comparison.

## Consequences
- **Positive:** measurable improvement on citation quality without a second resident model.
- **Negative:** self-evaluation bias (the model checks its own claims); one extra LLM call per answer, which counts against the evaluation time budget.
- **Revisit if:** in production a cheaper, separate verifier model or an NLI model reaches similar citation precision.
