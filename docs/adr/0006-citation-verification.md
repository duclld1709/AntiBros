# ADR-0006: Post-hoc citation verification

- **Status:** Accepted
- **Date:** 2026-10-20
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
**Verification by the pipeline LLM itself**, with all claims checked in one batched call.

Rationale:
- On a 6GB GPU only one LLM can stay resident next to the embedder and reranker (see ADR-0012).
- Production will use a single API provider.

Policy: drop unsupported claims; abstain when the share of dropped claims exceeds a configurable threshold (tuned on dev within a pre-declared grid).

Implementation (M4): `verify_citations` sends every claim with only the passages it cites
(`prompts/verify.md`) and expects one verdict per claim. Claims without a "supported"
verdict are dropped; the answer abstains when no claim survives or the dropped share exceeds
`pipeline.verify.max_removed_ratio` (grid in `configs/full.yaml` → `tuning`). Invalid JSON
from the verifier means "cannot verify" and also abstains (ADR-0021). Citations are
renumbered after the drop.

The independent judge (a different model family) measures citation precision/recall during evaluation, so the self-check bias is visible rather than hidden.

## Evidence
The completed E2E leave-one-out results are in
[`reports/m4_e2e_loo.md`](../../reports/m4_e2e_loo.md), comparing
[`experiments/20261020-1533_m4_full/`](../../experiments/20261020-1533_m4_full/) with
[`experiments/20261020-1541_m4_loo_no_verify/`](../../experiments/20261020-1541_m4_loo_no_verify/).
Both runs used the `laptop` profile (pipeline `qwen3.5:4b`, judge `gemma4:e4b`) and 16 common
golden dev items. Both had citation precision/recall 1.000 and faithfulness 0.993; the paired
95% intervals showed no significant difference on those metrics. The full run marked 0.188 of
claims as unsupported and removed them, versus 0.000 with verification disabled; this paired
difference was also inconclusive at n=16 (95% CI [-0.500, 0.000]). Predict plus judge took
165.67 s for full and remained within the 600 s budget. Latency comparisons were cache-heavy
and are not treated as a clean estimate of verification's incremental cost. These small dev
results support retaining the safety check while flagging that its measured quality effect is
not statistically established. The final effect will be measured by the pre-registered `full`
vs `baseline` test comparison.

## Consequences
- **Positive:** measurable improvement on citation quality without a second resident model.
- **Negative:** self-evaluation bias (the model checks its own claims); one extra LLM call per answer, which counts against the evaluation time budget.
- **Revisit if:** in production a cheaper, separate verifier model or an NLI model reaches similar citation precision.
