# ADR-0007: Abstain policy and thresholds

- **Status:** Accepted
- **Date:** 2026-10-20
- **Milestone:** M4

## Context
The system must say "insufficient evidence" instead of guessing. Abstaining too often makes the product useless; too rarely makes it unsafe and untrustworthy.

## Options considered
| Option | Pros | Cons |
|---|---|---|
| Reranker score threshold | Cheap; deterministic | Scores are poorly calibrated across queries |
| LLM `grade_context` (sufficient / insufficient) | Understands the question | Extra call; can be inconsistent |
| Post-generation: abstain when too many claims fail verification | Uses real support signal | Spends a full generation before deciding |
| Combination (grade → one retry → verify) | Catches both missing context and unsupported answers | Most complex; more latency |

## Decision
Proposed: **Combination**, with all thresholds in config. Choose the operating point on dev by reporting both abstention F1 and the false-abstain rate on answerable questions.

Implementation (M4, ADR-0021): `grade_context` requires the top reranker score to reach
`pipeline.grade.min_rerank_score` (only with reranking on) and the LLM grade to say
"sufficient"; one retry with new query formulations; then abstain. After generation,
`verify_citations` abstains when too many claims are unsupported. Grids are declared in
`configs/full.yaml` (`tuning`) and swept with `antibros eval tune`, which reports abstention
F1 and the false-abstain rate for every value.

Selection rule for each threshold (declared before any tuning run): the grid value with the
highest abstention F1 on golden dev; ties go to the lower false-abstain rate, then to the
value closest to the default in `configs/base.yaml`. Grade is tuned first (with verify at its
default), then verify with the chosen grade threshold.

## Evidence
TBD — abstention F1, false-abstain rate, latency p95 for each threshold setting.

## Consequences
- **Positive:** explicit, tunable trade-off between helpfulness and safety.
- **Negative:** more moving parts; thresholds can overfit dev.
- **Revisit if:** the false-abstain rate on answerable questions exceeds ~15%.
