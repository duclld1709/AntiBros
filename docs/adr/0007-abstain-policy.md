# ADR-0007: Abstain policy and thresholds

- **Status:** Proposed
- **Date:** TBD
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

## Evidence
TBD — abstention F1, false-abstain rate, latency p95 for each threshold setting.

## Consequences
- **Positive:** explicit, tunable trade-off between helpfulness and safety.
- **Negative:** more moving parts; thresholds can overfit dev.
- **Revisit if:** the false-abstain rate on answerable questions exceeds ~15%.
