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
**Accepted: Combination**, with all thresholds in config. Choose the operating point on dev by reporting both abstention F1 and the false-abstain rate on answerable questions.

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
Grade sweep: [`reports/_tune_grade.md`](../../reports/_tune_grade.md), with completed runs in
`experiments/20261020-2119_tune_grade_min_rerank_score_0p02/`,
`experiments/20261020-2127_tune_grade_min_rerank_score_0p05/`,
`experiments/20261020-2134_tune_grade_min_rerank_score_0p1/`,
`experiments/20261020-2140_tune_grade_min_rerank_score_0p2/`, and
`experiments/20261020-2147_tune_grade_min_rerank_score_0p3/`. All completed grade runs used
the same `laptop_light` profile (pipeline `qwen3.5:2b`, digest
`0689d44085e06d165161a8a9a1731344278cfb5aade63a3c3dbdb48ab54b130a`; judge `gemma4:e4b`,
digest `dc35e8d9c6061baa6f0fa870975ab6932e2542b579b13ea0f199fa4bb7300c9c`). Values 0.02,
0.05, 0.1, and 0.2 tied at abstention F1 0.500 and false-abstain rate 0.667; 0.3 scored F1
0.471 and false-abstain rate 0.750. Per the preregistered tie-break, `grade.min_rerank_score`
is **0.1**, the tied value closest to the base default. The aborted attempts remain preserved
and are not counted as completed runs.

The verify grid was rerun on V100 with the same `laptop_light` model digests and CUDA
reranking for all values. The complete comparable grid gives F1 0.471 at 0.25, 0.500 at 0.5,
and 0.533 at 0.75; `verify.max_removed_ratio` is **0.75**. The first run took 111.44 s and
the two cache replays took 5.44 s and 5.45 s; all were below the 600 s budget. See
[`reports/_tune_verify.md`](../../reports/_tune_verify.md) and run directories
`experiments/20261020-1531_tune_verify_max_removed_ratio_0p25/`,
`experiments/20261020-1532_tune_verify_max_removed_ratio_0p5/`, and
`experiments/20261020-1533_tune_verify_max_removed_ratio_0p75/`. Previous partial/completed
RTX 3050 runs were preserved and excluded from this V100 grid comparison.

## Consequences
- **Positive:** explicit, tunable trade-off between helpfulness and safety.
- **Negative:** more moving parts; thresholds can overfit dev.
- **Revisit if:** the false-abstain rate on answerable questions exceeds ~15%.
