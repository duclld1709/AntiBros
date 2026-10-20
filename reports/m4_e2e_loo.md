# M4 E2E leave-one-out comparisons (golden dev)

All comparisons use paired bootstrap with 16 common dev items. A is `m4_full`; B is the leave-one-out variant. A difference is marked significant only when the 95% CI excludes zero. Full timing includes predict + judge: 165.67 s. Every run stayed within the 600 s budget. See each run's `timing.json` and artifacts under `experiments/`.


## Without reranking

# Paired comparison: `20261020-1533_m4_full` (A) vs `20261020-1536_m4_loo_no_rerank` (B)

Common items: 16. Soft fingerprint differences: config_sha256.

| Metric | A | B | B - A | 95% CI | Significant |
|---|---|---|---|---|---|
| abstention_correct | 0.688 | 0.625 | -0.062 | [-0.188, +0.000] | no (CI contains 0) |
| answer_relevancy | 1.000 | 1.000 | +0.000 | [+0.000, +0.000] | no (CI contains 0) |
| citation_precision | 1.000 | 0.983 | -0.017 | [-0.050, +0.000] | no (CI contains 0) |
| citation_recall | 1.000 | 0.983 | -0.017 | [-0.050, +0.000] | no (CI contains 0) |
| citation_validity | 1.000 | 1.000 | +0.000 | [+0.000, +0.000] | no (CI contains 0) |
| doc_recall@3 | 0.786 | 0.772 | -0.014 | [-0.125, +0.083] | no (CI contains 0) |
| evidence_recall@10 | 0.728 | 0.533 | -0.195 | [-0.330, -0.070] | yes |
| evidence_recall@5 | 0.628 | 0.460 | -0.167 | [-0.312, -0.040] | yes |
| faithfulness | 0.992 | 0.950 | -0.042 | [-0.092, -0.008] | yes |
| false_abstain | 0.417 | 0.500 | +0.083 | [+0.000, +0.250] | no (CI contains 0) |
| keypoint_coverage | 1.000 | 0.950 | -0.050 | [-0.083, -0.017] | yes |
| latency_ms | 9190.965 | 7292.760 | -1898.205 | [-2848.137, -975.564] | yes |
| llm_cache_hits | 0.125 | 2.500 | +2.375 | [+2.125, +2.625] | yes |
| llm_calls | 4.312 | 5.062 | +0.750 | [+0.250, +1.250] | yes |
| mrr | 0.917 | 0.746 | -0.170 | [-0.318, -0.042] | yes |
| ndcg@10 | 0.748 | 0.549 | -0.199 | [-0.295, -0.105] | yes |
| structured_output_validity | 0.972 | 0.954 | -0.018 | [-0.060, +0.025] | no (CI contains 0) |
| total_tokens | 4628.250 | 5610.688 | +982.438 | [-122.508, +2093.228] | no (CI contains 0) |
| unsupported_claims_removed | 0.188 | 0.188 | +0.000 | [-0.188, +0.188] | no (CI contains 0) |


## Without query rewriting

# Paired comparison: `20261020-1533_m4_full` (A) vs `20261020-1538_m4_loo_no_rewrite` (B)

Common items: 16. Soft fingerprint differences: config_sha256.

| Metric | A | B | B - A | 95% CI | Significant |
|---|---|---|---|---|---|
| abstention_correct | 0.688 | 0.688 | +0.000 | [+0.000, +0.000] | no (CI contains 0) |
| answer_relevancy | 1.000 | 1.000 | +0.000 | [+0.000, +0.000] | no (CI contains 0) |
| citation_precision | 1.000 | 1.000 | +0.000 | [+0.000, +0.000] | no (CI contains 0) |
| citation_recall | 1.000 | 1.000 | +0.000 | [+0.000, +0.000] | no (CI contains 0) |
| citation_validity | 1.000 | 1.000 | +0.000 | [+0.000, +0.000] | no (CI contains 0) |
| doc_recall@3 | 0.786 | 0.786 | +0.000 | [+0.000, +0.000] | no (CI contains 0) |
| evidence_recall@10 | 0.728 | 0.719 | -0.009 | [-0.051, +0.032] | no (CI contains 0) |
| evidence_recall@5 | 0.628 | 0.614 | -0.014 | [-0.042, +0.000] | no (CI contains 0) |
| faithfulness | 0.993 | 0.993 | +0.000 | [+0.000, +0.000] | no (CI contains 0) |
| false_abstain | 0.417 | 0.417 | +0.000 | [+0.000, +0.000] | no (CI contains 0) |
| keypoint_coverage | 1.000 | 1.000 | +0.000 | [+0.000, +0.000] | no (CI contains 0) |
| latency_ms | 9190.965 | 4447.736 | -4743.229 | [-7249.794, -2401.433] | yes |
| llm_cache_hits | 0.125 | 1.938 | +1.812 | [+1.250, +2.438] | yes |
| llm_calls | 4.312 | 3.812 | -0.500 | [-0.750, -0.250] | yes |
| mrr | 0.917 | 0.917 | +0.000 | [+0.000, +0.000] | no (CI contains 0) |
| ndcg@10 | 0.748 | 0.739 | -0.009 | [-0.028, +0.004] | no (CI contains 0) |
| structured_output_validity | 0.972 | 0.931 | -0.041 | [-0.103, +0.000] | no (CI contains 0) |
| total_tokens | 4628.250 | 4120.500 | -507.750 | [-1124.709, -114.422] | yes |
| unsupported_claims_removed | 0.188 | 0.188 | +0.000 | [+0.000, +0.000] | no (CI contains 0) |


## Without grading

# Paired comparison: `20261020-1533_m4_full` (A) vs `20261020-1539_m4_loo_no_grade` (B)

Common items: 16. Soft fingerprint differences: config_sha256.

| Metric | A | B | B - A | 95% CI | Significant |
|---|---|---|---|---|---|
| abstention_correct | 0.688 | 0.812 | +0.125 | [+0.000, +0.312] | no (CI contains 0) |
| answer_relevancy | 1.000 | 1.000 | +0.000 | [+0.000, +0.000] | no (CI contains 0) |
| citation_precision | 1.000 | 1.000 | +0.000 | [+0.000, +0.000] | no (CI contains 0) |
| citation_recall | 1.000 | 1.000 | +0.000 | [+0.000, +0.000] | no (CI contains 0) |
| citation_validity | 1.000 | 1.000 | +0.000 | [+0.000, +0.000] | no (CI contains 0) |
| doc_recall@3 | 0.786 | 0.744 | -0.042 | [-0.125, +0.000] | no (CI contains 0) |
| evidence_recall@10 | 0.728 | 0.728 | +0.000 | [+0.000, +0.000] | no (CI contains 0) |
| evidence_recall@5 | 0.628 | 0.607 | -0.021 | [-0.062, +0.000] | no (CI contains 0) |
| faithfulness | 0.993 | 0.993 | +0.000 | [+0.000, +0.000] | no (CI contains 0) |
| false_abstain | 0.417 | 0.250 | -0.167 | [-0.417, +0.000] | no (CI contains 0) |
| keypoint_coverage | 1.000 | 1.000 | +0.000 | [+0.000, +0.000] | no (CI contains 0) |
| latency_ms | 9190.965 | 3517.670 | -5673.295 | [-9070.495, -2514.120] | yes |
| llm_cache_hits | 0.125 | 2.938 | +2.812 | [+2.250, +3.375] | yes |
| llm_calls | 4.312 | 3.562 | -0.750 | [-1.000, -0.500] | yes |
| mrr | 0.917 | 0.903 | -0.014 | [-0.042, +0.000] | no (CI contains 0) |
| ndcg@10 | 0.748 | 0.730 | -0.018 | [-0.055, +0.001] | no (CI contains 0) |
| structured_output_validity | 0.972 | 0.958 | -0.014 | [-0.073, +0.027] | no (CI contains 0) |
| total_tokens | 4628.250 | 4336.000 | -292.250 | [-1370.195, +792.584] | no (CI contains 0) |
| unsupported_claims_removed | 0.188 | 0.188 | +0.000 | [+0.000, +0.000] | no (CI contains 0) |


## Without citation verification

# Paired comparison: `20261020-1533_m4_full` (A) vs `20261020-1541_m4_loo_no_verify` (B)

Common items: 16. Soft fingerprint differences: config_sha256.

| Metric | A | B | B - A | 95% CI | Significant |
|---|---|---|---|---|---|
| abstention_correct | 0.688 | 0.688 | +0.000 | [+0.000, +0.000] | no (CI contains 0) |
| answer_relevancy | 1.000 | 1.000 | +0.000 | [+0.000, +0.000] | no (CI contains 0) |
| citation_precision | 1.000 | 1.000 | +0.000 | [+0.000, +0.000] | no (CI contains 0) |
| citation_recall | 1.000 | 1.000 | +0.000 | [+0.000, +0.000] | no (CI contains 0) |
| citation_validity | 1.000 | 1.000 | +0.000 | [+0.000, +0.000] | no (CI contains 0) |
| doc_recall@3 | 0.786 | 0.786 | +0.000 | [+0.000, +0.000] | no (CI contains 0) |
| evidence_recall@10 | 0.728 | 0.728 | +0.000 | [+0.000, +0.000] | no (CI contains 0) |
| evidence_recall@5 | 0.628 | 0.628 | +0.000 | [+0.000, +0.000] | no (CI contains 0) |
| faithfulness | 0.993 | 0.993 | +0.000 | [+0.000, +0.000] | no (CI contains 0) |
| false_abstain | 0.417 | 0.417 | +0.000 | [+0.000, +0.000] | no (CI contains 0) |
| keypoint_coverage | 1.000 | 1.000 | +0.000 | [+0.000, +0.000] | no (CI contains 0) |
| latency_ms | 9190.965 | 344.370 | -8846.596 | [-10927.587, -6873.962] | yes |
| llm_cache_hits | 0.125 | 3.875 | +3.750 | [+3.500, +4.000] | yes |
| llm_calls | 4.312 | 3.875 | -0.438 | [-0.688, -0.188] | yes |
| mrr | 0.917 | 0.917 | +0.000 | [+0.000, +0.000] | no (CI contains 0) |
| ndcg@10 | 0.748 | 0.748 | +0.000 | [+0.000, +0.000] | no (CI contains 0) |
| structured_output_validity | 0.972 | 0.972 | +0.000 | [+0.000, +0.000] | no (CI contains 0) |
| total_tokens | 4628.250 | 3868.312 | -759.938 | [-1291.694, -293.547] | yes |
| unsupported_claims_removed | 0.188 | 0.000 | -0.188 | [-0.500, +0.000] | no (CI contains 0) |
