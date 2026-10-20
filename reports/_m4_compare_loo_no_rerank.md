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
