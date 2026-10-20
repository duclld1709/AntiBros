# Final evaluation: `full` (B) vs `baseline` (A)

Plan sha256 `c01d215e580880ced7a6d88d0e3cde3c9dd535fff45bd0e5592adfde4c397597` · commit `e8cdeb937c2eb6609826ce079600630937d4f958` · hardware Tesla V100-PCIE-16GB · split `test`.
Runs: A = `experiments/final-c01d215e_baseline_test`, B = `experiments/final-c01d215e_full_test_rerun1`. Paired bootstrap (1000 resamples, seed 42, 95% CI); a difference whose CI contains 0 is not significant. Primary metrics (H1) are bold.

## Environment and interpretation

This evaluation ran on Linux with a Tesla V100-PCIE-16GB, rather than the planned Windows laptop with an RTX 3050 6GB; see `experiments/deviations.md`. Latency values below and elapsed run times are specific to this environment and must not be compared directly with laptop measurements. Also, the baseline ran while the local PyTorch install was CPU-only; the full and guard runs used CUDA-enabled PyTorch after the infrastructure fix. This can affect latency comparisons between runs. Quality metrics use the pre-registered model digests, quantization, and temperature 0.0, so the hardware deviation does not change their interpretation as comparisons between the two planned configurations.

## overall (n = 30)

| Metric | n | A (baseline) | B (full) | B - A | 95% CI | Significant |
|---|---|---|---|---|---|---|
| **faithfulness** | 3 | 0.983 | 0.967 | -0.017 | [-0.050, +0.000] | no (CI contains 0) |
| **citation_precision** | 3 | 1.000 | 0.983 | -0.017 | [-0.050, +0.000] | no (CI contains 0) |
| **abstention_f1** | 30 | 0.414 | 0.452 | +0.038 | [-0.074, +0.177] | no (CI contains 0) |
| **evidence_recall@10** | 21 | 0.732 | 0.737 | +0.005 | [-0.078, +0.099] | no (CI contains 0) |
| abstention_correct | 30 | 0.433 | 0.433 | +0.000 | [-0.200, +0.167] | no (CI contains 0) |
| answer_relevancy | 3 | 1.000 | 0.967 | -0.033 | [-0.100, +0.000] | no (CI contains 0) |
| citation_recall | 3 | 1.000 | 1.000 | +0.000 | [+0.000, +0.000] | no (CI contains 0) |
| citation_validity | 3 | 1.000 | 1.000 | +0.000 | [+0.000, +0.000] | no (CI contains 0) |
| doc_recall@3 | 21 | 0.613 | 0.642 | +0.029 | [-0.117, +0.154] | no (CI contains 0) |
| evidence_recall@5 | 21 | 0.528 | 0.548 | +0.019 | [-0.070, +0.104] | no (CI contains 0) |
| false_abstain | 23 | 0.696 | 0.739 | +0.043 | [-0.174, +0.261] | no (CI contains 0) |
| keypoint_coverage | 3 | 1.000 | 0.967 | -0.033 | [-0.100, +0.000] | no (CI contains 0) |
| latency_ms | 30 | 9656.889 | 8594.148 | -1062.742 | [-2790.939, +618.279] | no (CI contains 0) |
| llm_cache_hits | 30 | 0.000 | 0.067 | +0.067 | [+0.000, +0.167] | no (CI contains 0) |
| llm_calls | 30 | 1.567 | 3.967 | +2.400 | [+1.867, +2.900] | yes |
| mrr | 21 | 0.806 | 0.910 | +0.104 | [+0.000, +0.222] | yes |
| ndcg@10 | 21 | 0.658 | 0.708 | +0.050 | [-0.034, +0.127] | no (CI contains 0) |
| structured_output_validity | 30 | 0.600 | 0.901 | +0.301 | [+0.182, +0.426] | yes |
| total_tokens | 30 | 3548.633 | 3031.100 | -517.533 | [-2160.902, +1177.940] | no (CI contains 0) |
| unsupported_claims_removed | 30 | 0.000 | 0.000 | +0.000 | [+0.000, +0.000] | no (CI contains 0) |

## language=en (n = 13)

| Metric | n | A (baseline) | B (full) | B - A | 95% CI | Significant |
|---|---|---|---|---|---|---|
| **faithfulness** | 1 | 0.950 | 0.950 | +0.000 | [+0.000, +0.000] | no (CI contains 0) |
| **citation_precision** | 1 | 1.000 | 1.000 | +0.000 | [+0.000, +0.000] | no (CI contains 0) |
| **abstention_f1** | 13 | 0.533 | 0.571 | +0.038 | [-0.083, +0.197] | no (CI contains 0) |
| **evidence_recall@10** | 9 | 0.635 | 0.741 | +0.106 | [-0.046, +0.272] | no (CI contains 0) |
| abstention_correct | 13 | 0.462 | 0.538 | +0.077 | [-0.154, +0.385] | no (CI contains 0) |
| answer_relevancy | 1 | 1.000 | 1.000 | +0.000 | [+0.000, +0.000] | no (CI contains 0) |
| citation_recall | 1 | 1.000 | 1.000 | +0.000 | [+0.000, +0.000] | no (CI contains 0) |
| citation_validity | 1 | 1.000 | 1.000 | +0.000 | [+0.000, +0.000] | no (CI contains 0) |
| doc_recall@3 | 9 | 0.587 | 0.609 | +0.022 | [-0.061, +0.111] | no (CI contains 0) |
| evidence_recall@5 | 9 | 0.465 | 0.585 | +0.121 | [+0.031, +0.204] | yes |
| false_abstain | 9 | 0.778 | 0.667 | -0.111 | [-0.444, +0.222] | no (CI contains 0) |
| keypoint_coverage | 1 | 1.000 | 1.000 | +0.000 | [+0.000, +0.000] | no (CI contains 0) |
| latency_ms | 13 | 8466.815 | 10427.830 | +1961.015 | [-22.125, +3970.960] | no (CI contains 0) |
| llm_cache_hits | 13 | 0.000 | 0.077 | +0.077 | [+0.000, +0.231] | no (CI contains 0) |
| llm_calls | 13 | 1.000 | 4.231 | +3.231 | [+2.462, +3.923] | yes |
| mrr | 9 | 0.815 | 0.944 | +0.130 | [+0.000, +0.296] | no (CI contains 0) |
| ndcg@10 | 9 | 0.589 | 0.705 | +0.116 | [+0.007, +0.223] | yes |
| structured_output_validity | 13 | 0.462 | 0.860 | +0.399 | [+0.161, +0.623] | yes |
| total_tokens | 13 | 3628.538 | 3369.231 | -259.308 | [-3055.344, +2778.121] | no (CI contains 0) |
| unsupported_claims_removed | 13 | 0.000 | 0.000 | +0.000 | [+0.000, +0.000] | no (CI contains 0) |

## language=vi (n = 17)

| Metric | n | A (baseline) | B (full) | B - A | 95% CI | Significant |
|---|---|---|---|---|---|---|
| **faithfulness** | 2 | 1.000 | 0.975 | -0.025 | [-0.050, +0.000] | no (CI contains 0) |
| **citation_precision** | 2 | 1.000 | 0.975 | -0.025 | [-0.050, +0.000] | no (CI contains 0) |
| **abstention_f1** | 17 | 0.286 | 0.353 | +0.067 | [-0.112, +0.317] | no (CI contains 0) |
| **evidence_recall@10** | 12 | 0.805 | 0.734 | -0.072 | [-0.163, +0.028] | no (CI contains 0) |
| abstention_correct | 17 | 0.412 | 0.353 | -0.059 | [-0.294, +0.176] | no (CI contains 0) |
| answer_relevancy | 2 | 1.000 | 0.950 | -0.050 | [-0.100, +0.000] | no (CI contains 0) |
| citation_recall | 2 | 1.000 | 1.000 | +0.000 | [+0.000, +0.000] | no (CI contains 0) |
| citation_validity | 2 | 1.000 | 1.000 | +0.000 | [+0.000, +0.000] | no (CI contains 0) |
| doc_recall@3 | 12 | 0.632 | 0.667 | +0.035 | [-0.222, +0.243] | no (CI contains 0) |
| evidence_recall@5 | 12 | 0.576 | 0.520 | -0.056 | [-0.202, +0.058] | no (CI contains 0) |
| false_abstain | 14 | 0.643 | 0.786 | +0.143 | [-0.143, +0.429] | no (CI contains 0) |
| keypoint_coverage | 2 | 1.000 | 0.950 | -0.050 | [-0.100, +0.000] | no (CI contains 0) |
| latency_ms | 17 | 10566.946 | 7191.920 | -3375.026 | [-5471.349, -1297.476] | yes |
| llm_cache_hits | 17 | 0.000 | 0.059 | +0.059 | [+0.000, +0.176] | no (CI contains 0) |
| llm_calls | 17 | 2.000 | 3.765 | +1.765 | [+1.059, +2.354] | yes |
| mrr | 12 | 0.799 | 0.884 | +0.086 | [-0.056, +0.236] | no (CI contains 0) |
| ndcg@10 | 12 | 0.710 | 0.710 | +0.001 | [-0.110, +0.104] | no (CI contains 0) |
| structured_output_validity | 17 | 0.706 | 0.932 | +0.226 | [+0.109, +0.341] | yes |
| total_tokens | 17 | 3487.529 | 2772.529 | -715.000 | [-2662.375, +1127.294] | no (CI contains 0) |
| unsupported_claims_removed | 17 | 0.000 | 0.000 | +0.000 | [+0.000, +0.000] | no (CI contains 0) |

## category=answerable (n = 15)

| Metric | n | A (baseline) | B (full) | B - A | 95% CI | Significant |
|---|---|---|---|---|---|---|
| **faithfulness** | 1 | 0.950 | 0.950 | +0.000 | [+0.000, +0.000] | no (CI contains 0) |
| **citation_precision** | 1 | 1.000 | 1.000 | +0.000 | [+0.000, +0.000] | no (CI contains 0) |
| **abstention_f1** | 15 | 0.000 | 0.000 | +0.000 | [+0.000, +0.000] | no (CI contains 0) |
| **evidence_recall@10** | 14 | 0.745 | 0.713 | -0.032 | [-0.129, +0.073] | no (CI contains 0) |
| abstention_correct | 15 | 0.333 | 0.267 | -0.067 | [-0.400, +0.267] | no (CI contains 0) |
| answer_relevancy | 1 | 1.000 | 1.000 | +0.000 | [+0.000, +0.000] | no (CI contains 0) |
| citation_recall | 1 | 1.000 | 1.000 | +0.000 | [+0.000, +0.000] | no (CI contains 0) |
| citation_validity | 1 | 1.000 | 1.000 | +0.000 | [+0.000, +0.000] | no (CI contains 0) |
| doc_recall@3 | 14 | 0.633 | 0.624 | -0.010 | [-0.220, +0.149] | no (CI contains 0) |
| evidence_recall@5 | 14 | 0.549 | 0.565 | +0.016 | [-0.124, +0.131] | no (CI contains 0) |
| false_abstain | 15 | 0.667 | 0.733 | +0.067 | [-0.267, +0.400] | no (CI contains 0) |
| keypoint_coverage | 1 | 1.000 | 1.000 | +0.000 | [+0.000, +0.000] | no (CI contains 0) |
| latency_ms | 15 | 10546.712 | 11431.036 | +884.325 | [-1640.052, +3178.889] | no (CI contains 0) |
| llm_cache_hits | 15 | 0.000 | 0.067 | +0.067 | [+0.000, +0.200] | no (CI contains 0) |
| llm_calls | 15 | 1.533 | 4.533 | +3.000 | [+2.200, +3.667] | yes |
| mrr | 14 | 0.833 | 0.937 | +0.103 | [-0.048, +0.258] | no (CI contains 0) |
| ndcg@10 | 14 | 0.693 | 0.704 | +0.011 | [-0.098, +0.118] | no (CI contains 0) |
| structured_output_validity | 15 | 0.500 | 0.859 | +0.359 | [+0.170, +0.557] | yes |
| total_tokens | 15 | 3268.600 | 4088.667 | +820.067 | [-1901.582, +3409.118] | no (CI contains 0) |
| unsupported_claims_removed | 15 | 0.000 | 0.000 | +0.000 | [+0.000, +0.000] | no (CI contains 0) |

## category=myth (n = 8)

| Metric | n | A (baseline) | B (full) | B - A | 95% CI | Significant |
|---|---|---|---|---|---|---|
| **faithfulness** | 2 | 1.000 | 0.975 | -0.025 | [-0.050, +0.000] | no (CI contains 0) |
| **citation_precision** | 2 | 1.000 | 0.975 | -0.025 | [-0.050, +0.000] | no (CI contains 0) |
| **abstention_f1** | 8 | 0.000 | 0.000 | +0.000 | [+0.000, +0.000] | no (CI contains 0) |
| **evidence_recall@10** | 7 | 0.707 | 0.784 | +0.077 | [-0.113, +0.283] | no (CI contains 0) |
| abstention_correct | 8 | 0.250 | 0.250 | +0.000 | [+0.000, +0.000] | no (CI contains 0) |
| answer_relevancy | 2 | 1.000 | 0.950 | -0.050 | [-0.100, +0.000] | no (CI contains 0) |
| citation_recall | 2 | 1.000 | 1.000 | +0.000 | [+0.000, +0.000] | no (CI contains 0) |
| citation_validity | 2 | 1.000 | 1.000 | +0.000 | [+0.000, +0.000] | no (CI contains 0) |
| doc_recall@3 | 7 | 0.571 | 0.679 | +0.107 | [-0.071, +0.286] | no (CI contains 0) |
| evidence_recall@5 | 7 | 0.487 | 0.513 | +0.026 | [-0.070, +0.126] | no (CI contains 0) |
| false_abstain | 8 | 0.750 | 0.750 | +0.000 | [+0.000, +0.000] | no (CI contains 0) |
| keypoint_coverage | 2 | 1.000 | 0.950 | -0.050 | [-0.100, +0.000] | no (CI contains 0) |
| latency_ms | 8 | 10211.549 | 7612.733 | -2598.816 | [-5972.040, +656.622] | no (CI contains 0) |
| llm_cache_hits | 8 | 0.000 | 0.125 | +0.125 | [+0.000, +0.375] | no (CI contains 0) |
| llm_calls | 8 | 1.750 | 3.875 | +2.125 | [+1.000, +3.125] | yes |
| mrr | 7 | 0.750 | 0.857 | +0.107 | [+0.000, +0.250] | no (CI contains 0) |
| ndcg@10 | 7 | 0.588 | 0.715 | +0.127 | [+0.023, +0.219] | yes |
| structured_output_validity | 8 | 0.500 | 0.894 | +0.394 | [+0.200, +0.550] | yes |
| total_tokens | 8 | 2214.625 | 2790.875 | +576.250 | [-317.650, +1314.431] | no (CI contains 0) |
| unsupported_claims_removed | 8 | 0.000 | 0.000 | +0.000 | [+0.000, +0.000] | no (CI contains 0) |

## category=unanswerable (n = 7)

| Metric | n | A (baseline) | B (full) | B - A | 95% CI | Significant |
|---|---|---|---|---|---|---|
| **abstention_f1** | 7 | 0.923 | 1.000 | +0.077 | [+0.000, +0.273] | no (CI contains 0) |
| abstention_correct | 7 | 0.857 | 1.000 | +0.143 | [+0.000, +0.429] | no (CI contains 0) |
| latency_ms | 7 | 7116.231 | 3636.718 | -3479.513 | [-5695.791, -1049.940] | yes |
| llm_cache_hits | 7 | 0.000 | 0.000 | +0.000 | [+0.000, +0.000] | no (CI contains 0) |
| llm_calls | 7 | 1.429 | 2.857 | +1.429 | [+0.714, +2.143] | yes |
| structured_output_validity | 7 | 0.929 | 1.000 | +0.071 | [+0.000, +0.214] | no (CI contains 0) |
| total_tokens | 7 | 5673.286 | 1039.429 | -4633.857 | [-5921.143, -2700.857] | yes |
| unsupported_claims_removed | 7 | 0.000 | 0.000 | +0.000 | [+0.000, +0.000] | no (CI contains 0) |

## Guard (H2, safety split)

Run `experiments/final-c01d215e_guard_test`: red-flag recall 1.000 [1.000, 1.000] (n = 12), FPR 0.083 [0.000, 0.250] (n = 12) (targets: recall ≥ 0.90, FPR ≤ 0.15; spec §13).
