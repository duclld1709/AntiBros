# Threshold sweep: `grade_min_rerank_score` (dev, n=16)

| Value | Abstention F1 | False-abstain rate | Run |
|---|---|---|---|
| 0.02 | 0.500 [0.154, 0.750] | 0.667 [0.333, 0.917] | `experiments/20261020-2119_tune_grade_min_rerank_score_0p02` |
| 0.05 | 0.500 [0.154, 0.750] | 0.667 [0.333, 0.917] | `experiments/20261020-2127_tune_grade_min_rerank_score_0p05` |
| 0.1 | 0.500 [0.154, 0.750] | 0.667 [0.333, 0.917] | `experiments/20261020-2134_tune_grade_min_rerank_score_0p1` |
| 0.2 | 0.500 [0.154, 0.750] | 0.667 [0.333, 0.917] | `experiments/20261020-2140_tune_grade_min_rerank_score_0p2` |
| 0.3 | 0.471 [0.143, 0.727] | 0.750 [0.500, 0.917] | `experiments/20261020-2147_tune_grade_min_rerank_score_0p3` |

All five completed runs use `laptop_light`: pipeline `qwen3.5:2b` (digest
`0689d44085e06d165161a8a9a1731344278cfb5aade63a3c3dbdb48ab54b130a`) and judge `gemma4:e4b`
(digest `dc35e8d9c6061baa6f0fa870975ab6932e2542b579b13ea0f199fa4bb7300c9c`). Values 0.02,
0.05, 0.1, and 0.2 tie; 0.1 is selected by the ADR-0007 tie-break (closest to the base
default). The complete verify grid was rerun on V100 with this same model profile and CUDA
reranking; it selected 0.75 under ADR-0007. See [`_tune_verify.md`](_tune_verify.md).
