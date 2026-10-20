# Threshold sweep: `verify_max_removed_ratio` (dev, n=16)

| Value | Abstention F1 | False-abstain rate | Run |
|---|---|---|---|
| 0.25 | 0.471 [0.143, 0.706] | 0.750 [0.500, 1.000] | `experiments/20261020-1531_tune_verify_max_removed_ratio_0p25` |
| 0.5 | 0.500 [0.154, 0.750] | 0.667 [0.333, 0.917] | `experiments/20261020-1532_tune_verify_max_removed_ratio_0p5` |
| 0.75 | 0.533 [0.167, 0.800] | 0.583 [0.333, 0.833] | `experiments/20261020-1533_tune_verify_max_removed_ratio_0p75` |

## Reproducibility and decision

The complete grid was rerun on the Tesla V100-PCIE-16GB with PyTorch 2.7.1+cu126 and
`rerank.device: cuda`. All runs used `laptop_light`: pipeline `qwen3.5:2b` (digest
`0689d44085e06d165161a8a9a1731344278cfb5aade63a3c3dbdb48ab54b130a`) and judge `gemma4:e4b`
(digest `dc35e8d9c6061baa6f0fa870975ab6932e2542b579b13ea0f199fa4bb7300c9c`). The 0.25 run
took 111.44 s; cache replays for 0.5 and 0.75 took 5.44 s and 5.45 s. Every run completed
within the 600 s budget. Since 0.75 had the highest abstention F1, it is selected under
ADR-0007 and set as `pipeline.verify.max_removed_ratio` in `configs/full.yaml`.

The earlier RTX 3050 CPU-reranker 0.25/0.5 runs and interrupted 0.75 run are retained. They
were not mixed into this comparison because the full grid is now available under one model
profile, GPU, and reranker device configuration.
