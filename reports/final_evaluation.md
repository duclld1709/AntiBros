# Final evaluation on the test split (M5)

The pre-registered plan [`configs/final_eval_plan.yaml`](../configs/final_eval_plan.yaml)
(sha256 `c01d215e580880ce…`, locked at the end of M4) ran once with `antibros eval final` on
2026-10-20, at commit `e8cdeb9`. Every number below comes from these directories:

| Run | Directory | Hardware (fingerprint) | Time / budget |
|---|---|---|---|
| `baseline_test` | [`final-c01d215e_baseline_test`](../experiments/final-c01d215e_baseline_test/) | `cpu:x86_64` (CPU-only torch; see deviations) | 311 s / 1,200 s |
| `full_test` (re-run 1) | [`final-c01d215e_full_test_rerun1`](../experiments/final-c01d215e_full_test_rerun1/) | Tesla V100-PCIE-16GB | 275 s / 1,200 s |
| `guard_test` (`rule_llm`) | [`final-c01d215e_guard_test`](../experiments/final-c01d215e_guard_test/) | Tesla V100-PCIE-16GB | 20 s / 300 s |

Paired comparison: [`final-c01d215e_comparison.md`](../experiments/final-c01d215e_comparison.md).
Run log: [`test_runs.log`](../experiments/test_runs.log). Deviations:
[`deviations.md`](../experiments/deviations.md). Models: pipeline `qwen3.5:4b` (digest
`d8b0f5e9760c…`), judge `gemma4:e4b` (`dc35e8d9c606…`), temperature 0. All intervals are 95%
bootstrap CIs (1,000 resamples, seed 42). A difference whose CI contains 0 is reported as not
significant.

## H1: `full` vs `baseline` on golden test

Paired bootstrap of B − A over the items both runs have a value for:

| Metric (H1) | n | baseline | full | full − baseline | 95% CI | Significant |
|---|---|---|---|---|---|---|
| faithfulness | 3 | 0.983 | 0.967 | −0.017 | [−0.050, +0.000] | no |
| citation precision | 3 | 1.000 | 0.983 | −0.017 | [−0.050, +0.000] | no |
| abstention F1 | 30 | 0.414 | 0.452 | +0.038 | [−0.074, +0.177] | no |
| Evidence Recall@10 | 21 | 0.732 | 0.737 | +0.005 | [−0.078, +0.099] | no |
| Evidence Recall@10, Vietnamese | 12 | 0.805 | 0.734 | −0.072 | [−0.163, +0.028] | no |

**H1 is not supported.** None of the four primary metrics differs significantly. The two
judge metrics compare only the 3 questions both pipelines answered, so they cannot show an
effect either way.

Other differences: MRR improved significantly (+0.104 [+0.000, +0.222], n = 21), mostly a
reranking effect. Structured-output validity improved significantly (0.600 → 0.901,
+0.301 [+0.182, +0.426]). `full` makes 2.4 more LLM calls per question
([+1.9, +2.9]). Per-language and per-category tables are in the comparison file.

## H2: guard on safety test

`guard_rule_llm` (selected on dev by the pre-declared rule):
- red-flag recall **1.000** [1.000, 1.000] (12/12);
- FPR **0.083** [0.000, 0.250] (1 of 12 near-red-flag questions).

Both point estimates meet the targets (recall ≥ 0.90, FPR ≤ 0.15). The FPR interval
reaches 0.25, so 12 near-red-flag questions are too few to establish FPR ≤ 0.15 with
confidence. In the golden test run of `full`, the same guard also redirected 2 of 23
answerable/myth questions (see `error_analysis.md`).

## Spec §13 targets (declared before any result)

Unpaired per-run values from each run's `metrics.json` (`full` unless stated):

| Target | Value | Met? |
|---|---|---|
| Evidence Recall@10 ≥ 0.80 | 0.737 [0.642, 0.830] (n = 21) | no (point estimate below) |
| Faithfulness ≥ 0.75 | 0.967 [0.950, 0.983] (n = 6 answered) | yes, on 6 items |
| Citation precision ≥ 0.75 | 0.992 [0.975, 1.000] (n = 6) | yes, on 6 items |
| Abstention F1 ≥ 0.70 | 0.452 [0.214, 0.647] (n = 30) | no |
| Red-flag recall ≥ 0.90 / FPR ≤ 0.15 | 1.000 / 0.083 (n = 12 / 12) | yes on point estimates; FPR CI up to 0.25 |
| Latency p95 ≤ 30 s (laptop) | 15.7 s [12.8, 20.4] on V100 | not measured on the laptop |
| Eval time per command (§6.8) | 311 s, 275 s, 20 s | yes (V100) |

The main failure is over-abstention: false-abstain rate 0.696 (baseline) and 0.739 (full) on
23 answerable/myth items. `error_analysis.md` classifies all 35 wrong items. In summary:
- invalid JSON from the generator (24 of the wrong items across both runs);
- `grade_context` rejecting questions whose evidence was in context (7);
- LLM guard false positives (3);
- one unanswerable question answered by baseline.

## Deviations from the plan

1. **Hardware.** The plan ran on Linux with a Tesla V100, not the Windows RTX 3050 laptop,
   at the owner's request (`deviations.md`). Quality metrics use the pinned model digests at
   temperature 0. Latency and run times are not comparable with laptop numbers.
2. **Baseline ran with CPU-only torch.** `baseline_test` ran before CUDA torch was installed
   (fingerprint `hardware: cpu:x86_64`, `torch 2.14.1+cpu`), so its bge-m3 query embeddings
   (dense and sparse), and nothing else, ran on CPU in FP32. `full_test` and `guard_test` used torch 2.7.1+cu126.
   - The LLMs run in Ollama and were on the GPU in all runs.
   - The two runs use different embedders (bge-m3 vs e5-large) and indexes anyway.
   - Small numeric differences in baseline query vectors relative to a GPU run cannot be
     excluded; they would affect baseline retrieval only marginally.
   - Preflight now refuses this situation (commit `f7fb96f`).
3. **One re-run.** The first `full_test` attempt failed after 8 s, before any prediction,
   because the CUDA reranker found no CUDA torch. It was re-run once with `--rerun` (attempt
   1, `_rerun1`), as ADR-0014 allows, and recorded in `test_runs.log` and `deviations.md`.
   No run was repeated because of its results.
4. **`-dirty` in the run fingerprints.** The fingerprints show `e8cdeb9…-dirty` because the
   harness's own outputs under `experiments/` (the log, the run directories) were uncommitted
   while it ran. Preflight had verified that nothing outside `experiments/` was modified. The
   fingerprint now ignores `experiments/` (commit `f7fb96f`).
5. **Not pinned by the plan, verified unchanged.** The plan hashes config files but not
   `prompts/` or `configs/guard/rules.yaml`. Neither changed between the lock (`15e02cb`) and
   the test commit (`e8cdeb9`). Of the pipeline code under `src/antibros/llm`, `graph`,
   `guardrails`, `generation`, `reranking` and `retrieval`, only one file changed: an added
   Ollama digest lookup used by preflight.

## Limitations

- **Test split is small.** 30 golden questions (the frozen evalset kept 46 of 48 drafts) and
  24 safety questions. Judge-based metrics rest on the 6–7 answered questions.
- **Narrow corpus and model-built evalset.** One niche corpus of 12 open-access papers. The
  evalset was written by Claude Code and checked by two model verifiers, without human
  review.
- **Model-based scoring.** The judge (`gemma4:e4b`) gives model judgments. The pipeline LLM
  verifies its own citations. Both the judge (`gemma4:e4b`) and the pipeline model
  (`qwen3.5:4b`) were the two evalset verifiers (`configs/models/evalset_verifiers.yaml`), so
  their blind spots may be shared by the labels.
- **Dev results are indicative only.** Tuning and ablations used 16 dev questions. The test
  shortfall in abstention (dev false-abstain 0.417 for `m4_full`, test 0.739) is consistent
  with thresholds over-fitted to that small dev set.
