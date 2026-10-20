# Deviations from the pre-registered M5 evaluation plan

## 2026-10-20 — Run the final evaluation on Linux Tesla V100

- **Planned environment:** Laptop running Windows with an NVIDIA RTX 3050 6 GB, as specified by `docs/runbooks/final_eval.md` and `configs/final_eval_plan.yaml`.
- **Actual environment:** Linux with an NVIDIA Tesla V100-PCIE-16GB. The plan's Ollama models are installed with matching digests: pipeline `qwen3.5:4b` (`d8b0f5e9760c...`) and judge `gemma4:e4b` (`dc35e8d9c606...`). Both use Q4_K_M and the locked plan's temperature 0.0.
- **Reason:** The final evaluation is being run in the available V100 environment rather than waiting for the Windows laptop; this deviation was explicitly requested by the project owner.
- **Interpretation:** Quality metrics remain comparable under the locked model digests and deterministic temperature setting. Latency and elapsed time are hardware- and environment-dependent, so they must not be compared directly with laptop measurements. Reports must identify this run as V100/Linux and state this limitation.
- **Scope:** No model, digest, quantization, temperature, prompt, config, threshold, evalset, or test questions are changed.

## 2026-10-20 — CUDA-enabled PyTorch required to resume `full_test`

- `baseline_test` completed. The first `full_test` attempt failed before producing predictions because `configs/full.yaml` requests the CUDA reranker while the Linux venv contains the CPU-only PyTorch wheel (`Torch not compiled with CUDA enabled`).
- This is an environment issue, not an evaluation result. The runbook permits a minimal infrastructure fix and one resume of the failed run. Install a CUDA-enabled PyTorch wheel in the local venv, without changing repository dependency files, then resume `full_test` once with the documented `--rerun` and this reason.
- The failed attempt and its failure remain in `experiments/test_runs.log`; do not rerun `baseline_test`.

## 2026-10-20: re-run of `full_test`

- Plan sha256: `c01d215e580880ced7a6d88d0e3cde3c9dd535fff45bd0e5592adfde4c397597`; commit `e8cdeb937c2eb6609826ce079600630937d4f958`; hardware Tesla V100-PCIE-16GB
- Failed attempt: `experiments/final-c01d215e_full_test` (Torch not compiled with CUDA enabled)
- New attempt: `experiments/final-c01d215e_full_test_rerun1`
- Reason (infrastructure only, never results): Infrastructure failure: locked full config requires CUDA reranker, but environment had CPU-only torch. Installed local CUDA torch 2.7.1+cu126 supporting V100; no repo files/config changed.
