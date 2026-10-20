# Hardware report (M0)

All numbers are **measured** on 2026-10-20 with `scripts/vram_probe.py` (raw data:
[`hardware_probe.json`](hardware_probe.json)). Each figure is a **single run (n=1)**, so treat
throughput as indicative ±10–20% (no repeated runs yet). The M3 run will re-measure end-to-end
seconds per question.

## Environment
| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 6GB Laptop GPU (6144 MiB), driver 581.86 |
| OS | Windows 11 Home 10.0.26200 (native, no WSL) |
| PyTorch | 2.11.0+cu128, `torch.cuda.is_available() == True` |
| Ollama | 0.40.6 (Windows native), models in `D:\cache\ollama\models` |
| Qdrant | server `qdrant/qdrant:v1.15.5` (Docker Desktop 28.5.1), client `qdrant-client` 1.19.1 |

## Display GPU
Windows lists only one video controller: the RTX 3050. **The RTX 3050 drives the display**, so
there is no iGPU offload (Optimus off or MUX in dGPU mode). Idle VRAM use with normal desktop
apps open: **548–563 MiB**. That memory is not available to models. Closing GPU-heavy apps
(browser, Electron apps) before an eval frees part of it.

## Sysmem fallback
| Check | Result |
|---|---|
| Driver setting *CUDA – Sysmem Fallback Policy* | Present on driver 581.86 |
| Default (*Driver Default*): allocate 512 MiB blocks up to 6144 MiB | **No OOM**: the driver silently spilled into system RAM |
| After setting **Prefer No Sysmem Fallback** (global) | **`torch.OutOfMemoryError` after 5120 MiB** (expected) |

The setting is now **Prefer No Sysmem Fallback**, so running out of VRAM fails loudly.

## LLMs (Ollama, thinking off, temperature 0)
Prompt ≈ 2,185 tokens, generation ≤ 256 tokens. VRAM = increase in `nvidia-smi` used memory
after loading (includes KV cache and CUDA context).

| Model | Quant | `num_ctx` | VRAM (MiB) | Fully on GPU | Prefill tok/s | Decode tok/s |
|---|---|---|---|---|---|---|
| `qwen3.5:4b` (pipeline) | Q4_K_M | 4096 | 3,751 | yes | 959* | 41.9 |
| `qwen3.5:4b` (pipeline) | Q4_K_M | 8192 | 4,016 | yes | 1,947 | 42.3 |
| `gemma4:e4b` (judge) | Q4_K_M | 4096 | 4,577 | yes | 1,066 | 54.8 |
| `gemma4:e4b` (judge) | Q4_K_M | 8192 | 4,554 | yes | 1,385 | 50.1 |
| `qwen3.5:2b` (light) | Q8_0 | 4096 | 3,033 | yes | 4,085 | 58.0 |
| `qwen3.5:2b` (light) | Q8_0 | 8192 | 3,107 | yes | 4,013 | 58.3 |

\* The first measured run after a cold load; likely a warm-up artefact (n=1).

Notes:
- For `gemma4:e4b`, Ollama's `/api/ps` reports `size_vram` ≈ 350 MiB, which disagrees with
  `nvidia-smi` (+4.5 GiB). We trust `nvidia-smi`. "Fully on GPU" for gemma4 is inferred from the
  VRAM increase plus decode speed, not from `/api/ps`.
- `qwen3.5:2b`'s default registry tag is **Q8_0**, not Q4 as assumed in spec §4.4.
- `qwen3.5:9b` (evalset verifier): not probed. It needs partial CPU offload, which the spec accepts for a one-off M2 run.

## Encoders (PyTorch FP16, CUDA)
| Model | VRAM via nvidia-smi (MiB) | Torch peak allocated (MiB) |
|---|---|---|
| `BAAI/bge-m3` | 1,346 | 1,147 |
| `BAAI/bge-reranker-v2-m3` | 1,108 | 1,148 |

(The batch timings in the JSON include the first-time model download, so they are not throughput.)

## Combined: pipeline LLM + embedder + reranker
`qwen3.5:4b` at `num_ctx` 8192, plus bge-m3 and the reranker (FP16), all resident. The test ran
one rerank batch of 20 pairs and one long generation.

| Used before | Used after | Total | Headroom | LLM fully on GPU | Decode tok/s |
|---|---|---|---|---|---|
| 873 MiB | 5,821 MiB | 6,144 MiB | **323 MiB** | yes | 36.5 |

## Conclusions
1. **The `laptop` profile fits in 6 GB**: pipeline LLM (8k context) + bge-m3 + reranker run fully on GPU, with fallback disabled.
2. **Reranker on GPU by default.** Headroom is only ~320 MiB with the display on the same GPU.
   The reranker's `device` stays configurable and is the first thing to move to CPU on OOM. Larger rerank batches or a bigger `num_ctx` are not safe.
3. **Pipeline LLM and judge cannot be co-resident** (≈4.0 + 4.5 GiB). This confirms the `predict` → `judge` split (ADR-0012).
4. `num_ctx` 4096 → 8192 costs only ~265 MiB for `qwen3.5:4b`. 8192 stays the default.
5. End-to-end M3 performance is measured below: predict averages 14.53 s/question and the judge phase averages 1.96 s/question over all 16 dev items.
6. Embedded Qdrant (`vector_store.mode: local`) supports dense + sparse named vectors and RRF fusion. Verified on disk and `:memory:` with `qdrant-client` 1.19.1.

## M3 baseline (2026-10-20)

Run: [`experiments/20261020-1633_m3_baseline_v2/`](../experiments/20261020-1633_m3_baseline_v2/). It used all 16 frozen golden dev questions: 6 answered and 10 abstained. Predict took 232.55 s (14.53 s/question); judge took 31.38 s (1.96 s/question averaged over 16; abstained cases require no judge call); total 263.93 s, within the 600 s budget. `timing.json`, predictions, judgments, per-item metrics, and aggregate metrics are in the run directory.

The baseline emitted 18 valid structured outputs from 24 attempts (75%). All citations on the six answered items mapped to retrieved context (citation validity 100%). Judge means for answered items: faithfulness 0.925, key-point coverage 0.900, citation precision 0.967, citation recall 0.917. Abstention F1 was 0.571. These are small-sample dev results, not a test-set estimate.

Cache replay run [`experiments/20261020-1646_m3_cache_replay_verified/`](../experiments/20261020-1646_m3_cache_replay_verified/) produced 24 cache hits from 24 structured-output requests (100%); 18 cached outputs validated. Invalid raw outputs are cached too, so a malformed response does not trigger another provider call during identical replay.

During the cache replay, sampled `nvidia-smi` memory peaked at 5,215 MiB used (one-second sampling, n=1; includes display and other GPU allocations). The existing isolated model probe measured 4,016 MiB for qwen3.5:4b at 8k context and 1,346 MiB for BGE-M3; do not sum them because they are loaded in separate phases. The 32-question test split is projected at about 8.8 minutes by linear extrapolation of the full predict+judge baseline (actual dev run: 4.4 minutes). Full 48-question evaluation is projected at about 13.2 minutes. Projections are estimates; workload composition and warm cache can change runtime.

API/UI smoke: `GET /health` returned 200; `POST /ask` returned an answered response with three citations for a golden dev query; its trace was appended to `experiments/traces.jsonl`. Streamlit served HTTP 200 on localhost. The UI/API smoke was manual on the laptop.


## M4 tune OOM note (2026-10-20)

The `grade_min_rerank_score` tuning run using the GPU reranker was interrupted after a VRAM OOM that crashed VS Code. Its partial run is preserved at `experiments/20261020-2100_tune_grade_min_rerank_score_0p02/` and has no metrics or timing, so it is not a completed evaluation. Following the runbook, `configs/base.yaml` and `configs/full.yaml` set `rerank.device: cpu` to free GPU memory before resuming the tuning grid. The resolved config for the first resume attempt still selected CUDA, so that attempt was stopped before the tuning grid completed; the full config now pins CPU explicitly.
