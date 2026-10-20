# CV bullets (every number links to its run)

- Built **AntiBros**, a cited RAG assistant for protein/creatine questions (Vietnamese and
  English, 12 open-access papers).
  - **Pipeline:** LangGraph pipeline with configurable steps: dense/hybrid retrieval,
    cross-encoder reranking, an evidence grade, LLM citation verification and a medical
    guard.
  - **Footprint:** runs on one 6 GB GPU (pipeline LLM + embedder + reranker with 323 MiB
    headroom, [hardware report](hardware.md)) and packaged as a one-command Docker stack.
- Designed a **pre-registered evaluation**.
  - **Setup:** a frozen evalset (180 items, leak-checked), span-level retrieval labels, a
    cross-family LLM judge and paired bootstrap CIs. Test runs only through a locked plan
    (config hashes, model digests, guard-selection rule).
  - **Reported result:** the full pipeline did **not** significantly beat the baseline on
    test: abstention F1 +0.038 [−0.074, +0.177], Evidence Recall@10 +0.005 [−0.078, +0.099]
    ([comparison](../experiments/final-c01d215e_comparison.md)).
- Found by **error analysis** of every wrong test item that over-abstention dominates.
  - **Causes:** 24 of 35 errors were invalid JSON from the 4B generator, 7 an over-strict
    evidence grade ([error analysis](error_analysis.md)).
  - **Guard:** reached red-flag recall 12/12 at FPR 0.083 [0.000, 0.250]
    ([guard run](../experiments/final-c01d215e_guard_test/)).
- **Dev ablations** (16 questions, indicative only): reranking raised Evidence Recall@10 by
  0.195 [0.070, 0.330] over the same pipeline without it ([ablation report](ablation.md)).
