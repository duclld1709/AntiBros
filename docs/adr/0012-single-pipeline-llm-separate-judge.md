# ADR-0012: One LLM for the whole pipeline, a separate judge from another family

- **Status:** Accepted
- **Date:** 2026-10-20
- **Milestone:** M3, M5

## Context
The pipeline has several LLM steps: input guard, query rewrite, context grading, generation and citation verification. Evaluation also needs a judge model. On a 6GB RTX 3050, a ~4B Q4 pipeline model plus bge-m3 and the reranker nearly fill the card (estimates in spec §4.3, measured at M0). There is no room for a second LLM at the same time. Production will likely use a single API provider.

## Options considered
| Option | Pros | Cons |
|---|---|---|
| Different models per pipeline step (e.g. guard on a second model) | Specialised models; less self-check bias | Two LLMs must be resident at once; does not fit 6GB; more complex production setup |
| One pipeline LLM; a judge loaded only during scoring (predict → judge phases) | Fits VRAM; mirrors a single-provider production setup | Verification and guard are done by the generator itself |
| Same model for pipeline and judge | Simplest | Strong self-evaluation bias in reported metrics |

## Decision
- **The pipeline uses one LLM** for all steps.
- **The judge is a different model family**, loaded only in the `judge` phase. Evaluation is split into `predict` and `judge`, and the pipeline model is unloaded (`keep_alive: 0`) before the judge loads.

## Evidence
`reports/hardware.md` (measured VRAM); model profiles in `configs/models/`.

## Consequences
- **Positive:** fits the hardware; matches production; the judge provides an independent check; scoring can be redone from cache.
- **Negative:** self-check bias inside the pipeline (guard and verify), listed under Limitations. Model swapping adds a few seconds per phase.
- **Revisit if:** production allows a cheap secondary model for guard or verify.
