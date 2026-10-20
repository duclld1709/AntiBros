# ADR-0013: Laptop-only development with a small niche corpus and a per-run evaluation time budget

- **Status:** Accepted
- **Date:** 2026-10-20
- **Milestone:** M0–M6
- **Deciders:** project owner
- **Supersedes:** earlier draft "Time-boxed GPU session with a pre-registered plan" (150-article corpus, one 2h30 V100 session)

## Context
The only machine is a laptop with an RTX 3050 (6GB VRAM) running native Windows (ADR-0016). The owner wants every evaluation run to finish in roughly **10–20 minutes**, so the project can be iterated on and understood step by step. Once the techniques are understood, the owner will build a broader corpus and evaluation set (`v2`) independently.

A local ~4B model on this GPU is expected to handle roughly one question every 20–30 seconds end to end, including judging (to be measured at M0/M3). That rules out a large evaluation set.

## Options considered
| Option | Pros | Cons |
|---|---|---|
| 150 articles, 4 topics, one rented V100 session | Broader coverage; larger models | Extra infrastructure; one-shot session; not what the owner wants |
| Laptop only, 150 articles, large evalset | Broad coverage | Each evaluation run takes hours; slow learning loop |
| **Laptop only, 12 articles in one niche (protein & creatine), evalset sized for 10–20 min runs** | Fast iteration; everything reproducible on one machine; clear scope | Wide confidence intervals; results do not generalise beyond the niche; retrieval at document level becomes trivial |

## Decision
- **Hardware:** everything (development, evalset verification, ablation, final evaluation) runs on the laptop. No GPU rental.
- **Corpus v1:** 12 open-access articles on muscle-building nutrition: protein (T1, 7 articles) and creatine (T2, 5 articles).
- **Evalset v1:** 48 golden questions (dev 16 / test 32), 40 safety questions (dev 16 / test 24), and 96 retrieval-synthetic queries (dev only).
- **Time budget per command** (see spec §6.8):

  | Command | Budget |
  |---|---|
  | Retrieval evaluation | ≤ 2 min |
  | Safety evaluation | ≤ 5 min |
  | End-to-end on dev | ≤ 10 min |
  | End-to-end on test | ≤ 20 min |

  Every run records `timing.json`.
- **If over budget,** apply the pre-declared order: engineering optimisations → smaller context → `laptop_light` model profile. Never shrink the frozen evaluation set.
- Retrieval quality is measured at the evidence-span level, not the document level (ADR-0015).

## Evidence
`reports/hardware.md` (measured VRAM and throughput), `timing.json` in every `experiments/` run.

## Consequences
- **Positive:**
  - Short feedback loop.
  - No session logistics.
  - Anyone with a 6GB GPU can reproduce every number.
  - The constraint itself is a good engineering story.
- **Negative:**
  - 95% CIs are about ±0.15 on test and ±0.22 on dev for proportion metrics.
  - Conclusions apply only to this niche.
  - Model quality is limited to small local models.
- **Revisit if:** the owner starts `v2` (broader corpus and evalset), or more compute becomes available. Either way, create `evalsets/v2` rather than enlarging `v1`.
