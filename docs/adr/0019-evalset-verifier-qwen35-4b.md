# ADR-0019: Use Qwen 3.5 4B for evalset verification

- **Status:** Accepted
- **Date:** 2026-10-20
- **Milestone:** M2
- **Deciders:** Project owner

## Context

The evalset verifier pair must use different model families and finish on the project's
Windows laptop. The original Qwen 3.5 9B verifier ran with partial CPU offload, spilled into
shared GPU memory, and took about 35 seconds per item. That made verification of the complete
evalset unnecessarily slow and increased pressure on the laptop's limited GPU memory.

## Options considered

| Option | Pros | Cons |
|---|---|---|
| Keep Qwen 3.5 9B | Larger model; preserves the initial verifier profile | Partial offload, shared-memory spill, about 35 seconds per item |
| Replace it with Qwen 3.5 4B | Fits the laptop workload better and completed the outstanding verifier items | Lower model capacity; still shares a model family with the pipeline LLM |

## Decision

Use `qwen3.5:4b` (Q4_K_M, digest `d8b0f5e9760c…`) with `gemma4:e4b` as the two evalset
verifiers. Keep the two verifier families distinct. Record the model identities with every
verdict so the frozen evalset's review history remains auditable.

## Evidence

- Laptop observation during the M2 verifier run: Qwen 3.5 9B required partial CPU offload,
  spilled into shared GPU memory, and took approximately 35 seconds per item.
- The updated verifier profile is `configs/models/evalset_verifiers.yaml`; the review log
  records the verifier model and digest for each verdict.

## Consequences

- **Positive:** Evalset verification fits the laptop's memory and runtime constraints better.
- **Negative / risks:** The smaller Qwen model may miss issues the 9B model would identify;
  both verifiers are automated and neither constitutes human review.
- **Revisit if:** A future laptop can run the larger model without shared-memory spill, or
  sampled checks show materially weaker verification with 4B.
