# ADR-0009: Frozen evaluation sets and milestone tags on a single branch

- **Status:** Accepted
- **Date:** 2026-10-20
- **Milestone:** M2
- **Supersedes:** earlier draft (reference branch + learner branches + separate judge package)

## Context
The repository is a portfolio project and a reference for learners. Claude Code implements everything; the owner supervises. Learners re-implement each milestone in their own repo or branch and need (a) the code state at each milestone and (b) a fair, fixed way to score their work.

## Options considered
| Option | Pros | Cons |
|---|---|---|
| Reference branch + learner branches + separate judge package | Built-in comparison workflow | Heavy process; duplicate implementations; not needed when one author writes all code |
| Single `main` branch, tag per milestone, frozen evalsets with checksums | Simple; `git checkout mX` / `git diff mX-1 mX` shows each step; same scoring for everyone | Learners must bring their own discipline |
| No frozen evalsets | Simplest | Results drift; comparisons become meaningless |

## Decision
- **Single `main` branch**, with tags `m0` … `m6`.
- **`evalsets/v1/` is frozen** with `CHECKSUMS.sha256` + `VERSION`; `antibros evalset verify` runs in CI. Any change creates `v2`, and old runs stop being comparable.
- Every run writes a fingerprint; `antibros compare` refuses runs whose hard fields differ (evalset, corpus, split, metric set, judge model + digest, bootstrap seed).
- Each milestone document contains: Goal · Tasks · Acceptance criteria · Self-check questions.

## Evidence
Process decision, not an experiment.

## Consequences
- **Positive:** low process overhead; transparent history; reproducible scoring.
- **Negative:** learners who use a different judge model cannot compare numbers directly (the fingerprint warns them).
- **Revisit if:** multiple contributors need parallel implementations.
