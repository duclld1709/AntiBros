# ADR-0014: Pre-registered final evaluation on the test split

- **Status:** Accepted
- **Date:** 2026-10-20
- **Milestone:** M4 (plan), M5 (run)

## Context
With everything on one laptop, nothing physically prevents running the test split repeatedly while tuning. With only 32 golden test questions, picking the best of several configurations on test would inflate the reported numbers far more than the true improvement.

## Options considered
| Option | Pros | Cons |
|---|---|---|
| Honour system ("don't look at test") | No tooling | Easy to break, especially for an automated agent |
| No test split; report dev only | Simple | Dev is used for tuning, so numbers are optimistic |
| **Test split gated by tooling and a committed, pre-registered plan** | Honest numbers; violations are visible | Some tooling work; less flexibility |

## Decision
- `antibros eval predict --split test` is refused. Test runs happen only via `antibros eval final --plan configs/final_eval_plan.yaml`.
- Before running, `eval final` checks:
  - the working tree is clean and the plan is committed;
  - config hashes and model digests match the plan;
  - the evaluation set checksums verify;
  - no planned run already appears in `experiments/test_runs.log`.
- **Hypotheses** (fixed at the end of M4):
  - Main comparison: `full` vs `baseline` on golden test.
  - Guard: the variant is chosen on dev by a fixed rule (highest red-flag recall with FPR ≤ 0.15; ties go to lower latency), then run once on safety test.
  - Leave-one-out ablations are reported on dev only.
- **Re-runs:** a run that fails for infrastructure reasons (OOM, runtime crash) may be re-run once, recorded in `experiments/deviations.md`. A run is never re-run because of its results.
- A `--rehearsal` mode exercises the whole flow on dev with `--limit 3` and does not write to `test_runs.log`.

## Evidence
`configs/final_eval_plan.yaml`, `experiments/test_runs.log`, `experiments/deviations.md`.

## Consequences
- **Positive:** test numbers are unbiased estimates for this setup; any deviation is visible.
- **Negative:** design mistakes found on test cannot be fixed within v1. They are documented and carried to v2.
- **Revisit if:** a new evaluation set version is created; it gets its own plan.
