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

## Implementation (M5)
- `antibros eval final` (`src/antibros/eval/final.py`) runs the checks above before touching the
  test split. It also verifies the profile sha256, and compares the digests the local Ollama
  server reports (`/api/tags`) with the plan. Changes under `experiments/` (the harness's own
  outputs) do not count as a dirty tree, so an interrupted run can resume.
- `experiments/test_runs.log` is JSONL, one line per state change of a planned run:
  `started`, `resumed`, `interrupted`, `failed`, `completed`. Each line carries the plan
  sha256, commit, hardware and run directory.
- An *interrupted* run resumes in place, on the same commit and hardware. This is not a
  re-run: written predictions and judgments are kept.
- A *failed* run (infrastructure: prediction errors, judge transport errors, crashes) is
  never resumed. `--rerun <run> --reason ...` starts it once in a new `_rerun1` directory and
  appends the reason to `experiments/deviations.md`; a second re-run is refused.
- A *completed* run is never run again. All decisions are taken before any run starts, so a
  refusal leaves nothing half-done.
- `--rehearsal` exercises the same flow on dev with `--limit` questions, without log entries.
- After `baseline_test` and `full_test`, a paired-bootstrap comparison (overall, per
  language, per category, including abstention F1) is written next to the runs.
- A unit test fails CI if any planned config, the model profile or the evalset stops matching
  the locked plan.

## Evidence
`configs/final_eval_plan.yaml`, `experiments/test_runs.log`, `experiments/deviations.md`.

## Consequences
- **Positive:** test numbers are unbiased estimates for this setup; any deviation is visible.
- **Negative:** design mistakes found on test cannot be fixed within v1. They are documented and carried to v2.
- **Revisit if:** a new evaluation set version is created; it gets its own plan.
