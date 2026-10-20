# Error analysis (M5, golden test + safety test)

> **Status: TBD.** The final evaluation has not run yet. Every number here must come from the
> `experiments/final-*` run directories; procedure: `docs/runbooks/final_eval.md` §6.

Draft labels come from `antibros eval errors` (rules in `src/antibros/eval/errors.py`,
thresholds in `configs/error_analysis.yaml`). Each wrong item below carries the reviewed
label; where the review changed the draft, the reason is given. Dev examples may be added
when a category has no test example, and are marked as dev.

## Wrong items by category

| Category | Meaning | baseline | full | guard |
|---|---|---|---|---|
| retrieval_miss | No evidence span in the final ranking | TBD | TBD | — |
| ranking | Evidence retrieved but below the passages given to generation | TBD | TBD | — |
| unfaithful | Answer not supported by the cited passages (faithfulness < 0.5) | TBD | TBD | — |
| citation | Citations do not support their claims (citation precision < 0.5) | TBD | TBD | — |
| incomplete | Key points missing (coverage < 0.5) | TBD | TBD | — |
| translation | Vietnamese query lost meaning in translation / rewrite | TBD | TBD | — |
| wrong_abstain | Abstained on an answerable item, or answered an unanswerable one | TBD | TBD | — |
| guard | Golden item redirected; red flag missed; near-red-flag flagged | TBD | TBD | TBD |
| **Total wrong / items** | | TBD | TBD | TBD |

## Every wrong test item

| Run | ID | Lang | Category | Draft label | Reviewed label | Note |
|---|---|---|---|---|---|---|
| TBD | | | | | | |

## Typical examples

TBD: one or two examples per frequent category, with the retrieved passages and the answer.

## Directions for v2 (not applied in v1)

TBD. Findings on test are never used to change the v1 pipeline (spec §7, M5).

## Limitations
Small test split (32 golden, 24 safety questions); one niche corpus of 12 papers; evalset
written by Claude Code and checked by models, without human review; judge scores are model
judgments; the pipeline verifies its own citations.
