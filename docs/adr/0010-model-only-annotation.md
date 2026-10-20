# ADR-0010: Evaluation sets drafted by Claude Code, verified by models, no human review

- **Status:** Accepted
- **Date:** 2026-10-20
- **Milestone:** M2
- **Deciders:** project owner

## Context
The evaluation sets (48 golden questions, 40 safety questions, 96 retrieval-synthetic queries) need reference answers, evidence labels and safety labels. **The project owner decided that Claude Code writes every evaluation item on its own, with no human help or review.** This decision must be recorded because it affects how much the reported numbers can be trusted. The owner plans to build and personally verify a broader `v2` dataset later (see ADR-0013).

## Options considered
| Option | Pros | Cons |
|---|---|---|
| Human expert annotation | Highest quality | Not available |
| Non-expert human review against sources | Cheap quality gate | Owner chose not to do it |
| Claude Code drafts; deterministic checks; two local verifier models from different families | No human time; every decision logged; reproducible | No human ground truth; model biases can survive both verifiers |
| A single model drafts and checks | Simplest | Self-confirmation; weakest |

## Decision
1. **Rubric first.** Claude Code writes `evalsets/v1/rubric.md` before writing any item.
2. **Drafting.** Claude Code drafts all items.
   - Each reference answer is a list of key points with **verbatim evidence quotes** (`doc_id` + quote) from the cleaned document text (ADR-0015).
   - Extra evidence is found by **pooling** the top-10 results from BM25, bge-m3 and e5, then graded 0/1/2.
3. **Deterministic checks:**
   - Quotes exist in the cleaned text; document IDs exist in the manifest; schema is valid.
   - No duplicates or near-duplicates.
   - **Lexical overlap between each question and its evidence stays below a threshold**, to limit synthetic-question bias.
   - Unanswerable items have no answering passage in the top-20 results.
   - Stratified split proportions are correct.
4. **Model verification.** Two local models from different families (`configs/models/evalset_verifiers.yaml`) independently grade every item against the rubric.
   - If both agree, the item is kept.
   - If they disagree, Claude Code revises the item (at most two rounds, re-verified each time) or drops it.
5. **Logging and freezing.**
   - Every decision is logged in `evalsets/v1/review_log.jsonl`.
   - `DATA_CARD.md` reports keep, revise and drop rates.
   - The sets are frozen (checksums) **before** prompts are written and the pipeline is tuned.
6. **Leakage control.** The same agent writes both the evaluation sets and the pipeline, so:
   - `antibros evalset leakcheck` (run in CI) rejects any ≥ 8-word n-gram from questions or key points that appears in `src/`, `configs/` or `prompts/`.
   - When writing prompts, Claude Code looks only at aggregate metrics and dev-split failures.

## Evidence
`evalsets/v1/DATA_CARD.md`, `evalsets/v1/review_log.jsonl`, CI job `evalset-verify`.

## Consequences
- **Positive:** fully reproducible annotation pipeline; transparent decision log; no human time required.
- **Negative:**
  - No human ground truth.
  - Labels may share the blind spots of the drafting and verifying models.
  - A verifier model may also be the judge model.
  - Red-flag criteria are not clinically validated.
  - The author of the evaluation set also wrote the system under test; leakcheck catches copying, not subtle familiarity.
  - The README must state: *"Evaluation sets drafted by Claude Code and verified by two local models; no human review."*
- **Revisit if:** a human (ideally domain-trained) becomes available. Reviewing a random sample of ~20 items and reporting the agreement rate is the cheapest upgrade. The owner's planned `v2` dataset supersedes this for broader claims.
