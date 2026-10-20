# AntiBros evaluation set v1 — Data Card

## Summary

Version `v1` is a frozen evaluation set for retrieval and answer-quality development on the
12-paper open-access protein and creatine corpus listed in `corpus_manifest.jsonl`. The
golden and safety sets have stratified `dev` and `test` splits; retrieval-synth is `dev` only
and is intended for retrieval ablations. Do not use these sets as clinical guidance.

**Drafted by Claude Code, no human review.** Two local models from different families
reviewed each retained item's final revision: `gemma4:e4b` and `qwen3.5:4b`. The complete
verdict and decision history is in `review_log.jsonl`.

## Composition and review outcomes

The initial drafts contained 48 golden, 40 safety, and 96 synthetic queries (184 total).
Review kept 180 items (97.8%) and dropped four (2 golden, 2 synth; 2.2%). Forty-two unique
items were revised at least once before their final keep/drop decision. Revision counts
overlap with keep/drop counts: 38 revised items were retained and 4 revised items were later
dropped.

| Set | Drafted | Kept | Revised at least once | Dropped | Final distribution |
|---|---:|---:|---:|---:|---|
| Golden | 48 | 46 | 7 | 2 | answerable 23; myth 12; unanswerable 11 |
| Safety | 40 | 40 | 3 | 0 | red_flag 20; near_red_flag 20 |
| Retrieval-synth | 96 | 94 | 32 | 2 | 8 queries per paper except two papers with 7 |

The retained golden counts are within one item per target category/language cell: answerable
11 EN / 12 VI, myth 4 EN / 8 VI, unanswerable 6 EN / 5 VI. Safety has 10 items in each
category/language cell. Synthetic queries are balanced 47 EN / 47 VI; two papers have seven
queries after exclusions, with the remaining ten at eight each. These deviations follow the
M2 allowance and are recorded here.

Golden has 16 dev and 30 test items; safety has 16 dev and 24 test items. Splits were assigned
with seed 42 and stratified by category, language, and topic. All 94 retrieval-synth queries
are dev-only.

Four items were excluded after the final allowed verifier round because of unresolved
verifier disagreement: golden `g004` (key-point support) and `g044` (Vietnamese phrasing),
and synth `x048` and `x057` (whether the quoted passage directly answered the query). Their
drop decisions and verifier outputs remain in `review_log.jsonl`.

## Intended use

- Measure evidence retrieval on the golden dev split and retrieval-synth set.
- Compare retrieval configurations and chunking/indexing ablations.
- Develop answer and safety evaluation workflows using only the dev splits.

The test splits are reserved for the pre-registered final evaluation described in
`docs/runbooks/final_eval.md`. Retrieval-synth is generated from source passages and should
not be used as a stand-in for naturally occurring user questions.

## Baseline retrieval results

Both baselines used hybrid RRF retrieval, `section_aware` chunks, and the `bge_m3` embedder
from `configs/base.yaml`. Qdrant ran in local embedded mode for these runs because the server
was unavailable. The indexed corpus had 393 section-aware chunks. Intervals are the harness's
95% bootstrap confidence intervals.

| Set / split | Queries | Evidence Recall@10 | MRR | Retrieval time | Run |
|---|---:|---:|---:|---:|---|
| Golden dev | 12 | 0.476 [0.299, 0.645] | 0.550 [0.305, 0.783] | 3.25 s | `experiments/20261020-1609_m2_golden_dev/` |
| Retrieval-synth dev | 94 | 0.713 [0.628, 0.809] | 0.472 [0.392, 0.555] | 7.12 s | `experiments/20261020-1610_m2_synth_dev/` |

The intervals for golden dev are wide because the sample is small. The synth score is likely
optimistic relative to real queries because each question was written from a source passage,
even with a lexical-overlap check.

## Data sources and construction

The corpus consists of 12 open-access papers on protein and creatine for muscle-building
contexts. The manifest records document identifiers, titles, years, topics, licenses, and
source-text hashes. Questions and evidence follow `rubric.md`; evidence is stored as verbatim
quotes rather than chunk identifiers so retrieval can be evaluated across chunking changes.
Additional pooled evidence and grading artifacts are kept under `work/`.

Question types are answerable, myth, and unanswerable in golden; red-flag and near-red-flag in
safety; and passage-linked paraphrases in retrieval-synth. English and Vietnamese items are
included. Quotes and document IDs are validated against the cleaned corpus and manifest.

## Limitations and risks

- There was no human review. Model agreement and automated checks do not establish factual or
  clinical correctness, and the verifier models can share systematic blind spots.
- The verifier pair used for this version is also used elsewhere in the project (Gemma 4 E4B
  as judge and Qwen 3.5 4B as pipeline LLM), so their judgments are not independent of all
  downstream project components.
- The corpus is small, limited to selected open-access papers, and is not a representative
  sample of supplement questions or users.
- Synthetic queries are conditioned on source passages and may favor retrieval of those
  passages. They are for ablations only.
- The four unresolved items were excluded after the configured revision limit; their
  exclusion can leave edge cases underrepresented.
- Confidence intervals do not capture errors in the corpus, annotations, or verifier rubric.
- Red-flag categories are project safeguards, not a clinically validated screening tool.

## Integrity

`VERSION` records the frozen version, corpus-manifest digest, and final item counts.
`CHECKSUMS.sha256` covers every file under `evalsets/v1`, including this data card and the
draft/review artifacts. Run `antibros evalset verify` to check byte-exact integrity and
`antibros evalset leakcheck` to scan project source, configs, and prompts for question leakage.
