# ADR-0020: Score generation with a separate LLM judge

- **Status:** Accepted
- **Date:** 2026-10-20
- **Context:** M3

## Context

M3 needs end-to-end scores for faithfulness, answer relevancy, reference key-point coverage,
and citation precision/recall. We need these scores to use the same frozen prediction and evidence
artifacts as later milestone comparisons. The laptop cannot keep the pipeline model and judge model
resident together, so judging runs as a separate phase.

## Decision

Use a provider-agnostic structured-output LLM judge after prediction, with a model family separate
from the generation model. The judge receives the question, reference key points, answer claims,
and only the context passages cited by those claims. It returns five normalized scores in `[0, 1]`.
Abstentions receive null judge scores because those dimensions do not apply. The runner records
per-item judgments and errors in resumable `judgments.jsonl`; aggregate metrics are computed from
those stored records.

## Consequences

- Judging is reproducible against stored predictions and can be resumed without rerunning generation.
- Separate prediction and judge phases keep their model memory footprints from overlapping.
- Scores depend on the judge model and prompt; treat them as model-based measurements, not human
  ground truth. Keep the judge profile, prompt, run artifacts, and model digest with each baseline.
- The implementation avoids an extra judge framework dependency and keeps provider selection behind
  the shared LLM client.
