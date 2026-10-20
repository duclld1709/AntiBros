# ADR-0018: Deterministic JATS cleaning and evidence-level rules

- **Status:** Accepted
- **Date:** 2026-10-20
- **Milestone:** M1
- **Deciders:** project owner

## Context
Evidence spans in the evalset (M2) are anchored to each document's clean text (ADR-0015), so
the clean text must be identical on every run and every OS. The spec (§3.3, M1) fixes the goals
(drop references and acknowledgements, normalise whitespace and Unicode, one consistent rule
for captions) but not the exact rules. Checking the 12 real articles also showed gaps:

- The narrative reviews and position stands use few standard headings. Several nest their
  whole text under a "Background" or "Introduction" section.
- Some articles put administrative sections in `<body>` (competing interests, funding,
  "cite this article as" notes), not in `<back>`.
- The RCT (PMC5214805) is tagged only `research-article`, and the 2013 meta-analysis
  (PMC3879660) only `review-article`. Rule "pub_types, then title" from spec §3.4 would put the
  RCT at level 5 ("other").

## Options considered
| Option | Pros | Cons |
|---|---|---|
| Keep tables (cells as text) | More numbers available to retrieval | Cell order and spacing differ between publishers; noisy chunks; spans in tables are hard to quote |
| **Captions only for tables and figures** | Simple, deterministic, same rule for every article | Numbers that appear only in table cells cannot be retrieved |
| Evidence level from pub_types, then title (spec) | As specified | RCT tagged `research-article` becomes level 5 |
| **Add abstract design keywords as a last rule** (after generic review types) | Fixes the RCT; reviews are unaffected | One more rule to test |

## Decision
Cleaning (`CLEANING_VERSION = "1"`, `src/antibros/ingestion/jats.py`):
- Clean text = "Abstract" + abstract, then the body. The title is metadata only.
- Blocks (paragraphs, headings, list items, captions) are joined by a blank line. Inside a
  block, whitespace runs become one space, and text is NFC.
- Bibliographic `xref`s are removed, then empty bracket leftovers.
- Tables and figures keep `"<label>. <caption>"`; cells, table footnotes, images and
  supplementary material are dropped.
- `<back>`, administrative body sections, and untitled top-level body sections are dropped.
- Section label comes from `sec-type`, then heading keywords, then the parent section. The
  exception is a child of `introduction`, which becomes `body`. A heading-only segment is
  merged into the next segment.

Evidence level (`src/antibros/ingestion/evidence.py`), first match wins:
1. Scoping review, position stand or consensus statement gives level 4.
2. Specific publication types: the lowest level among them.
3. Title keywords.
4. A generic review type gives level 4.
5. Abstract design keywords (randomised, cohort, …). Otherwise level 5.

## Evidence
- `reports/corpus_stats.md`: 12/12 documents parse and every one has at least 3 standard
  sections. The evidence levels match the publication type of each article in spec §3.3.
- `tests/unit/test_jats.py` pins the clean-text sha256 of three fixtures. CI runs it on
  Windows and Linux.

## Consequences
- **Positive:** One rule set for every article, documented and tested. Clean text does not
  depend on the OS or on input line endings.
- **Negative / risks:** Numbers that appear only in table cells cannot be retrieved. Any change
  to these rules changes the character offsets, so it requires a new `CLEANING_VERSION` and,
  after the M2 freeze, a new evalset version.
- **Revisit if:** error analysis (M5) shows questions that only table data could answer.
