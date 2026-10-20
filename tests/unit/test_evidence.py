from __future__ import annotations

import pytest

from antibros.ingestion.evidence import evidence_level


@pytest.mark.parametrize(
    ("pub_types", "title", "abstract", "expected"),
    [
        (["Meta-Analysis", "Systematic Review", "Journal Article"], "", "", 1),
        (["Systematic Review", "review-article"], "", "", 1),
        (["Randomized Controlled Trial", "Journal Article"], "", "", 2),
        (["Observational Study"], "", "", 3),
        (["Practice Guideline"], "", "", 4),
        # Scoping reviews map evidence without pooling: level 4, whatever else is tagged.
        (["Scoping Review", "review-article", "Journal Article"], "", "", 4),
        (["Systematic Review"], "Creatine: a scoping review of RCTs", "", 4),
        # Position stands tagged only as "Review".
        (["review-article", "Review"], "ISSN position stand: protein and exercise", "", 4),
        # No specific type: title rule.
        (["review-article", "Journal Article"], "Protein timing: a meta-analysis", "", 1),
        (["Journal Article"], "A randomized trial of whey", "", 2),
        # Generic review beats abstract keywords (reviews talk about RCTs).
        (["Review"], "Common questions", "We summarise randomized trials.", 4),
        # Research article: abstract design words.
        (["research-article", "Journal Article"], "Pre- vs post", "were randomly assigned", 2),
        (["research-article"], "Diet and muscle", "a prospective cohort of adults", 3),
        (["research-article"], "Commentary", "", 5),
        ([], "", "", 5),
        (["Meta\N{HYPHEN}Analysis"], "", "", 1),
    ],
)
def test_evidence_level(pub_types: list[str], title: str, abstract: str, expected: int) -> None:
    assert evidence_level(pub_types, title, abstract) == expected
