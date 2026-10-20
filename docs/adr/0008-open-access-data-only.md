# ADR-0008: Ingest open-access sources only

- **Status:** Accepted
- **Date:** 2026-10-19
- **Milestone:** M1

## Context
The repo and demo are public. Copyrighted guideline books (e.g. ACSM Guidelines) cannot be redistributed or served as retrieved passages.

## Options considered
| Option | Pros | Cons |
|---|---|---|
| Open-access only (PMC OA subset, e.g. JISSN position stands, MDPI Nutrients) | Legal to store and quote snippets; reproducible | Some authoritative sources excluded |
| Include paywalled/copyrighted sources privately | Better coverage | Cannot publish corpus or demo; legal risk |

## Decision
**Open-access only.** Every document stores a `license` field, verified via the Europe PMC API at M1; the UI and README show attribution. Any NC-licensed article is used for a non-commercial demo only.

## Evidence
Licensing requirement, not an experiment.

## Consequences
- **Positive:** fully public, reproducible corpus snapshot.
- **Negative:** some authoritative sources are excluded; documented in Limitations.
- **Revisit if:** the project becomes commercial (re-check NC licenses).
