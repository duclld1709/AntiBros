# AntiBros

Evidence-based Q&A for muscle-building nutrition (protein & creatine): cited RAG with hybrid
search, reranking, citation verification, abstention and medical guardrails — built and
evaluated entirely on one RTX 3050 6GB laptop.

> Status: **M0 — setup**. Results: TBD (every number will link to `experiments/`).

## Quickstart (Windows, PowerShell or Git Bash)
```powershell
uv sync
uv run poe check          # lint + typecheck + test
uv run poe up             # Qdrant in Docker Desktop
uv run antibros --help
```
Setup details: [docs/setup_windows.md](docs/setup_windows.md). Hardware measurements:
[reports/hardware.md](reports/hardware.md).

## Learn from this repo
Each milestone has a brief in [docs/milestones/](docs/milestones/) and a git tag `m0` … `m6`.
Design decisions live in [docs/adr/](docs/adr/).
