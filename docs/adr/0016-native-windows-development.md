# ADR-0016: Native Windows as the development environment

- **Status:** Accepted
- **Date:** 2026-10-20
- **Milestone:** M0
- **Deciders:** project owner
- **Supersedes:** earlier assumption "Windows + WSL2, code runs inside Linux/WSL2"

## Context
The only machine is a Windows laptop with an RTX 3050 (6GB). The owner decided to develop, run and evaluate directly on Windows rather than inside WSL2. CI runners and the Docker images are Linux, so the code must behave identically on both operating systems. This matters in particular for checksums, character offsets and Vietnamese text.

## Options considered
| Option | Pros | Cons |
|---|---|---|
| WSL2 (Linux inside Windows) | Same OS as CI and Docker; Unix tooling (`make`, bash) | Extra layer to install and debug; files and GPU shared across a VM boundary; not what the owner wants |
| **Native Windows, cross-platform code, CI on Windows + Linux** | Owner works in a familiar environment; Ollama for Windows uses the GPU directly; portability is enforced by CI | Must avoid Unix-only tooling; Windows pitfalls (encoding, CRLF, file locks, path limits, sysmem fallback) |
| Develop on Windows, run evaluations in Docker only | One runtime for eval | GPU passthrough and a 6GB budget inside containers add friction; slower iteration |

## Decision
- **Development and all evaluation run natively on Windows.** Tools:
  - `uv` for Python and dependencies;
  - Ollama for Windows;
  - Qdrant in Docker Desktop, with embedded client mode as a fallback and for tests;
  - PowerShell for the owner, Git Bash for Claude Code.
- **No Unix-only tooling in the dev workflow:**
  - Tasks run through `poethepoet` (`uv run poe …`) instead of `make`.
  - Utility scripts are Python, not `.sh`/`.ps1`. Only scripts that run inside Linux containers stay shell.
- **PyTorch:** CUDA wheels on Windows and CPU wheels on Linux CI, selected by platform markers in `[tool.uv.sources]`; one `uv.lock`.
- **Cross-platform coding rules** (spec §4.6):
  - `pathlib` only;
  - explicit UTF-8 everywhere;
  - `.gitattributes` with `eol=lf` and `evalsets/** -text`, and checksums on bytes;
  - Windows-safe file names (no `:`);
  - no `shell=True` and no `SIGALRM`;
  - `spawn`-safe entry points;
  - atomic writes with retry on `PermissionError`.
- **CI runs the test suite on `windows-latest` and `ubuntu-latest`.** Docker jobs stay on Linux. Docker images remain Linux, because that is the deployment target.
- **NVIDIA driver:** set *CUDA – Sysmem Fallback Policy* to *Prefer No Sysmem Fallback*, so running out of VRAM fails loudly instead of silently spilling into system RAM and breaking the evaluation time budget (verified at M0).

## Evidence
`reports/hardware.md` (M0), CI matrix results, cross-platform checksum and offset tests (M1, M2).

## Consequences
- **Positive:**
  - No VM layer.
  - The GPU is used directly by Ollama and PyTorch.
  - Portability is tested on every commit.
  - Learners on Windows can follow the milestones without installing WSL.
- **Negative:**
  - Windows pitfalls must be handled explicitly.
  - Docker Desktop is still needed for Qdrant and the one-command demo, though not for development if embedded mode is used.
  - Some community tooling assumes Unix.
- **Revisit if:** a dependency stops supporting Windows, or Docker Desktop is unavailable. In that case switch to embedded Qdrant, or reconsider WSL2 with a new ADR.
