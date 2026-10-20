# AntiBros

Evidence-based Q&A for muscle-building nutrition (protein and creatine): cited RAG with hybrid
search, reranking, citation verification, abstention and medical guardrails, evaluated on an RTX
3050 6GB laptop.

> Status: M4 dev runs complete and the final evaluation plan is locked. Retrieval and guard
> ablations ran on the RTX 3050; threshold tuning and E2E leave-one-out ran on a Linux V100.
> The `m4` tag is not created. Run artifacts are stored under `experiments/`.

## Quickstart (Windows, PowerShell)

```powershell
uv sync
uv run antibros --help
uv run antibros index --config configs/baseline.yaml --chunker section_aware --embedder bge_m3
uv run antibros serve --config configs/baseline.yaml --models configs/models/laptop.yaml
```

The API listens on `http://127.0.0.1:8000` (`/health`, `/ask`, `/feedback`). The web UI (React +
TypeScript, `frontend/`, Node.js 22) is served on the same address once built with
`uv run poe ui-build`; for UI work run `uv run poe ui` (Vite dev server on `http://localhost:5173`,
proxied to the API). Ollama must be running with the models listed in `configs/models/laptop.yaml`. Setup details: [docs/setup_windows.md](docs/setup_windows.md).

Run the M3 development baseline with:

```powershell
uv run antibros eval predict --config configs/baseline.yaml --models configs/models/laptop.yaml --tag m3_dev
uv run antibros eval judge --models configs/models/laptop.yaml --run experiments/<run-id>
```

The full pipeline is `configs/full.yaml` (baseline + every M4 step). Ablation configs live in
`configs/ablations/` and inherit from it with `extends:`; the order of the M4 dev runs is in
[docs/runbooks/m4_dev_runs.md](docs/runbooks/m4_dev_runs.md). The test split runs only through
the pre-registered plan `configs/final_eval_plan.yaml` (ADR-0014).

See [reports/hardware.md](reports/hardware.md) for measured throughput and projections. Milestones
are documented in [docs/milestones/](docs/milestones/) and design decisions in [docs/adr/](docs/adr/).
