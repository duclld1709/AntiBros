# AntiBros

Evidence-based Q&A for muscle-building nutrition (protein and creatine): cited RAG with hybrid
search, reranking, citation verification, abstention and medical guardrails, evaluated on an RTX
3050 6GB laptop.

> Status: M3 baseline RAG, providers, API and UI. Run artifacts are stored under `experiments/`.

## Quickstart (Windows, PowerShell)

```powershell
uv sync
uv run antibros --help
uv run antibros index --config configs/baseline.yaml --chunker section_aware --embedder bge_m3
uv run antibros serve --config configs/baseline.yaml --models configs/models/laptop.yaml
```

The API listens on `http://127.0.0.1:8000` (`/health`, `/ask`, `/feedback`). Start the UI in a
second terminal with `uv run streamlit run ui/streamlit_app.py`. Ollama must be running with the
models listed in `configs/models/laptop.yaml`. Setup details: [docs/setup_windows.md](docs/setup_windows.md).

Run the M3 development baseline with:

```powershell
uv run antibros eval predict --config configs/baseline.yaml --models configs/models/laptop.yaml --tag m3_dev
uv run antibros eval judge --config configs/baseline.yaml --models configs/models/laptop.yaml --run experiments/<run-id>
```

See [reports/hardware.md](reports/hardware.md) for measured throughput and projections. Milestones
are documented in [docs/milestones/](docs/milestones/) and design decisions in [docs/adr/](docs/adr/).
