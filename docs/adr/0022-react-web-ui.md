# ADR-0022: Replace the Streamlit UI with a React (Vite) single-page app

- **Status:** Accepted
- **Date:** 2026-10-20
- **Milestone:** M4 (UI upgrade requested by the project owner)

## Context
The M3 Streamlit page proved the end-to-end flow but offered little control over layout and
interaction: citation markers could not be clicked to reveal their source, the debug view was
a raw JSON dump, and theming was limited. The owner asked for a friendlier, more polished UI
with a fitness theme, built with React or Next.js. The UI only calls the existing FastAPI
endpoints (`/ask`, `/health`, `/feedback`); it has no server-side logic of its own.

## Options considered
| Option | Pros | Cons |
|---|---|---|
| Keep Streamlit, add custom CSS | No new toolchain | Limited interaction; fights the framework's layout |
| Next.js | File routing, SSR, image optimisation | A second server (Node) to run next to FastAPI; SSR brings nothing to a single page that calls a local API |
| **React + Vite (TypeScript), static build served by FastAPI** | One process in production; same-origin API, no CORS; fast dev server with proxy; small dependency set | Node.js needed to build and develop the UI |

## Decision
Use **React 19 + TypeScript with Vite** in `frontend/`. `antibros serve` mounts the built
`frontend/dist` at `/` after the API routes, so `uv run antibros serve` still serves everything
on one port. During UI work, the Vite dev server proxies the API paths to `antibros serve`.
Plain CSS with design tokens (dark/light themes, volt-green accent), self-hosted fonts
(`@fontsource`, so the UI works offline), and `lucide-react` icons; no CSS framework.
`scripts/frontend.py` runs npm the same way on Windows and Linux (`uv run poe ui`,
`ui-build`, `ui-check`). CI gets a `frontend` job (ESLint, `tsc`, Vitest, build).
`leakcheck` now also scans `frontend/src` (`.ts`/`.tsx`), because the UI shows example
questions.

## Evidence
Frontend unit tests (`frontend/src/**/*.test.ts*`, mocked `fetch`) and
`tests/unit/test_m4_tooling.py::test_api_serves_built_ui_without_shadowing_api_routes`. No
experiment numbers are involved.

## Consequences
- **Positive:** clickable citations linked to source cards with evidence-level badges, a
  readable pipeline timeline (per-step latency, rerank scores), bilingual copy, light/dark
  themes, responsive layout down to phone width.
- **Negative / risks:** a second toolchain (Node.js, `package-lock.json`). The `streamlit`
  dependency remains in `pyproject.toml` until `uv remove streamlit` is run on a machine that
  can reach the PyTorch wheel index (the lock cannot be regenerated without it).
- **Revisit if:** the UI needs server-side features (auth, streaming via a BFF), where a
  framework with a server (e.g. Next.js) may pay off.
