# AntiBros web UI

React 19 + TypeScript single-page app built with Vite (ADR-0022). It talks only to the
FastAPI endpoints `/ask`, `/health` and `/feedback`, on the same origin.

| Task | From the repo root | Inside `frontend/` |
|---|---|---|
| Dev server (http://localhost:5173, proxies the API to `ANTIBROS_API_URL`, default `http://127.0.0.1:8000`) | `uv run poe ui` | `npm run dev` |
| Production build → `dist/` (served by `uv run antibros serve` at http://127.0.0.1:8000) | `uv run poe ui-build` | `npm run build` |
| Lint + typecheck + unit tests | `uv run poe ui-check` | `npm run lint && npm run typecheck && npm test` |

Layout: `src/api/` (typed client), `src/components/` (UI), `src/i18n.ts` (Vietnamese/English
copy and example questions, written for the UI and never taken from the evalset; `antibros
evalset leakcheck` scans this folder), `src/styles/app.css` (design tokens, light/dark themes).
Tests use Vitest + Testing Library with a mocked `fetch`; they never call the real API.
