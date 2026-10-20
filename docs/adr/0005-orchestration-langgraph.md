# ADR-0005: Orchestrate the pipeline with LangGraph

- **Status:** Accepted
- **Date:** 2026-10-20
- **Milestone:** M3

## Context
The pipeline has branches (safety redirect, abstain) and a loop (rewrite and retrieve again at most once). We want each step traceable as a span and easy to test in isolation.

## Options considered
| Option | Pros | Cons |
|---|---|---|
| Plain Python functions | No dependency; full control; easiest to debug | Branching/loop and state handling written by hand; tracing per step is manual |
| LangGraph | Explicit state graph; conditional edges and loops; good tracing integrations | Extra dependency; fast-changing API |
| LangChain chains (LCEL) | Concise for linear flows | Awkward for loops and conditional routing |

## Decision
**LangGraph**, with nodes as plain, unit-testable methods so the graph can be replaced if needed. Since M4 the compiled graph is built from config: a disabled step is not a node (ADR-0021).

## Evidence
- The pipeline has 10 node types, 4 conditional edges and the single grade retry loop (`src/antibros/graph/pipeline.py`).
- Every node writes a trace span (`steps` in each prediction, `spans` in the API trace). These spans drove the M5 error analysis.
- Each node and each toggle is covered by `tests/unit/test_m4_pipeline.py`.
- LangGraph reducers merge LLM accounting, so no request state lives on the pipeline object.

## Consequences
- **Positive:** the flow in the architecture diagram maps 1:1 to code.
- **Negative:** dependency on a library with frequent API changes (mitigated by pinning versions).
- **Revisit if:** the graph stays linear and the dependency adds no value.
