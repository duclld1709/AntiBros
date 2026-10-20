# ADR-0005: Orchestrate the pipeline with LangGraph

- **Status:** Proposed
- **Date:** TBD
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
Proposed: **LangGraph**, with nodes as plain, unit-testable functions so the graph can be replaced if needed.

## Evidence
TBD — code complexity, per-node trace availability, test coverage of nodes.

## Consequences
- **Positive:** the flow in the architecture diagram maps 1:1 to code.
- **Negative:** dependency on a library with frequent API changes (mitigated by pinning versions).
- **Revisit if:** the graph stays linear and the dependency adds no value.
