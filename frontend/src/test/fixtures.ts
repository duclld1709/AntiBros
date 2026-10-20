import type { AskResponse } from "../api/types";

export function answered(overrides: Partial<AskResponse> = {}): AskResponse {
  return {
    answer: "Creatine increases lean mass [1]\nHigher intakes help up to a point. [1][2]",
    status: "answered",
    language: "en",
    citations: [
      {
        id: 1,
        doc_id: "PMC1",
        chunk_id: "PMC1::results::001",
        title: "A meta-analysis of creatine",
        year: 2023,
        journal: "Nutrients",
        url: "https://europepmc.org/article/PMC/PMC1",
        evidence_level: 1,
        evidence_label: "Meta-analysis / Systematic review",
        snippet: "Creatine supplementation increased lean mass.",
      },
      {
        id: 2,
        doc_id: "PMC2",
        chunk_id: "PMC2::discussion::004",
        title: "A randomized trial",
        year: 2017,
        journal: "PeerJ",
        url: "https://europepmc.org/article/PMC/PMC2",
        evidence_level: 2,
        evidence_label: "RCT",
        snippet: "Participants were randomly assigned.",
      },
    ],
    confidence: "high",
    unsupported_claims_removed: 1,
    disclaimer: "For information only.",
    trace_id: "trace-1",
    latency_ms: 4200,
    debug: {
      retrieved: [
        { chunk_id: "PMC1::results::001", score: 0.03, rerank_score: 0.91, text: "Creatine…" },
        { chunk_id: "PMC2::discussion::004", score: 0.02, rerank_score: null, text: "Randomly…" },
      ],
      steps: [
        { node: "input_guard", ms: 12, label: "safe" },
        { node: "retrieve", ms: 140, hits: [["PMC1::results::001", 0.03]] },
        { node: "grade_context", ms: 900, sufficient: true },
      ],
    },
    ...overrides,
  };
}
