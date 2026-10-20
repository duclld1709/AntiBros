/** Shapes of the FastAPI endpoints in `src/antibros/api.py` (spec §5.4). */

export type Status = "answered" | "abstained" | "safety_redirect";
export type Confidence = "high" | "moderate" | "low";

export interface Citation {
  id: number;
  doc_id: string;
  chunk_id: string;
  title: string;
  year: number | null;
  journal: string;
  url: string;
  evidence_level: number;
  evidence_label: string;
  snippet: string;
}

/** One pipeline node's trace span: `node`, `ms` and node-specific details. */
export interface Step {
  node: string;
  ms: number;
  [detail: string]: unknown;
}

export interface RetrievedChunk {
  chunk_id: string;
  score: number;
  rerank_score: number | null;
  text: string;
}

export interface AskResponse {
  answer: string;
  status: Status;
  language: string;
  citations: Citation[];
  confidence: Confidence;
  unsupported_claims_removed: number;
  disclaimer: string | null;
  trace_id: string;
  latency_ms: number;
  debug: { retrieved: RetrievedChunk[]; steps: Step[] };
}

export interface Filters {
  min_year?: number;
  max_evidence_level?: number;
}
