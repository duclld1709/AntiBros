import type { AskResponse, Step } from "../api/types";
import type { Copy } from "../i18n";
import { nodeKey } from "../lib/citations";

/** Short, human-readable details of a trace span (the full span is in the trace file). */
function stepDetail(step: Step): string | null {
  const parts: string[] = [];
  if (typeof step.label === "string") parts.push(String(step.label));
  if (Array.isArray(step.queries)) parts.push(step.queries.map(String).join(" · "));
  if (Array.isArray(step.hits)) parts.push(`${step.hits.length} hits`);
  if (typeof step.sufficient === "boolean") parts.push(step.sufficient ? "sufficient" : "insufficient");
  if (typeof step.n_claims === "number") parts.push(`${step.n_claims} claims`);
  if (typeof step.removed === "number" && step.removed > 0) parts.push(`${step.removed} removed`);
  if (typeof step.blocked === "string") parts.push(`blocked: ${step.blocked}`);
  return parts.length ? parts.join(" — ") : null;
}

export function PipelinePanel({ t, result }: { t: Copy; result: AskResponse }) {
  const { steps, retrieved } = result.debug;
  const longest = Math.max(1, ...steps.map((s) => s.ms));
  const maxScore = Math.max(1e-9, ...retrieved.map((r) => r.score));
  return (
    <div className="pipeline">
      <h3>{t.stepsTitle}</h3>
      <ol className="timeline">
        {steps.map((step, i) => {
          const detail = stepDetail(step);
          return (
            <li key={`${step.node}-${i}`}>
              <div className="timeline-row">
                <span className="timeline-node">{nodeKey(step.node)}</span>
                <span className="timeline-ms">{Math.round(step.ms)} ms</span>
              </div>
              <div className="timeline-track" aria-hidden="true">
                <span style={{ width: `${Math.max(2, (step.ms / longest) * 100)}%` }} />
              </div>
              {detail && <p className="timeline-detail">{detail}</p>}
            </li>
          );
        })}
      </ol>

      {retrieved.length > 0 && (
        <>
          <h3>{t.retrievedTitle}</h3>
          <ul className="chunks">
            {retrieved.map((chunk, i) => (
              <li key={chunk.chunk_id}>
                <div className="chunk-head">
                  <span className="chunk-rank">#{i + 1}</span>
                  <code className="chunk-id">{chunk.chunk_id}</code>
                </div>
                <div className="chunk-scores">
                  <span>
                    {t.score} {chunk.score.toFixed(3)}
                  </span>
                  <span className="score-track" aria-hidden="true">
                    <span style={{ width: `${(chunk.score / maxScore) * 100}%` }} />
                  </span>
                  {chunk.rerank_score !== null && (
                    <span className="rerank">
                      {t.rerank} {chunk.rerank_score.toFixed(3)}
                    </span>
                  )}
                </div>
                <p className="chunk-text">{chunk.text}</p>
              </li>
            ))}
          </ul>
        </>
      )}
    </div>
  );
}
