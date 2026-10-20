import { CheckCircle2, Clock, Info, RotateCcw, SearchX, ShieldAlert, ShieldCheck } from "lucide-react";
import type { ReactNode } from "react";

import type { AskResponse, Confidence, Status } from "../api/types";
import type { Copy } from "../i18n";
import { parseAnswer } from "../lib/citations";
import { FeedbackStars } from "./FeedbackStars";

const STATUS_ICON: Record<Status, ReactNode> = {
  answered: <CheckCircle2 size={18} aria-hidden="true" />,
  abstained: <SearchX size={18} aria-hidden="true" />,
  safety_redirect: <ShieldAlert size={18} aria-hidden="true" />,
};

const CONFIDENCE_BARS: Record<Confidence, number> = { low: 1, moderate: 2, high: 3 };

interface Props {
  t: Copy;
  result: AskResponse;
  question: string;
  activeCitation: number | null;
  onCitation: (id: number) => void;
  onReset: () => void;
}

export function AnswerCard({ t, result, question, activeCitation, onCitation, onReset }: Props) {
  const label = {
    answered: t.statusAnswered,
    abstained: t.statusAbstained,
    safety_redirect: t.statusSafety,
  }[result.status];
  const confidence = { high: t.confHigh, moderate: t.confModerate, low: t.confLow }[result.confidence];
  const known = new Set(result.citations.map((c) => c.id));
  const lines = parseAnswer(result.answer);

  return (
    <article className={`answer-card status-${result.status}`} aria-live="polite">
      <header className="answer-head">
        <span className={`status-badge status-${result.status}`}>
          {STATUS_ICON[result.status]}
          {label}
        </span>
        {result.status === "answered" && (
          <span className="confidence" title={t.confidence}>
            <span className="confidence-bars" aria-hidden="true">
              {[1, 2, 3].map((bar) => (
                <span key={bar} className={bar <= CONFIDENCE_BARS[result.confidence] ? "on" : ""} />
              ))}
            </span>
            {t.confidence}: {confidence}
          </span>
        )}
        <span className="latency">
          <Clock size={14} aria-hidden="true" /> {(result.latency_ms / 1000).toFixed(1)} s
        </span>
      </header>

      <p className="answer-question">{question}</p>

      <div className="answer-body">
        {lines.map((segments, i) => (
          <p key={i} className={result.status === "answered" ? "claim" : ""}>
            {segments.map((segment, j) =>
              segment.kind === "text" ? (
                <span key={j}>{segment.text}</span>
              ) : known.has(segment.id) ? (
                <button
                  key={j}
                  className={`cite-chip ${activeCitation === segment.id ? "active" : ""}`}
                  onClick={() => onCitation(segment.id)}
                  aria-label={`${t.sources} ${segment.id}`}
                >
                  {segment.id}
                </button>
              ) : (
                <span key={j}>[{segment.id}]</span>
              ),
            )}
          </p>
        ))}
      </div>

      {result.unsupported_claims_removed > 0 && (
        <p className="note">
          <Info size={16} aria-hidden="true" /> {t.removedClaims(result.unsupported_claims_removed)}
        </p>
      )}
      {result.disclaimer && (
        <p className="disclaimer">
          <ShieldCheck size={16} aria-hidden="true" /> {result.disclaimer}
        </p>
      )}

      <footer className="answer-foot">
        <FeedbackStars key={result.trace_id} t={t} traceId={result.trace_id} />
        <button className="ghost-button" onClick={onReset}>
          <RotateCcw size={16} aria-hidden="true" /> {t.newQuestion}
        </button>
      </footer>
    </article>
  );
}
