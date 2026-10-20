import { Dumbbell, LoaderCircle, SlidersHorizontal, Sparkles } from "lucide-react";
import { useId, useState, type FormEvent, type KeyboardEvent } from "react";

import type { Filters } from "../api/types";
import { EXAMPLES, type Copy, type Lang } from "../i18n";

const MAX_CHARS = 4000;
const YEARS = [2010, 2015, 2020];

interface Props {
  t: Copy;
  lang: Lang;
  question: string;
  onQuestion: (q: string) => void;
  filters: Filters;
  onFilters: (f: Filters) => void;
  onSubmit: (q: string) => void;
  loading: boolean;
  /** Hide the example chips once a question is under way or answered. */
  compact?: boolean;
}

export function AskPanel({ t, lang, question, onQuestion, filters, onFilters, onSubmit, loading, compact = false }: Props) {
  const [showFilters, setShowFilters] = useState(false);
  const inputId = useId();
  const canSend = question.trim().length >= 3 && !loading;

  const submit = (event?: FormEvent) => {
    event?.preventDefault();
    if (canSend) onSubmit(question.trim());
  };
  const onKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) submit();
  };
  const activeFilters = Number(filters.min_year !== undefined) + Number(filters.max_evidence_level !== undefined);
  const evidenceOptions: [number | undefined, string][] = [
    [undefined, t.evidenceAll],
    [2, t.evidenceRct],
    [1, t.evidenceMeta],
  ];

  return (
    <form className="ask-card" onSubmit={submit}>
      <label htmlFor={inputId} className="sr-only">
        {t.placeholder}
      </label>
      <textarea
        id={inputId}
        value={question}
        onChange={(e) => onQuestion(e.target.value.slice(0, MAX_CHARS))}
        onKeyDown={onKeyDown}
        placeholder={t.placeholder}
        rows={3}
        disabled={loading}
      />
      <div className="ask-toolbar">
        <button
          type="button"
          className={`chip-button ${showFilters ? "active" : ""}`}
          aria-expanded={showFilters}
          onClick={() => setShowFilters((v) => !v)}
        >
          <SlidersHorizontal size={16} aria-hidden="true" />
          {t.filters}
          {activeFilters > 0 && <span className="count-badge">{activeFilters}</span>}
        </button>
        <span className="ask-hint">{t.shortcut}</span>
        <button type="submit" className="primary-button" disabled={!canSend}>
          {loading ? (
            <LoaderCircle size={18} className="spin" aria-hidden="true" />
          ) : (
            <Dumbbell size={18} aria-hidden="true" />
          )}
          {loading ? t.asking : t.ask}
        </button>
      </div>

      {showFilters && (
        <div className="filters">
          <label className="filter">
            <span>{t.fromYear}</span>
            <select
              value={filters.min_year ?? ""}
              onChange={(e) =>
                onFilters({ ...filters, min_year: e.target.value ? Number(e.target.value) : undefined })
              }
            >
              <option value="">{t.anyYear}</option>
              {YEARS.map((y) => (
                <option key={y} value={y}>
                  ≥ {y}
                </option>
              ))}
            </select>
          </label>
          <div className="filter">
            <span>{t.evidence}</span>
            <div className="segmented" role="group" aria-label={t.evidence}>
              {evidenceOptions.map(([level, label]) => (
                <button
                  type="button"
                  key={label}
                  className={filters.max_evidence_level === level ? "active" : ""}
                  aria-pressed={filters.max_evidence_level === level}
                  onClick={() => onFilters({ ...filters, max_evidence_level: level })}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      {!compact && (
        <div className="examples">
          <span className="examples-label">
            <Sparkles size={14} aria-hidden="true" /> {t.tryAsking}
          </span>
          {EXAMPLES[lang].map((example) => (
            <button
              type="button"
              key={example}
              className="example-chip"
              disabled={loading}
              onClick={() => {
                onQuestion(example);
                onSubmit(example);
              }}
            >
              {example}
            </button>
          ))}
        </div>
      )}
    </form>
  );
}
