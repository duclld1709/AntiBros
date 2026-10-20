import { ExternalLink } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import type { Citation } from "../api/types";
import type { Copy } from "../i18n";
import { EvidenceBadge } from "./EvidenceBadge";

function SourceCard({ t, citation, active }: { t: Copy; citation: Citation; active: boolean }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLLIElement>(null);
  useEffect(() => {
    if (active) ref.current?.scrollIntoView?.({ behavior: "smooth", block: "nearest" });
  }, [active]);
  const long = citation.snippet.length > 260;
  return (
    <li ref={ref} id={`source-${citation.id}`} className={`source-card ${active ? "active" : ""}`}>
      <div className="source-top">
        <span className="source-num">{citation.id}</span>
        <EvidenceBadge level={citation.evidence_level} t={t} />
      </div>
      <h3 className="source-title">{citation.title}</h3>
      <p className="source-meta">
        {citation.journal}
        {citation.year ? ` · ${citation.year}` : ""} · <code>{citation.doc_id}</code>
      </p>
      <blockquote className={`source-snippet ${open || !long ? "open" : ""}`}>{citation.snippet}</blockquote>
      <div className="source-actions">
        {long && (
          <button className="link-button" onClick={() => setOpen((v) => !v)}>
            {open ? t.showLess : t.showMore}
          </button>
        )}
        <a href={citation.url} target="_blank" rel="noreferrer" className="link-button">
          {t.openPaper} <ExternalLink size={14} aria-hidden="true" />
        </a>
      </div>
    </li>
  );
}

export function SourcesPanel({ t, citations, active }: { t: Copy; citations: Citation[]; active: number | null }) {
  if (!citations.length) return <p className="empty-note">{t.noSources}</p>;
  return (
    <ol className="sources">
      {citations.map((c) => (
        <SourceCard key={c.id} t={t} citation={c} active={active === c.id} />
      ))}
    </ol>
  );
}
