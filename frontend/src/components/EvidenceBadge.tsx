import type { Copy } from "../i18n";

/** Evidence level 1 (meta-analysis) … 5 (other), spec §3.4. */
export function EvidenceBadge({ level, t }: { level: number; t: Copy }) {
  const safe = level >= 1 && level <= 5 ? level : 5;
  return (
    <span className={`evidence-badge level-${safe}`} title={`Level ${safe}`}>
      <span className="evidence-bars" aria-hidden="true">
        {[1, 2, 3, 4].map((bar) => (
          <span key={bar} className={bar <= 5 - safe ? "on" : ""} />
        ))}
      </span>
      {t.evidenceLevels[safe]}
    </span>
  );
}

export function EvidenceLadder({ t }: { t: Copy }) {
  return (
    <div className="ladder" aria-label={t.pyramid}>
      <span className="ladder-title">{t.pyramid}</span>
      {[1, 2, 3, 4, 5].map((level) => (
        <EvidenceBadge key={level} level={level} t={t} />
      ))}
    </div>
  );
}
