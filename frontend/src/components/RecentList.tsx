import { History, X } from "lucide-react";

import type { Copy } from "../i18n";

interface Props {
  t: Copy;
  items: string[];
  onPick: (q: string) => void;
  onClear: () => void;
  disabled: boolean;
}

export function RecentList({ t, items, onPick, onClear, disabled }: Props) {
  if (!items.length) return null;
  return (
    <section className="recent" aria-label={t.recent}>
      <span className="recent-label">
        <History size={14} aria-hidden="true" /> {t.recent}
      </span>
      {items.map((q) => (
        <button key={q} className="recent-item" onClick={() => onPick(q)} disabled={disabled} title={q}>
          {q}
        </button>
      ))}
      <button className="link-button" onClick={onClear} aria-label={t.clear}>
        <X size={14} aria-hidden="true" /> {t.clear}
      </button>
    </section>
  );
}
