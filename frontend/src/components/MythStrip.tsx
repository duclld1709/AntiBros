import { Zap } from "lucide-react";

import { MYTHS, type Copy, type Lang } from "../i18n";

interface Props {
  t: Copy;
  lang: Lang;
  onPick: (question: string) => void;
  disabled: boolean;
}

export function MythStrip({ t, lang, onPick, disabled }: Props) {
  return (
    <section className="myths" aria-labelledby="myth-title">
      <div className="section-head">
        <h2 id="myth-title">{t.mythTitle}</h2>
        <p>{t.mythSubtitle}</p>
      </div>
      <div className="myth-grid">
        {MYTHS[lang].map(({ myth, question }, i) => (
          <button
            key={myth}
            className="myth-card"
            style={{ animationDelay: `${i * 80}ms` }}
            disabled={disabled}
            onClick={() => onPick(question)}
          >
            <span className="myth-icon" aria-hidden="true">
              <Zap size={18} />
            </span>
            <span className="myth-text">“{myth}”</span>
            <span className="myth-cta">{t.mythCta}</span>
          </button>
        ))}
      </div>
    </section>
  );
}
