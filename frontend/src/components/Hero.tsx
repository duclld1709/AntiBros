import { BookOpenCheck, Quote, ShieldQuestion } from "lucide-react";

import type { Copy } from "../i18n";

export function Hero({ t }: { t: Copy }) {
  return (
    <section className="hero">
      <h1>
        {t.heroTitleA} <span className="highlight">{t.heroTitleB}</span>
      </h1>
      <p className="hero-subtitle">{t.heroSubtitle}</p>
      <ul className="feature-pills">
        <li>
          <BookOpenCheck size={16} aria-hidden="true" /> {t.featureCorpus}
        </li>
        <li>
          <Quote size={16} aria-hidden="true" /> {t.featureCited}
        </li>
        <li>
          <ShieldQuestion size={16} aria-hidden="true" /> {t.featureAbstain}
        </li>
      </ul>
    </section>
  );
}
