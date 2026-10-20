import { Check } from "lucide-react";
import { useEffect, useState } from "react";

import type { Copy } from "../i18n";

/** Shown while `/ask` runs. The API answers in one response, so the stages are an
 * illustration of the pipeline order, not live progress. */
export function LoadingWorkout({ t }: { t: Copy }) {
  const stages = [t.stageGuard, t.stageSearch, t.stageRank, t.stageGrade, t.stageWrite, t.stageVerify];
  const [active, setActive] = useState(0);
  useEffect(() => {
    const timer = window.setInterval(
      () => setActive((i) => Math.min(i + 1, stages.length - 1)),
      2200,
    );
    return () => window.clearInterval(timer);
  }, [stages.length]);

  return (
    <section className="loading-card" aria-live="polite" aria-busy="true">
      <div className="barbell" aria-hidden="true">
        <span className="plate plate-l" />
        <span className="bar" />
        <span className="plate plate-r" />
      </div>
      <h2>{t.loadingTitle}</h2>
      <ol className="stages">
        {stages.map((stage, i) => (
          <li key={stage} className={i < active ? "done" : i === active ? "current" : ""}>
            <span className="stage-dot" aria-hidden="true">
              {i < active ? <Check size={12} /> : i + 1}
            </span>
            {stage}
          </li>
        ))}
      </ol>
    </section>
  );
}
