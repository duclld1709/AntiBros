import { Star } from "lucide-react";
import { useState } from "react";

import { sendFeedback } from "../api/client";
import type { Copy } from "../i18n";

export function FeedbackStars({ t, traceId }: { t: Copy; traceId: string }) {
  const [hover, setHover] = useState(0);
  const [rating, setRating] = useState(0);
  const [state, setState] = useState<"idle" | "sent" | "error">("idle");

  const rate = async (value: number) => {
    setRating(value);
    try {
      await sendFeedback(traceId, value);
      setState("sent");
    } catch {
      setState("error");
    }
  };

  return (
    <div className="feedback">
      <span>{t.rateTitle}</span>
      <div className="stars" role="radiogroup" aria-label={t.rateTitle} onMouseLeave={() => setHover(0)}>
        {[1, 2, 3, 4, 5].map((value) => (
          <button
            key={value}
            role="radio"
            aria-checked={rating === value}
            aria-label={`${value}/5`}
            disabled={state === "sent"}
            onMouseEnter={() => setHover(value)}
            onClick={() => void rate(value)}
            className={value <= (hover || rating) ? "filled" : ""}
          >
            <Star size={20} aria-hidden="true" />
          </button>
        ))}
      </div>
      {state === "sent" && <span className="feedback-ok">{t.thanks}</span>}
      {state === "error" && <span className="feedback-err">{t.feedbackError}</span>}
    </div>
  );
}
