import { useEffect, useState } from "react";

import { checkHealth } from "../api/client";

/** Polls `GET /health`; `null` until the first answer. */
export function useHealth(intervalMs = 15000): boolean | null {
  const [ok, setOk] = useState<boolean | null>(null);
  useEffect(() => {
    let alive = true;
    const controller = new AbortController();
    const poll = async () => {
      const healthy = await checkHealth(controller.signal);
      if (alive) setOk(healthy);
    };
    void poll();
    const timer = window.setInterval(() => void poll(), intervalMs);
    return () => {
      alive = false;
      controller.abort();
      window.clearInterval(timer);
    };
  }, [intervalMs]);
  return ok;
}
