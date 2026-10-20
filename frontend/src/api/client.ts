import type { AskResponse, Filters } from "./types";

export class ApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function postJson<T>(path: string, body: unknown, signal?: AbortSignal): Promise<T> {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const data = (await response.json()) as { detail?: unknown };
      if (typeof data.detail === "string") detail = data.detail;
    } catch {
      // keep statusText
    }
    throw new ApiError(response.status, detail);
  }
  return (await response.json()) as T;
}

export function ask(question: string, filters: Filters, signal?: AbortSignal): Promise<AskResponse> {
  const active = Object.fromEntries(
    Object.entries(filters).filter(([, value]) => value !== undefined),
  );
  return postJson<AskResponse>(
    "/ask",
    { question, filters: Object.keys(active).length ? active : null },
    signal,
  );
}

export function sendFeedback(traceId: string, rating: number, comment = ""): Promise<unknown> {
  return postJson("/feedback", { trace_id: traceId, rating, comment });
}

export async function checkHealth(signal?: AbortSignal): Promise<boolean> {
  try {
    const response = await fetch("/health", { signal });
    return response.ok;
  } catch {
    return false;
  }
}
