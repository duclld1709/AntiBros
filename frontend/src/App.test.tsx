import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App";
import { answered } from "./test/fixtures";

function mockFetch(handler: (url: string, init?: RequestInit) => Response | Promise<Response>) {
  const fn = vi.fn((input: RequestInfo | URL, init?: RequestInit) =>
    Promise.resolve(handler(String(input), init)),
  );
  vi.stubGlobal("fetch", fn);
  return fn;
}

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

beforeEach(() => {
  localStorage.setItem("antibros.lang", JSON.stringify("en"));
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("App", () => {
  it("asks a question, shows the answer and highlights a cited source", async () => {
    const fetchMock = mockFetch((url) => (url === "/ask" ? json(answered()) : json({ status: "ok" })));
    render(<App />);
    await userEvent.type(screen.getByRole("textbox"), "Does creatine build muscle?");
    await userEvent.click(screen.getByRole("button", { name: "Ask" }));

    expect(await screen.findByText("Answered")).toBeInTheDocument();
    const askCall = fetchMock.mock.calls.find(([url]) => String(url) === "/ask");
    expect(JSON.parse(String(askCall?.[1]?.body))).toEqual({
      question: "Does creatine build muscle?",
      filters: null,
    });
    expect(screen.getByText("A meta-analysis of creatine")).toBeInTheDocument();

    await userEvent.click(screen.getAllByRole("button", { name: "Sources 2" })[0]!);
    expect(document.getElementById("source-2")).toHaveClass("active");

    await userEvent.click(screen.getByRole("tab", { name: /Pipeline/ }));
    expect(screen.getByText("grade context")).toBeInTheDocument();
    expect(screen.getByText("Rerank 0.910")).toBeInTheDocument();

    // the question is remembered under "Recent"
    const recent = screen.getByRole("region", { name: "Recent" });
    expect(within(recent).getByText("Does creatine build muscle?")).toBeInTheDocument();
  });

  it("sends active filters", async () => {
    const fetchMock = mockFetch((url) => (url === "/ask" ? json(answered()) : json({ status: "ok" })));
    render(<App />);
    await userEvent.click(screen.getByRole("button", { name: /Filters/ }));
    await userEvent.selectOptions(screen.getByRole("combobox"), "2015");
    await userEvent.click(screen.getByRole("button", { name: "Meta-analyses only" }));
    await userEvent.type(screen.getByRole("textbox"), "Protein per day?");
    await userEvent.click(screen.getByRole("button", { name: "Ask" }));
    await screen.findByText("Answered");
    const askCall = fetchMock.mock.calls.find(([url]) => String(url) === "/ask");
    expect(JSON.parse(String(askCall?.[1]?.body)).filters).toEqual({
      min_year: 2015,
      max_evidence_level: 1,
    });
  });

  it("explains how to start the API when it is unreachable", async () => {
    mockFetch(() => {
      throw new TypeError("Failed to fetch");
    });
    render(<App />);
    await userEvent.click(screen.getAllByRole("button", { name: /Fact-check/ })[0]!);
    expect(await screen.findByRole("alert")).toHaveTextContent("antibros serve");
    await waitFor(() => expect(screen.getByText("API not running")).toBeInTheDocument());
  });

  it("surfaces API errors with their status", async () => {
    mockFetch((url) =>
      url === "/ask" ? json({ detail: "RAG pipeline unavailable" }, 503) : json({ status: "ok" }),
    );
    render(<App />);
    await userEvent.type(screen.getByRole("textbox"), "Is whey safe?");
    await userEvent.keyboard("{Control>}{Enter}{/Control}");
    expect(await screen.findByRole("alert")).toHaveTextContent("(503): RAG pipeline unavailable");
  });

  it("switches the interface language", async () => {
    mockFetch(() => json({ status: "ok" }));
    render(<App />);
    await userEvent.click(screen.getByRole("button", { name: "VI" }));
    expect(screen.getByRole("button", { name: "Hỏi" })).toBeInTheDocument();
    expect(document.documentElement.lang).toBe("vi");
  });
});
