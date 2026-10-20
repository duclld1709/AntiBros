import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { COPY } from "../i18n";
import { answered } from "../test/fixtures";
import { AnswerCard } from "./AnswerCard";

const t = COPY.en;

describe("AnswerCard", () => {
  it("renders claims with clickable citation chips and the disclaimer", async () => {
    const onCitation = vi.fn();
    render(
      <AnswerCard
        t={t}
        result={answered()}
        question="Does creatine work?"
        activeCitation={null}
        onCitation={onCitation}
        onReset={() => {}}
      />,
    );
    expect(screen.getByText("Answered")).toBeInTheDocument();
    expect(screen.getByText("Does creatine work?")).toBeInTheDocument();
    expect(screen.getByText("For information only.")).toBeInTheDocument();
    expect(screen.getByText("1 unsupported claim removed")).toBeInTheDocument();
    const chips = screen.getAllByRole("button", { name: "Sources 1" });
    expect(chips).toHaveLength(2);
    await userEvent.click(chips[0]!);
    expect(onCitation).toHaveBeenCalledWith(1);
  });

  it("shows unknown markers as plain text, never as a link", () => {
    render(
      <AnswerCard
        t={t}
        result={answered({ answer: "Claim [7]" })}
        question="q"
        activeCitation={null}
        onCitation={() => {}}
        onReset={() => {}}
      />,
    );
    expect(screen.queryByRole("button", { name: "Sources 7" })).not.toBeInTheDocument();
    expect(screen.getByText("[7]")).toBeInTheDocument();
  });

  it.each([
    ["abstained", "Insufficient evidence"],
    ["safety_redirect", "Ask a professional"],
  ] as const)("labels status %s", (status, label) => {
    render(
      <AnswerCard
        t={t}
        result={answered({ status, citations: [], answer: "No answer.", disclaimer: null })}
        question="q"
        activeCitation={null}
        onCitation={() => {}}
        onReset={() => {}}
      />,
    );
    expect(screen.getByText(label)).toBeInTheDocument();
    expect(screen.queryByText(/Confidence/)).not.toBeInTheDocument();
  });
});
