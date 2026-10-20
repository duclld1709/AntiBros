import { describe, expect, it } from "vitest";

import { parseAnswer, parseLine } from "./citations";

describe("parseLine", () => {
  it("splits text and code-numbered markers", () => {
    expect(parseLine("Creatine helps [1][2] a lot.")).toEqual([
      { kind: "text", text: "Creatine helps " },
      { kind: "cite", id: 1 },
      { kind: "cite", id: 2 },
      { kind: "text", text: " a lot." },
    ]);
  });

  it("keeps plain text and Vietnamese diacritics", () => {
    expect(parseLine("Chưa đủ bằng chứng.")).toEqual([{ kind: "text", text: "Chưa đủ bằng chứng." }]);
  });
});

describe("parseAnswer", () => {
  it("returns one paragraph per non-empty line", () => {
    const lines = parseAnswer("A [1]\n\n  B [2]  \n");
    expect(lines).toHaveLength(2);
    expect(lines[1]).toEqual([
      { kind: "text", text: "B " },
      { kind: "cite", id: 2 },
    ]);
  });
});
