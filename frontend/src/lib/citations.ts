/** The API renders an answer as one claim per line with code-numbered markers `[n]`. */

export type Segment = { kind: "text"; text: string } | { kind: "cite"; id: number };

const MARKER = /\[(\d+)\]/g;

export function parseLine(line: string): Segment[] {
  const out: Segment[] = [];
  let last = 0;
  for (const match of line.matchAll(MARKER)) {
    const index = match.index ?? 0;
    if (index > last) out.push({ kind: "text", text: line.slice(last, index) });
    out.push({ kind: "cite", id: Number(match[1]) });
    last = index + match[0].length;
  }
  if (last < line.length) out.push({ kind: "text", text: line.slice(last) });
  return out;
}

export function parseAnswer(answer: string): Segment[][] {
  return answer
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .map(parseLine);
}

/** Human label for a pipeline node name such as `grade_context`. */
export function nodeKey(node: string): string {
  return node.replace(/_/g, " ");
}
