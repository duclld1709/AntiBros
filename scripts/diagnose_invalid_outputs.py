"""Summarise invalid structured outputs stored in the LLM cache (M5 error analysis).

The pipeline caches the raw text of every LLM call, valid or not. This script scans the
cache, keeps responses whose text is not valid JSON, and reports, per model and max_tokens,
how many hit the output limit (``completion_tokens >= max_tokens``) or came close to the
context window (``prompt + completion >= num_ctx``). Read-only; no model is called.

    uv run python scripts/diagnose_invalid_outputs.py [--cache .llm_cache] [--model qwen3.5:4b]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def is_json(text: str) -> bool:
    try:
        json.loads(text)
    except ValueError:
        return False
    return True


def scan(cache: Path, model: str | None) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(cache.rglob("*.json")):
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        key, value = entry.get("key", {}), entry.get("value", {})
        if model and key.get("model") != model:
            continue
        params = key.get("params", {})
        text = str(value.get("text", ""))
        usage = value.get("usage", {})
        rows.append(
            {
                "model": key.get("model"),
                "max_tokens": params.get("max_tokens"),
                "num_ctx": params.get("num_ctx"),
                "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                "completion_tokens": int(usage.get("completion_tokens") or 0),
                "valid": is_json(text),
                "tail": text[-80:].replace("\n", " "),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cache", type=Path, default=Path(".llm_cache"))
    parser.add_argument("--model", default=None)
    parser.add_argument("--examples", type=int, default=5)
    args = parser.parse_args()
    rows = scan(args.cache, args.model)
    invalid = [r for r in rows if not r["valid"]]
    print(f"{len(rows)} cached responses, {len(invalid)} not valid JSON")
    groups = Counter((r["model"], r["max_tokens"]) for r in invalid)
    for (model, max_tokens), n in sorted(groups.items(), key=lambda kv: -kv[1]):
        sub = [r for r in invalid if (r["model"], r["max_tokens"]) == (model, max_tokens)]
        at_limit = sum(1 for r in sub if max_tokens and r["completion_tokens"] >= max_tokens)
        near_ctx = sum(
            1
            for r in sub
            if r["num_ctx"] and r["prompt_tokens"] + r["completion_tokens"] >= r["num_ctx"] - 16
        )
        print(
            f"- {model} max_tokens={max_tokens}: {n} invalid; {at_limit} hit max_tokens; "
            f"{near_ctx} reached num_ctx"
        )
    for r in invalid[: args.examples]:
        print(
            f"  prompt={r['prompt_tokens']} completion={r['completion_tokens']} "
            f"max_tokens={r['max_tokens']} num_ctx={r['num_ctx']} tail={r['tail']!r}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
