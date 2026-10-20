"""On-disk LLM response cache.

The key holds provider, model, digest, prompt hash and params. File names are the SHA-256
of the canonical key JSON: keys contain characters that are illegal in Windows file names
(``:`` in ``qwen3.5:4b``) and are unbounded in length. The full key is stored inside the
file and compared on read, so a hash collision can never return the wrong result.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from antibros.io import read_json, write_json


def _canonical(key: dict[str, Any]) -> str:
    return json.dumps(key, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class LLMCache:
    def __init__(self, root: Path):
        self.root = root

    def path_for(self, key: dict[str, Any]) -> Path:
        h = hashlib.sha256(_canonical(key).encode("utf-8")).hexdigest()
        return self.root / h[:2] / f"{h}.json"

    def get(self, key: dict[str, Any]) -> dict[str, Any] | None:
        path = self.path_for(key)
        if not path.exists():
            return None
        entry = read_json(path)
        if _canonical(entry.get("key", {})) != _canonical(key):
            return None
        value: dict[str, Any] = entry["value"]
        return value

    def put(self, key: dict[str, Any], value: dict[str, Any]) -> None:
        write_json(self.path_for(key), {"key": key, "value": value})
