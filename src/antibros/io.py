"""Cross-platform file I/O helpers.

Every text read/write goes through UTF-8 explicitly (Windows defaults to cp1252/cp1258,
which corrupts Vietnamese text), and every write is atomic: data goes to a temp file in the
same directory and is moved into place with ``os.replace``. Windows can briefly lock the
target (antivirus, indexer, an open handle), so ``os.replace`` is retried on PermissionError.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any

ENCODING = "utf-8"

_WINDOWS_FORBIDDEN_CHARS = set('<>:"/\\|?*')
_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}
_RUN_TAG_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def read_text(path: Path) -> str:
    return path.read_text(encoding=ENCODING)


def write_bytes_atomic(
    path: Path, data: bytes, *, retries: int = 5, retry_delay_s: float = 0.1
) -> None:
    """Write ``data`` to ``path`` atomically, retrying the final rename on PermissionError."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        for attempt in range(retries + 1):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                if attempt == retries:
                    raise
                time.sleep(retry_delay_s * (2**attempt))
    finally:
        tmp.unlink(missing_ok=True)


def write_text(path: Path, text: str) -> None:
    """Atomically write ``text`` as UTF-8 with LF line endings as given (no newline translation)."""
    write_bytes_atomic(path, text.encode(ENCODING))


def write_json(path: Path, obj: Any) -> None:
    write_text(path, json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(read_text(path))


def dumps_line(obj: Any) -> str:
    """One JSONL line: compact, UTF-8 friendly, stable key order."""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def read_jsonl(path: Path) -> list[Any]:
    """Read a JSONL file. A trailing partial line (interrupted append) is ignored."""
    if not path.exists():
        return []
    rows = []
    lines = read_text(path).split("\n")
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i == len(lines) - 1:  # last line without newline: interrupted write
                break
            raise
    return rows


def write_jsonl(path: Path, rows: list[Any]) -> None:
    write_text(path, "".join(dumps_line(r) + "\n" for r in rows))


def repair_jsonl_tail(path: Path) -> None:
    """Drop a trailing partial line left by an interrupted append, so appends stay valid."""
    if not path.exists():
        return
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        cut = data.rfind(b"\n") + 1
        write_bytes_atomic(path, data[:cut])


def append_jsonl(path: Path, row: Any) -> None:
    """Append one line and fsync it: a crash loses at most the line being written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding=ENCODING, newline="\n") as f:
        f.write(dumps_line(row) + "\n")
        f.flush()
        os.fsync(f.fileno())


def is_valid_windows_filename(name: str) -> bool:
    """Return True if ``name`` is a valid single path component on Windows (and thus on Linux)."""
    if not name or name in {".", ".."} or len(name) > 255:
        return False
    if any(c in _WINDOWS_FORBIDDEN_CHARS or ord(c) < 32 for c in name):
        return False
    if name[-1] in {" ", "."}:
        return False
    stem = name.split(".", 1)[0].upper()
    return stem not in _WINDOWS_RESERVED_NAMES


def is_valid_run_tag(tag: str) -> bool:
    """Run tags may only contain ``[A-Za-z0-9_-]`` (spec §4.6)."""
    return bool(_RUN_TAG_RE.fullmatch(tag))
