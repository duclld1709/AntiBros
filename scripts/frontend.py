"""Run the web UI's npm scripts from the repo root, the same way on Windows and Linux.

``uv run poe ui`` (dev server), ``uv run poe ui-build`` (production build served by
``antibros serve``), ``uv run poe ui-check`` (lint + typecheck + tests).

On Windows ``npm`` is ``npm.cmd``, which ``subprocess`` cannot start by bare name without a
shell; ``shutil.which`` resolves the real path, so no ``shell=True`` is needed.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

FRONTEND = Path(__file__).resolve().parent.parent / "frontend"
COMMANDS: dict[str, list[list[str]]] = {
    "install": [["ci"]],
    "dev": [["run", "dev"]],
    "build": [["run", "build"]],
    "check": [["run", "lint"], ["run", "typecheck"], ["run", "test"]],
}


def main(argv: list[str]) -> int:
    if len(argv) != 1 or argv[0] not in COMMANDS:
        print(f"usage: python scripts/frontend.py {{{'|'.join(COMMANDS)}}}", file=sys.stderr)
        return 2
    npm = shutil.which("npm")
    if npm is None:
        print("npm not found: install Node.js 20+ (see docs/setup_windows.md)", file=sys.stderr)
        return 1
    if argv[0] != "install" and not (FRONTEND / "node_modules").is_dir():
        code = subprocess.call([npm, "ci"], cwd=FRONTEND)
        if code:
            return code
    for args in COMMANDS[argv[0]]:
        code = subprocess.call([npm, *args], cwd=FRONTEND)
        if code:
            return code
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
