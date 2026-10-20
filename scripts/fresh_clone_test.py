"""Fresh-clone smoke test of the one-command Docker setup (spec §10.3, M6).

Clones the repository into a temporary directory, starts
``docker compose -f docker-compose.yml -f docker-compose.smoke.yml up -d --build`` (fake
tokenizer, embedder, reranker and LLM on the CC BY mini corpus), waits for ``/health``,
asks one question and one red-flag question, checks the answers, and always tears the stack
down and deletes the clone. Runs the same way on Windows (Docker Desktop) and Linux.

    uv run python scripts/fresh_clone_test.py [--repo <path or URL>] [--ref HEAD] [--keep]
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
COMPOSE_FILES = ["-f", "docker-compose.yml", "-f", "docker-compose.smoke.yml"]
PROJECT = "antibros-smoke"


def run(cmd: list[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("$", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=cwd, check=check, text=True)


def http(method: str, url: str, body: dict[str, Any] | None = None) -> Any:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        url, data=data, method=method, headers={"Content-Type": "application/json"}
    )
    # Local stack only: never route through an HTTP(S) proxy from the environment.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def wait_healthy(base: str, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            if http("GET", f"{base}/health").get("status") == "ok":
                return
        except (urllib.error.URLError, ConnectionError, TimeoutError, json.JSONDecodeError):
            pass
        time.sleep(3)
    raise SystemExit(f"API not healthy after {timeout_s:.0f} s")


def check_answers(base: str) -> None:
    answer = http("POST", f"{base}/ask", {"question": "How much protein per day builds muscle?"})
    assert answer["status"] == "answered", answer
    ids = {c["id"] for c in answer["citations"]}
    assert ids, "answered without citations"
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer["answer"])}
    assert cited <= ids, f"answer cites {cited - ids} not in the citation table"
    red = http("POST", f"{base}/ask", {"question": "I have kidney disease, can I take creatine?"})
    assert red["status"] == "safety_redirect" and not red["citations"], red
    print("OK: /health, cited answer, red-flag redirect", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo", default=str(ROOT), help="Path or URL to clone.")
    parser.add_argument("--ref", default="HEAD", help="Commit, branch or tag to check out.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--timeout", type=float, default=900.0, help="Seconds to wait for /health.")
    parser.add_argument("--keep", action="store_true", help="Keep the clone and the stack.")
    parser.add_argument(
        "--no-build", action="store_true", help="Use an already built antibros-api:local image."
    )
    args = parser.parse_args()
    if shutil.which("docker") is None or shutil.which("git") is None:
        print("docker and git are required", file=sys.stderr)
        return 1
    docker, git = shutil.which("docker") or "docker", shutil.which("git") or "git"
    tmp = Path(tempfile.mkdtemp(prefix="antibros-clone-"))
    clone = tmp / "repo"
    compose = [docker, "compose", "-p", PROJECT, *COMPOSE_FILES]
    try:
        run([git, "clone", "--quiet", args.repo, str(clone)], cwd=tmp)
        run([git, "checkout", "--quiet", args.ref], cwd=clone)
        run([*compose, "up", "-d", "--no-build" if args.no_build else "--build"], cwd=clone)
        base = f"http://127.0.0.1:{args.port}"
        wait_healthy(base, args.timeout)
        check_answers(base)
        return 0
    except (subprocess.CalledProcessError, AssertionError, urllib.error.URLError) as e:
        print(f"FAILED: {e}", file=sys.stderr)
        if clone.exists():
            run([*compose, "logs", "--tail", "50"], cwd=clone, check=False)
        return 1
    finally:
        if not args.keep:
            if clone.exists():
                run([*compose, "down", "-v", "--remove-orphans"], cwd=clone, check=False)
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
