"""Print the environment fingerprint: OS, GPU, driver, CUDA, torch, Ollama and model digests.

Usage: uv run python scripts/env_report.py [--json]
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from typing import Any

import httpx

OLLAMA_URL = "http://127.0.0.1:11434"


def _run(args: list[str]) -> str | None:
    if shutil.which(args[0]) is None:
        return None
    try:
        out = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def gpu_info() -> list[dict[str, str]]:
    out = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    if not out:
        return []
    gpus = []
    for line in out.splitlines():
        name, mem, driver = (x.strip() for x in line.split(","))
        gpus.append({"name": name, "memory_total_mib": mem, "driver": driver})
    return gpus


def torch_info() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"installed": False}
    return {
        "version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cudnn": torch.backends.cudnn.version()  # type: ignore[no-untyped-call]
        if torch.cuda.is_available()
        else None,
    }


def ollama_info() -> dict[str, Any]:
    try:
        version = httpx.get(f"{OLLAMA_URL}/api/version", timeout=5).json()["version"]
        tags = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=5).json()["models"]
    except (httpx.HTTPError, KeyError):
        return {"reachable": False}
    return {
        "reachable": True,
        "version": version,
        "models": {
            m["name"]: {
                "digest": m["digest"],
                "quantization": m.get("details", {}).get("quantization_level"),
                "parameter_size": m.get("details", {}).get("parameter_size"),
                "family": m.get("details", {}).get("family"),
            }
            for m in tags
        },
    }


def report() -> dict[str, Any]:
    return {
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "gpus": gpu_info(),
        "torch": torch_info(),
        "ollama": ollama_info(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args()
    data = report()
    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return
    print(f"OS:      {data['os']}")
    print(f"Python:  {data['python']}")
    for g in data["gpus"] or [{"name": "none", "memory_total_mib": "-", "driver": "-"}]:
        print(f"GPU:     {g['name']} ({g['memory_total_mib']} MiB), driver {g['driver']}")
    t = data["torch"]
    cuda = f"CUDA {t.get('cuda_build')}, available={t.get('cuda_available')}"
    print(f"Torch:   {t.get('version')} ({cuda})")
    o = data["ollama"]
    if o.get("reachable"):
        print(f"Ollama:  {o['version']}")
        for name, m in o["models"].items():
            print(f"  {name:<24} {m['quantization']:<8} {m['digest'][:12]}")
    else:
        print("Ollama:  not reachable")


if __name__ == "__main__":
    main()
