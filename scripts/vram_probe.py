"""Measure real VRAM use and LLM throughput on this machine.

Steps (each isolated; LLMs are unloaded between steps):
  1. idle VRAM and which GPU drives the display;
  2. each LLM at num_ctx 4096 and 8192: VRAM (nvidia-smi + Ollama /api/ps), prefill & decode tok/s;
  3. bge-m3 FP16 and bge-reranker-v2-m3 FP16 on CUDA;
  4. pipeline LLM (num_ctx 8192) + embedder + reranker resident together.

Usage: uv run python scripts/vram_probe.py --pipeline qwen3.5:4b --judge gemma4:e4b
Writes reports/hardware_probe.json.
"""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx

from antibros.io import write_json
from antibros.llm import LLMParams, Message, OllamaClient

OLLAMA = "http://127.0.0.1:11434"
# Neutral filler (not evalset content) to get a long prompt for prefill measurement.
FILLER = (
    "The committee reviewed the quarterly logistics schedule, noting that shipments from the "
    "northern warehouse were delayed by weather while the southern route operated normally. "
)


def gpu_used_mib() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    return int(out.stdout.strip().splitlines()[0])


def display_info() -> dict[str, str]:
    out = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=display_active,display_mode",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    active, mode = (x.strip() for x in out.stdout.strip().split(","))
    return {"display_active": active, "display_mode": mode}


def ollama_ps() -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = httpx.get(f"{OLLAMA}/api/ps", timeout=10).json()["models"]
    return models


def unload_all() -> None:
    for m in ollama_ps():
        httpx.post(f"{OLLAMA}/api/generate", json={"model": m["name"], "keep_alive": 0}, timeout=60)
    for _ in range(30):
        if not ollama_ps():
            break
        time.sleep(1)
    time.sleep(2)


def probe_llm(model: str, num_ctx: int, prompt_words: int) -> dict[str, Any]:
    unload_all()
    before = gpu_used_mib()
    client = OllamaClient(model, keep_alive="10m")
    reps = max(1, prompt_words // len(FILLER.split()))
    prompt = FILLER * reps + "\nSummarise the text above in about 150 words."
    msgs = [Message(role="user", content=prompt)]
    params = LLMParams(num_ctx=num_ctx, max_tokens=256)
    client.complete([Message(role="user", content="Say OK.")], params=params)  # load + warm-up
    res = client.complete(msgs, params=params)
    after = gpu_used_mib()
    ps = next((m for m in ollama_ps() if m["name"] == model), {})
    u = res.usage
    return {
        "model": model,
        "num_ctx": num_ctx,
        "vram_delta_mib": after - before,
        "ollama_size_mib": round(ps.get("size", 0) / 2**20),
        "ollama_size_vram_mib": round(ps.get("size_vram", 0) / 2**20),
        "fully_on_gpu": ps.get("size") == ps.get("size_vram"),
        "prompt_tokens": u.prompt_tokens,
        "completion_tokens": u.completion_tokens,
        "prefill_tok_s": round(u.prompt_tokens / (u.prompt_eval_ms / 1000), 1)
        if u.prompt_eval_ms
        else None,
        "decode_tok_s": round(u.completion_tokens / (u.eval_ms / 1000), 1) if u.eval_ms else None,
        "latency_s": round(res.latency_ms / 1000, 2),
    }


def load_encoders(device: str) -> tuple[Any, Any]:
    import torch
    from sentence_transformers import CrossEncoder, SentenceTransformer

    emb = SentenceTransformer(
        "BAAI/bge-m3", device=device, model_kwargs={"torch_dtype": torch.float16}
    )
    rr = CrossEncoder(
        "BAAI/bge-reranker-v2-m3", device=device, model_kwargs={"torch_dtype": torch.float16}
    )
    return emb, rr


def probe_encoder(kind: str) -> dict[str, Any]:
    import torch
    from sentence_transformers import CrossEncoder, SentenceTransformer

    unload_all()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before = gpu_used_mib()
    texts = [FILLER * 4] * 16
    t0 = time.perf_counter()
    if kind == "bge-m3":
        m: Any = SentenceTransformer(
            "BAAI/bge-m3", device="cuda", model_kwargs={"torch_dtype": torch.float16}
        )
        m.encode(texts, batch_size=16)
    else:
        m = CrossEncoder(
            "BAAI/bge-reranker-v2-m3", device="cuda", model_kwargs={"torch_dtype": torch.float16}
        )
        m.predict([("query about shipments", t) for t in texts], batch_size=16)
    elapsed = time.perf_counter() - t0
    result = {
        "model": kind,
        "vram_delta_mib": gpu_used_mib() - before,
        "torch_peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20),
        "batch16_seconds_incl_load": round(elapsed, 2),
    }
    del m
    gc.collect()
    torch.cuda.empty_cache()
    return result


def probe_combo(model: str) -> dict[str, Any]:
    import torch

    unload_all()
    before = gpu_used_mib()
    client = OllamaClient(model, keep_alive="10m")
    client.complete([Message(role="user", content="Say OK.")], params=LLMParams(num_ctx=8192))
    emb, rr = load_encoders("cuda")
    emb.encode(["query about shipments"])
    rr.predict([("query about shipments", FILLER * 4)] * 20, batch_size=20)
    res = client.complete(
        [Message(role="user", content=FILLER * 40 + "\nSummarise in 100 words.")],
        params=LLMParams(num_ctx=8192, max_tokens=200),
    )
    after = gpu_used_mib()
    ps = next((m for m in ollama_ps() if m["name"] == model), {})
    total = int(
        subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        ).stdout.strip()
    )
    out = {
        "llm": model,
        "vram_used_before_mib": before,
        "vram_used_after_mib": after,
        "vram_total_mib": total,
        "llm_fully_on_gpu": ps.get("size") == ps.get("size_vram"),
        "llm_size_vram_mib": round(ps.get("size_vram", 0) / 2**20),
        "decode_tok_s_with_encoders": round(
            res.usage.completion_tokens / (res.usage.eval_ms / 1000), 1
        )
        if res.usage.eval_ms
        else None,
    }
    del emb, rr
    gc.collect()
    torch.cuda.empty_cache()
    unload_all()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pipeline", default="qwen3.5:4b")
    ap.add_argument("--judge", default="gemma4:e4b")
    ap.add_argument("--extra", nargs="*", default=[], help="more LLMs to probe")
    ap.add_argument("--prompt-words", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=Path("reports/hardware_probe.json"))
    args = ap.parse_args()

    unload_all()
    data: dict[str, Any] = {
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "idle_vram_used_mib": gpu_used_mib(),
        "display": display_info(),
        "llm": [],
    }
    for model in [args.pipeline, args.judge, *args.extra]:
        for ctx in (4096, 8192):
            r = probe_llm(model, ctx, args.prompt_words)
            print(json.dumps(r))
            data["llm"].append(r)
    unload_all()
    data["encoders"] = [probe_encoder("bge-m3"), probe_encoder("bge-reranker-v2-m3")]
    print(json.dumps(data["encoders"]))
    data["combo"] = probe_combo(args.pipeline)
    print(json.dumps(data["combo"]))
    write_json(args.out, data)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
