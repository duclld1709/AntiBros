"""Small tracing backends: local append-only JSONL and optional Langfuse REST."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

import httpx

from antibros.config import Settings, TracingConfig
from antibros.io import append_jsonl


def record_trace(config: TracingConfig, settings: Settings, record: dict[str, Any]) -> str:
    trace_id = str(record.get("trace_id") or uuid.uuid4())
    event = {
        "trace_id": trace_id,
        "timestamp": datetime.now(UTC).isoformat(timespec="milliseconds"),
        **record,
    }
    if config.backend == "jsonl":
        append_jsonl(config.path, event)
    elif config.backend == "langfuse":
        public = settings.langfuse_public_key
        secret = settings.langfuse_secret_key
        if public is None or secret is None:
            raise ValueError("LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are required")
        response = httpx.post(
            f"{settings.langfuse_host.rstrip('/')}/api/public/traces",
            auth=(public.get_secret_value(), secret.get_secret_value()),
            json={
                "id": trace_id,
                "name": "antibros.ask",
                "input": record.get("input"),
                "output": record.get("output"),
                "metadata": record.get("metadata", {}),
            },
            timeout=5.0,
        )
        response.raise_for_status()
    return trace_id
