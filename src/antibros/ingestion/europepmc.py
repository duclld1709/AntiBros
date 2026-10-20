"""Minimal Europe PMC REST client: rate limited, retried with exponential backoff."""

from __future__ import annotations

import time
from typing import Any

import httpx

from antibros.config import EuropePMCConfig
from antibros.logging import get_logger

log = get_logger(__name__)

_RETRY_STATUS = {429, 500, 502, 503, 504}


class EuropePMCError(RuntimeError):
    pass


class EuropePMCClient:
    def __init__(self, cfg: EuropePMCConfig, transport: httpx.BaseTransport | None = None):
        self.cfg = cfg
        self._http = httpx.Client(base_url=cfg.base_url, timeout=cfg.timeout_s, transport=transport)
        self._last_request = 0.0
        self._sleep = time.sleep

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> EuropePMCClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _get(self, path: str, params: dict[str, str] | None = None) -> httpx.Response:
        for attempt in range(self.cfg.max_retries + 1):
            wait = self.cfg.min_interval_s - (time.monotonic() - self._last_request)
            if wait > 0:
                self._sleep(wait)
            self._last_request = time.monotonic()
            try:
                resp = self._http.get(path, params=params)
            except httpx.TransportError as exc:
                error: str = f"{type(exc).__name__}: {exc}"
            else:
                if resp.status_code not in _RETRY_STATUS:
                    resp.raise_for_status()
                    return resp
                error = f"HTTP {resp.status_code}"
            if attempt == self.cfg.max_retries:
                raise EuropePMCError(f"GET {path} failed after {attempt + 1} attempts: {error}")
            delay = self.cfg.backoff_s * 2**attempt
            log.warning("europepmc_retry", path=path, error=error, delay_s=delay)
            self._sleep(delay)
        raise AssertionError("unreachable")

    def search(self, query: str, page_size: int = 5) -> list[dict[str, Any]]:
        resp = self._get(
            "/search",
            {"query": query, "resultType": "core", "format": "json", "pageSize": str(page_size)},
        )
        results: list[dict[str, Any]] = resp.json().get("resultList", {}).get("result", [])
        return results

    def article(self, pmcid: str) -> dict[str, Any] | None:
        results = self.search(f"PMCID:{pmcid}", page_size=1)
        return results[0] if results else None

    def full_text_xml(self, pmcid: str) -> bytes:
        return self._get(f"/{pmcid}/fullTextXML").content
