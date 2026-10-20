"""Unit tests run with pytest-socket's --disable-socket: any real network call fails."""

from __future__ import annotations

import socket

import pytest
from pytest_socket import SocketBlockedError


def test_socket_is_blocked() -> None:
    with pytest.raises(SocketBlockedError):
        socket.socket(socket.AF_INET, socket.SOCK_STREAM)


def test_http_client_cannot_reach_network() -> None:
    import httpx

    with pytest.raises(SocketBlockedError):
        httpx.get("http://127.0.0.1:11434/api/tags", timeout=1)
