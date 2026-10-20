from __future__ import annotations

import os
from pathlib import Path

import pytest

from antibros import io

VI = "Mỗi ngày nên ăn bao nhiêu protein để tăng cơ? Creatine có gây hói đầu không — ổn chứ?"


def test_utf8_round_trip_vietnamese(tmp_path: Path) -> None:
    p = tmp_path / "vi.txt"
    io.write_text(p, VI + "\n")
    assert io.read_text(p) == VI + "\n"
    # Bytes are UTF-8 and LF is not translated to CRLF on Windows.
    assert p.read_bytes() == (VI + "\n").encode("utf-8")


def test_json_keeps_vietnamese_unescaped(tmp_path: Path) -> None:
    p = tmp_path / "vi.json"
    io.write_json(p, {"q": VI})
    assert VI in p.read_text(encoding="utf-8")
    assert io.read_json(p) == {"q": VI}


def test_atomic_write_replaces_and_leaves_no_temp(tmp_path: Path) -> None:
    p = tmp_path / "sub" / "a.txt"
    io.write_text(p, "old")
    io.write_text(p, "new")
    assert io.read_text(p) == "new"
    assert sorted(x.name for x in p.parent.iterdir()) == ["a.txt"]


def test_atomic_write_failure_keeps_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = tmp_path / "a.txt"
    io.write_text(p, "original")

    def always_locked(src: object, dst: object) -> None:
        raise PermissionError("locked")

    monkeypatch.setattr("antibros.io.os.replace", always_locked)
    monkeypatch.setattr("antibros.io.time.sleep", lambda s: None)
    with pytest.raises(PermissionError):
        io.write_text(p, "new")
    assert p.read_text(encoding="utf-8") == "original"
    assert [x.name for x in tmp_path.iterdir()] == ["a.txt"]


def test_atomic_write_retries_permission_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_replace = os.replace
    calls = {"n": 0}

    def flaky(src: str, dst: str) -> None:
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError("locked by antivirus")
        real_replace(src, dst)

    monkeypatch.setattr("antibros.io.os.replace", flaky)
    monkeypatch.setattr("antibros.io.time.sleep", lambda s: None)
    p = tmp_path / "a.txt"
    io.write_text(p, "ok")
    assert calls["n"] == 3
    assert io.read_text(p) == "ok"


@pytest.mark.parametrize(
    "name", ["chunk.json", "PMC5477153__results__003.json", "a-b_c.txt", "tăng_cơ.md"]
)
def test_valid_windows_filenames(name: str) -> None:
    assert io.is_valid_windows_filename(name)


@pytest.mark.parametrize(
    "name",
    ["PMC5477153::results::003", "qwen3.5:4b", "a?b", "a*b", "con", "NUL.txt", "trail.", "a/b", ""],
)
def test_invalid_windows_filenames(name: str) -> None:
    assert not io.is_valid_windows_filename(name)


def test_run_tag() -> None:
    assert io.is_valid_run_tag("baseline_v1-dev")
    assert not io.is_valid_run_tag("bad tag")
    assert not io.is_valid_run_tag("a:b")
