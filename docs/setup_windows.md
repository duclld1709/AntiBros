# Cài đặt môi trường trên Windows (native)

Áp dụng cho laptop Windows 11 + RTX 3050 6GB. Mọi lệnh gõ được trong cả PowerShell lẫn Git Bash.

## 1. Công cụ
| Công cụ | Cách cài | Kiểm tra |
|---|---|---|
| Driver NVIDIA (bản mới, Studio hoặc Game Ready) | GeForce Experience / nvidia.com | `nvidia-smi` |
| Git for Windows (kèm Git Bash) | `winget install Git.Git` | `git --version` |
| uv | `winget install astral-sh.uv` hoặc `powershell -c "irm https://astral.sh/uv/install.ps1 \| iex"` | `uv --version` |
| Docker Desktop (Linux containers) | `winget install Docker.DockerDesktop` | `docker version` |
| Ollama for Windows | `winget install Ollama.Ollama` | `ollama --version` |

Python không cần cài riêng: `uv` tự tải Python 3.11+ theo `pyproject.toml`.

## 2. Cấu hình Git (một lần cho repo)
```powershell
git config core.autocrlf false
git config core.longpaths true
```
`.gitattributes` đã ép `eol=lf` và `evalsets/** -text`, nên checksum và offset ký tự không bị CRLF làm lệch.

## 3. Cài dependencies
```powershell
uv sync                     # torch CUDA wheel (cu128) trên Windows, CPU wheel trên Linux CI
uv run python -c "import torch; print(torch.cuda.is_available())"   # phải in True
uv run poe check            # lint + typecheck + test
uv run pre-commit install
```

## 4. Ollama
- Nếu ổ C: chật, đặt biến môi trường người dùng `OLLAMA_MODELS` (ví dụ `D:\ollama\models`) rồi khởi động lại Ollama.
- Kéo các model trong `configs/models/*.yaml`, ví dụ `ollama pull qwen3.5:4b`.
- Kiểm tra digest khớp với config: `uv run python scripts/env_report.py`.

## 5. NVIDIA: tắt sysmem fallback
NVIDIA Control Panel → *Manage 3D settings* → *CUDA – Sysmem Fallback Policy* → **Prefer No Sysmem Fallback**
(đặt ở *Global Settings*, hoặc ở *Program Settings* cho `python.exe` của `.venv` và `ollama.exe`/`ollama_llama_server.exe`).
Setting có từ driver 536.40. Kết quả kiểm tra trên laptop: xem `reports/hardware.md`.

## 6. Qdrant
```powershell
uv run poe up               # = docker compose up -d qdrant
docker compose ps           # cột STATUS phải là "healthy"
```
Không có Docker: đặt `vector_store.mode: local` trong config (qdrant-client embedded).

## 7. Khuyến nghị
- Loại trừ thư mục repo, `.venv`, `%USERPROFILE%\.cache\huggingface` và thư mục model Ollama khỏi Windows Defender real-time scan.
- Khi chạy eval dài: cắm sạc, tắt sleep/hibernate.
- Đo lại phần cứng sau khi đổi driver hoặc model: `uv run python scripts/vram_probe.py`.
