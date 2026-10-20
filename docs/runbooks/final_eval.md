# Runbook — Đánh giá cuối trên test (M5)

> Tài liệu vận hành cho **lần chạy duy nhất** của split `test`, trên laptop **Windows** RTX 3050 6GB. Kế hoạch là **khai báo trước** (spec §7, ADR-0014). Trong lúc chạy **không viết tính năng mới, không đổi config, prompt hay ngưỡng.**
>
> Các lệnh dưới đây dùng PowerShell. Lệnh `antibros` có thể gõ dưới dạng `uv run antibros …`.

---

## 1. Checklist trước khi chạy

- [ ] `configs/final_eval_plan.yaml`, `full.yaml`, `baseline.yaml` và `ablations/guard_*.yaml` đã được commit ở cuối M4; tag `m4` đã tạo (sau khi hỏi chủ dự án).
- [ ] `git status` sạch.
- [ ] `antibros evalset verify; antibros evalset leakcheck` pass.
- [ ] `experiments/test_runs.log` không có run nào trùng với plan.
- [ ] Ollama for Windows đang chạy; `ollama list` cho digest khớp với `configs/models/laptop.yaml` và plan.
- [ ] Docker Desktop đang chạy; `docker compose up -d qdrant` ở trạng thái `healthy`. Nếu dùng `vector_store.mode: local`, bỏ qua bước này. Collection của cấu hình `full` và `baseline` đã index.
- [ ] `antibros eval final --plan configs/final_eval_plan.yaml --rehearsal` đã chạy trọn ít nhất 1 lần, và đã thử ngắt rồi resume 1 lần.
- [ ] NVIDIA Control Panel: *CUDA – Sysmem Fallback Policy* = *Prefer No Sysmem Fallback* (như đã đặt ở M0).
- [ ] Laptop cắm sạc. Tắt sleep trong lúc chạy, ví dụ: `powercfg /change standby-timeout-ac 0` (ghi lại giá trị cũ để khôi phục sau).
- [ ] Đóng các ứng dụng dùng GPU (trình duyệt có tăng tốc phần cứng, game, phần mềm quay màn hình). Tạm hoãn Windows Update nếu có thể.
- [ ] **Chủ dự án đã đồng ý** cho chạy bản chính thức.

## 2. Chạy

```powershell
uv run python scripts/env_report.py | Out-File -Encoding utf8 experiments/env_report_final.txt
uv run antibros eval final --plan configs/final_eval_plan.yaml 2>&1 | Tee-Object -FilePath experiments/final_console.log -Append
```

| Bước | Nội dung | Ngân sách |
|---|---|---|
| 1 | Kiểm tra trước khi chạy: git sạch, plan hash, config hash, digest model, checksum evalset | < 1 phút |
| 2 | `baseline` trên golden test: predict → unload → judge → report | ≤ 20 phút |
| 3 | `full` trên golden test: predict → unload → judge → report | ≤ 20 phút |
| 4 | Guard đã chọn trên safety test | ≤ 5 phút |
| 5 | `compare full baseline` (paired bootstrap) | < 1 phút |

Mỗi run ghi một dòng vào `experiments/test_runs.log` (thời điểm, commit, plan hash, tên run, trạng thái).

## 3. Theo dõi

Mở một cửa sổ PowerShell khác:
```powershell
Get-Content experiments/final_console.log -Wait -Tail 20
nvidia-smi -l 10        # VRAM, mức sử dụng GPU
```
Task Manager → Performance → GPU: nếu thấy **Shared GPU memory** tăng mạnh, nghĩa là đang tràn sang RAM hệ thống. Kiểm tra lại setting sysmem fallback.

## 4. Xử lý sự cố

| Tình huống | Làm gì |
|---|---|
| Bị ngắt (Ctrl+C, mất điện, sleep, Windows Update khởi động lại) | Chạy lại đúng lệnh ở mục 2. Harness resume từ cache, không tính là chạy lại. |
| OOM, hoặc Ollama crash giữa chừng | Sửa môi trường (đóng ứng dụng khác, thoát rồi mở lại Ollama ở khay hệ thống), rồi chạy lại **một lần**. Ghi vào `experiments/deviations.md`. |
| `PermissionError` khi ghi file (file bị khóa) | Kiểm tra antivirus hoặc trình đồng bộ (OneDrive) có đang khóa thư mục không; loại trừ thư mục repo, rồi chạy lại (resume). |
| Docker Desktop dừng | Mở lại Docker Desktop, đợi `qdrant` healthy, rồi chạy lại (resume). |
| Lỗi code chặn việc chạy (đường dẫn, phiên bản thư viện) | Sửa tối thiểu, commit với `fix:`, ghi vào `deviations.md`. Không đổi logic pipeline. |
| Run vượt ngân sách thời gian | Để chạy xong. Ghi thời gian thật vào report; không cắt bớt câu hỏi. |
| Kết quả "xấu" | **Không chạy lại.** Ghi trung thực, rồi phân tích ở `reports/error_analysis.md`. |

**Không được làm:** đổi config, prompt hay ngưỡng; chạy thêm run test ngoài plan; sửa evalset.

## 5. Sau khi chạy

```powershell
uv run antibros evalset verify
uv run antibros eval report --run experiments/<run>      # tính lại từ cache để kiểm tra
git add experiments/
git commit -m "exp: final evaluation on test (v1)"
```
Khôi phục setting sleep. Sau đó viết `reports/error_analysis.md` (M5) và chuyển sang M6.
