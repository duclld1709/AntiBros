# Runbook — Đánh giá cuối trên test (M5)

> Tài liệu vận hành cho **lần chạy duy nhất** của split `test`. Kế hoạch là **khai báo trước** (spec §7, ADR-0014) và đã lock ở cuối M4 (`configs/final_eval_plan.yaml`, guard đã chọn: `guard_rule_llm`). Trong lúc chạy **không viết tính năng mới, không đổi config, prompt hay ngưỡng.**
>
> Lệnh `uv run antibros …` gõ giống nhau trong PowerShell (Windows) và bash (Linux). Ví dụ theo dõi tiến trình có cả hai dạng.

---

## 0. Chọn máy (quyết định của chủ dự án, ghi lại trước khi chạy)

- `eval final` ghi `hardware` vào fingerprint và vào từng dòng của `test_runs.log`. **Cả plan phải chạy trên cùng một máy**: lệnh từ chối resume một run trên commit hoặc phần cứng khác.
- Spec giả định laptop RTX 3050 6GB; ngân sách ≤ 20 phút mỗi bước (§6.8) được đặt cho máy này. Ở M4, các run E2E leave-one-out chạy trên Linux Tesla V100 16GB (`reports/hardware.md`). Nếu chạy test trên V100: ghi lý do vào `experiments/deviations.md` trước khi chạy, và ghi rõ trong report rằng latency/thời gian không so sánh trực tiếp được với laptop. Metric chất lượng không phụ thuộc phần cứng, vì cùng model, cùng digest và temperature 0.
- Trên Linux, `uv sync` cài torch bản CPU. Để reranker và embedder chạy trên GPU, cài torch CUDA phù hợp vào venv như ở M4. Việc này không đổi file nào được git theo dõi, nên không phá điều kiện "cây git sạch".

## 1. Checklist trước khi chạy

- [ ] `git pull`: đang ở commit có `final_eval_plan.yaml` đã lock, `git status` sạch (riêng `experiments/` được phép thay đổi).
- [ ] `uv run poe check`, rồi `uv run antibros evalset verify` và `uv run antibros evalset leakcheck` đều pass.
- [ ] `experiments/test_runs.log` trống.
- [ ] Ollama đang chạy, đã `ollama pull` đúng hai model trong plan. `eval final` tự so digest qua `/api/tags` và từ chối nếu lệch.
- [ ] Qdrant sẵn sàng (Docker `docker compose up -d qdrant`, hoặc `vector_store.mode: local`). Collection của `baseline` (section_aware × bge_m3) và `full` (section_aware × e5_large) đã index; `data/processed/chunks_section_aware.jsonl` có mặt (cần cho Evidence Recall@k).
- [ ] Đã diễn tập trọn ít nhất 1 lần, và đã thử ngắt rồi resume 1 lần (mục 2).
- [ ] Laptop: *Prefer No Sysmem Fallback*, cắm sạc, tắt sleep (`powercfg /change standby-timeout-ac 0`, ghi lại giá trị cũ), đóng ứng dụng dùng GPU.
- [ ] **Chủ dự án đã đồng ý** cho chạy bản chính thức.

## 2. Diễn tập (dev, không ghi `test_runs.log`)

```powershell
uv run antibros eval final --rehearsal --limit 3
```
- Chạy đúng quy trình như bản chính thức, nhưng trên split dev, 3 câu mỗi run, trong các thư mục `experiments/rehearsal-<plan8>_*`. Kết quả so sánh nằm ở `experiments/rehearsal-<plan8>_comparison.md`. Diễn tập bỏ qua kiểm tra cây git sạch; mọi kiểm tra khác vẫn áp dụng.
- **Thử resume:** nhấn Ctrl+C khi `full_test` đang predict, rồi chạy lại đúng lệnh trên. Run tiếp tục trong cùng thư mục, không có prediction nào bị trùng (kiểm tra `predictions.jsonl`).
- Muốn diễn tập lại từ đầu: xóa các thư mục `experiments/rehearsal-*`. Không commit chúng nếu không cần.

## 3. Chạy chính thức

```powershell
uv run python scripts/env_report.py | Out-File -Encoding utf8 experiments/env_report_final.txt   # Linux: > experiments/env_report_final.txt
uv run antibros eval final --plan configs/final_eval_plan.yaml
```

| Bước | Nội dung | Ngân sách |
|---|---|---|
| 1 | Kiểm tra: plan lock và đã commit, cây git sạch, sha256 config, profile và digest model (cả digest Ollama đang cài), checksum evalset | < 1 phút |
| 2 | `baseline_test`: predict → unload → judge → report | ≤ 20 phút |
| 3 | `full_test`: predict → unload → judge → report | ≤ 20 phút |
| 4 | `guard_test` (`guard_rule_llm`) trên safety test | ≤ 5 phút |
| 5 | So sánh `full` với `baseline` (paired bootstrap; overall, theo ngôn ngữ, theo nhóm câu hỏi) → `experiments/final-<plan8>_comparison.md` | < 1 phút |

`experiments/test_runs.log` là JSONL. Mỗi lần một run đổi trạng thái (`started`, `resumed`, `interrupted`, `failed`, `completed`), lệnh ghi một dòng gồm thời điểm, plan sha256, commit, hardware và thư mục run.

## 4. Theo dõi

```powershell
Get-Content experiments/test_runs.log -Wait -Tail 5     # Linux: tail -f experiments/test_runs.log
nvidia-smi -l 10
```
Trên Windows: Task Manager → GPU. Nếu **Shared GPU memory** tăng mạnh, nghĩa là đang tràn sang RAM hệ thống.

## 5. Xử lý sự cố

| Tình huống | Làm gì |
|---|---|
| Bị ngắt (Ctrl+C, mất điện, sleep, Windows Update) | Chạy lại đúng lệnh ở mục 3, trên **cùng commit và cùng máy**. Run tiếp tục trong thư mục cũ (`resumed`); việc này không tính là chạy lại. |
| Run bị đánh dấu `failed` (Ollama crash, OOM, prediction hoặc judge lỗi hạ tầng) | Lệnh dừng lại và không tự resume run đó. Sửa môi trường, rồi chạy lại **một lần**: `uv run antibros eval final --rerun <tên_run> --reason "<mô tả sự cố hạ tầng>"`. Lần chạy lại dùng thư mục mới `_rerun1` và được ghi vào `experiments/deviations.md`. Lần thứ hai bị từ chối. |
| `Refusing to run: … no longer matches the plan` | Có ai đó đã đổi config, profile hoặc evalset. **Không** sửa plan; khôi phục file về đúng commit đã lock. |
| `installed digest … != plan` | Ollama đang giữ bản model khác. Pull đúng digest; không đổi plan. |
| `PermissionError` khi ghi file | Loại trừ thư mục repo khỏi antivirus hoặc OneDrive, rồi chạy lại (resume). |
| Run vượt ngân sách thời gian | Để run chạy xong. Ghi thời gian thật vào report; không cắt bớt câu hỏi. |
| Kết quả "xấu" | **Không chạy lại.** Ghi trung thực và phân tích ở `reports/error_analysis.md`. |

**Không được làm:** đổi config, prompt hay ngưỡng; chạy thêm run test ngoài plan; sửa evalset; sửa tay `test_runs.log`.

## 6. Sau khi chạy

```powershell
uv run antibros evalset verify
uv run antibros eval errors --run experiments/final-<plan8>_baseline_test --run experiments/final-<plan8>_full_test --run experiments/final-<plan8>_guard_test --out reports/_errors_test.md
git add experiments/ reports/_errors_test.md
git commit -m "exp: final evaluation on test (v1)"
git push
```
Khôi phục setting sleep. Sau đó viết `reports/error_analysis.md` từ bản nháp `reports/_errors_test.md`: rà lại nhãn của **100%** câu sai trên test, và đưa ví dụ điển hình cùng hướng cải thiện cho v2 (không sửa pipeline trong v1).
