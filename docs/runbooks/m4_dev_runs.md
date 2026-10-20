# Runbook — Các run trên dev ở M4 (laptop Windows)

> Thứ tự chạy mọi thí nghiệm dev của M4, sau khi code M4 đã có trên `main`. Mọi lệnh chỉ dùng split **dev**. Không mở `golden.jsonl` hay `safety.jsonl` khi sửa prompt (CLAUDE.md §7.5). Mỗi run tạo một thư mục trong `experiments/`; commit chúng với prefix `exp:`.
>
> Lệnh viết cho PowerShell. Chạy `nvidia-smi --query-gpu=name,memory.total --format=csv` trước để chắc chắn đang ở laptop RTX 3050.

## 0. Chuẩn bị
```powershell
uv sync
uv run antibros evalset verify; uv run antibros evalset leakcheck
uv run antibros index --config configs/baseline.yaml      # 3 chunker x 2 embedder
```
Reranker `bge-reranker-v2-m3` được tải lần đầu khi chạy. Nếu OOM, đổi `rerank.device: cpu` trong `configs/base.yaml` và ghi lại trong `reports/hardware.md`.

## 1. Ablation retrieval (không cần LLM, mỗi run ≤ 2 phút)
Điểm tham chiếu là `configs/ablations/retrieval_hybrid.yaml`; mỗi file `retrieval_*.yaml` khác đúng một yếu tố.
```powershell
$cfgs = "hybrid", "fixed", "recursive", "sparse", "dense", "dense_e5", "rerank"
foreach ($c in $cfgs) { foreach ($s in "golden", "synth") {
  uv run antibros eval retrieval --config configs/ablations/retrieval_$c.yaml --set $s --tag "retr_${c}_$s"
} }
$golden = (Get-ChildItem experiments -Directory -Filter "*_retr_*_golden").FullName
uv run antibros summarize @golden --out reports/_retrieval_golden.md
```
So sánh từng trục bằng paired bootstrap, ví dụ `uv run antibros compare experiments/<hybrid_golden> experiments/<fixed_golden>`. Các trục: chunking (fixed, recursive so với hybrid), retrieval (sparse, dense so với hybrid), embedding (dense_e5 so với dense, xem riêng `language=vi`), reranker (rerank so với hybrid). Ghi kết luận vào `reports/retrieval_ablation.md`. Nếu một trục cho thấy cấu hình khác tốt hơn **có ý nghĩa thống kê**, cập nhật `retrieval:` trong `configs/full.yaml` và ADR tương ứng (0002–0004).

## 2. Guard trên dev (mỗi run ≤ 5 phút)
```powershell
foreach ($g in "rule", "llm", "rule_llm") {
  uv run antibros eval safety --config configs/ablations/guard_$g.yaml --models configs/models/laptop.yaml --split dev --tag "guard_$g"
}
```
Quy tắc chọn (spec §7.1): recall cao nhất với FPR ≤ 0.15; hòa thì chọn latency thấp hơn. Đặt `guard.mode` trong `configs/full.yaml` bằng biến thể được chọn (`plan lock` sẽ kiểm tra).

## 3. Tinh chỉnh ngưỡng (chỉ lưới đã khai báo trong `configs/full.yaml` → `tuning`)
```powershell
uv run antibros eval tune --config configs/full.yaml --models configs/models/laptop.yaml --param grade_min_rerank_score --out reports/_tune_grade.md
```
Chọn theo quy tắc ở ADR-0007 (F1 abstention cao nhất; hòa thì tỷ lệ abstain sai thấp hơn, rồi gần giá trị mặc định). Ghi giá trị vào `configs/full.yaml`, rồi làm tương tự với `verify_max_removed_ratio`.

## 4. E2E leave-one-out (mỗi run predict + judge ≤ 10 phút)
```powershell
$runs = @{ "full" = "configs/full.yaml"; "loo_no_rerank" = "configs/ablations/loo_no_rerank.yaml"; "loo_no_rewrite" = "configs/ablations/loo_no_rewrite.yaml"; "loo_no_grade" = "configs/ablations/loo_no_grade.yaml"; "loo_no_verify" = "configs/ablations/loo_no_verify.yaml" }
foreach ($k in $runs.Keys) {
  uv run antibros eval predict --config $runs[$k] --models configs/models/laptop.yaml --split dev --tag "m4_$k"
}
# rồi với từng thư mục run:
uv run antibros eval judge --run experiments/<run> --models configs/models/laptop.yaml
uv run antibros compare experiments/<m4_full> experiments/<m4_loo_x>
```
Kiểm tra `timing.json` của từng run (`over_budget` phải là `false`). Nếu vượt ngân sách, xử lý theo thứ tự ở spec §6.8 và ghi ADR.

## 5. Chốt kế hoạch đánh giá cuối
```powershell
uv run antibros plan lock --guard-run experiments/<guard_rule> --guard-run experiments/<guard_llm> --guard-run experiments/<guard_rule_llm>
git add configs/final_eval_plan.yaml configs/full.yaml experiments/
git commit -m "exp: m4 dev runs and locked final evaluation plan"
```
`experiments/test_runs.log` phải vẫn trống. Cập nhật ADR 0002–0004, 0006, 0007 (Accepted/Rejected, có link `experiments/`) và `reports/retrieval_ablation.md`. **Hỏi chủ dự án trước khi tạo tag `m4`.**
