# CLAUDE.md

Hướng dẫn cho Claude Code khi làm việc trong repo **AntiBros**.

## Dự án
AntiBros là trợ lý hỏi đáp dinh dưỡng tăng cơ (**protein & creatine**) dựa trên bằng chứng khoa học. Hệ thống gồm: RAG có trích dẫn, hybrid search, reranking, citation verification, abstention, guardrail y tế, evaluation và ablation định lượng.

- Corpus v1 cố ý nhỏ: **12 bài open access**. Toàn bộ dự án chạy trên **một laptop Windows (native, không dùng WSL2), RTX 3050 6GB**.
- **Claude Code viết toàn bộ code trên `main`.** Chủ dự án giám sát và duyệt.
- **Claude Code tự soạn toàn bộ câu hỏi đánh giá, không cần người review** (ADR-0010).
- Cuối mỗi milestone gắn tag `m0` … `m6`. Người học dùng `docs/milestones/` và các tag này để tự code lại.

👉 **Trước mỗi task, đọc `PROJECT_SPEC.md` và `docs/milestones/<milestone hiện tại>.md`.** Trước khi chạy đánh giá cuối, đọc thêm `docs/runbooks/final_eval.md`.

## Milestone hiện tại
**M5 — Đánh giá cuối & phân tích lỗi** (công cụ đã xong; chờ diễn tập và chạy chính thức trên máy có GPU, xem `docs/runbooks/final_eval.md`)

---

## 1. Xác định máy đang chạy (BẮT BUỘC đầu mỗi phiên)
Chạy `nvidia-smi --query-gpu=name,memory.total --format=csv`, rồi nói rõ máy ở câu đầu tiên của câu trả lời.

| Kết quả | Máy | Được làm |
|---|---|---|
| RTX 3050 (6GB) | **Laptop Windows** (máy duy nhất của dự án) | Mọi việc: code, test, ingest, index, soạn và kiểm chứng evalset, eval và ablation trên dev, report. **Split `test` chỉ chạy qua `antibros eval final`** (§7.6). |
| Không có GPU | CI hoặc máy khác | Viết code, viết docs, chạy unit test với provider `fake`. Không chạy model, không tạo số liệu thí nghiệm. |
| GPU khác | Máy lạ | Hỏi chủ dự án trước khi chạy bất kỳ eval nào, vì `hardware` là một trường trong fingerprint. |

---

## 2. Quy trình mỗi milestone
1. Đọc đề bài milestone. Với task lớn, đề xuất plan ngắn trước khi code.
2. Implement kèm test. Commit nhỏ.
3. Kiểm tra từng acceptance criteria, và báo rõ tiêu chí nào đạt, tiêu chí nào chưa.
4. Cập nhật docs liên quan, ADR, và mục "Câu hỏi tự kiểm tra" nếu có thay đổi.
5. Cập nhật dòng "Milestone hiện tại" ở trên. **Hỏi chủ dự án trước khi tạo tag `mx` và trước khi push.**

## 3. Lệnh thường dùng
Các lệnh dưới đây gõ được cả trong PowerShell (chủ dự án) lẫn Git Bash (shell mà Claude Code dùng trên Windows). Lệnh `antibros` chạy trong venv: dùng `uv run antibros …`, hoặc kích hoạt venv trước bằng `.venv\Scripts\Activate.ps1`.
```powershell
uv sync                                        # cài dependencies (torch CUDA trên Windows)
docker compose up -d qdrant                    # Qdrant cho dev (Docker Desktop)
docker compose up                              # toàn bộ hệ thống (một lệnh)
uv run poe lint | uv run poe format | uv run poe typecheck | uv run poe test
uv run poe check                               # lint + typecheck + test
antibros evalset verify && antibros evalset leakcheck
antibros ingest --manifest evalsets/v1/corpus_manifest.jsonl
antibros index --config configs/<x>.yaml
antibros eval retrieval --config configs/ablations/retrieval_<x>.yaml --set golden|synth
antibros eval safety  --config configs/ablations/guard_<x>.yaml --split dev
antibros eval predict --config configs/<x>.yaml --models configs/models/laptop.yaml --split dev --tag <name> [--limit 5]
antibros eval judge --run experiments/<run> --models configs/models/laptop.yaml
antibros eval tune --config configs/full.yaml --models configs/models/laptop.yaml --param grade_min_rerank_score|verify_max_removed_ratio
antibros summarize experiments/<a> experiments/<b> [--out reports/<x>.md]
antibros compare experiments/<a> experiments/<b>
antibros plan lock --guard-run experiments/<g1> --guard-run experiments/<g2> --guard-run experiments/<g3>   # cuối M4
antibros eval final --rehearsal --limit 3                 # diễn tập trên dev, không ghi test_runs.log
antibros eval final --plan configs/final_eval_plan.yaml   # chỉ ở M5, hỏi chủ dự án trước
antibros eval final --rerun <run> --reason "<sự cố hạ tầng>" # chạy lại một lần run bị failed
antibros eval errors --run experiments/<run> [--run …] --out reports/_errors_test.md
antibros serve                                 # API + UI đã build (frontend/dist) tại :8000
uv run poe ui | uv run poe ui-build | uv run poe ui-check   # Vite dev server :5173 | build | lint+typecheck+test
uv run python scripts/env_report.py            # GPU, driver, torch, Ollama, digest model
```
- Nếu một tác vụ chưa có, thêm nó vào `[tool.poe.tasks]` trong `pyproject.toml` thay vì chạy lệnh dài thủ công. **Không dùng Makefile** (Windows không có `make`).
- Không viết script `.sh` hay `.ps1` cho quy trình dev. Script tiện ích viết bằng Python để chạy được cả trên Windows và CI Linux.

## 4. Tech stack
Python 3.11+ · uv · Typer · LangGraph · Qdrant · bge-m3 / bge-reranker-v2-m3 · PyTorch (CUDA, FP16) · Ollama (GGUF) · FastAPI + Pydantic · React + TypeScript (Vite, `frontend/`) · RAGAS · ruff · mypy · pytest · GitHub Actions · Docker Compose.

- **Laptop: Windows native.**
  - Ollama for Windows (native), Qdrant trong Docker Desktop (hoặc chế độ embedded `vector_store.mode: local`).
  - PyTorch dùng CUDA wheel trên Windows và CPU wheel trên CI Linux, phân nhánh bằng platform marker trong `[tool.uv.sources]`.
  - Code phải chạy được trên **cả Windows lẫn Linux**: CI chạy test trên cả hai, còn image Docker là Linux.
- **VRAM 6GB:**
  - Không bao giờ nạp hai LLM cùng lúc; eval chia `predict` → `judge`.
  - Reranker có thể chuyển sang CPU qua config.
  - Đo VRAM thật trước khi đổi model (`scripts/vram_probe.py`).
- **Phiên bản PyTorch** chọn ở M0, pin bằng `uv.lock`. Không nâng cấp giữa chừng nếu chưa hỏi, vì việc này làm đổi fingerprint.
- Với các thư viện thay đổi nhanh (LangGraph, RAGAS, Ollama, qdrant-client), **kiểm tra tài liệu hiện hành** thay vì dựa vào trí nhớ. Tên và tag model trên Ollama phải được xác minh trước khi ghi vào config.

## 5. Quy ước code
- Type hints đầy đủ; mọi schema dùng Pydantic.
- **Mọi tham số pipeline, model và ngưỡng nằm trong `configs/`**, không hardcode. Mỗi bước của pipeline bật/tắt được qua config để phục vụ ablation.
- Prompt nằm trong `prompts/`, không viết inline trong code.
- **Gọi LLM chỉ qua `antibros.llm.LLMClient`**, không gọi SDK của provider trực tiếp ở nơi khác (kể cả RAGAS, vốn phải đi qua wrapper). Mọi lời gọi đều qua cache.
- Mỗi chiến lược (chunker, embedder, retriever, reranker, verifier, guard) theo một interface chung, chọn qua config. Embedder và reranker có implementation `fake` cho CI.
- Dùng `logging`/`structlog`, không dùng `print` trong `src/`.
- Code, docstring, comment, commit message, README và ADR viết bằng **tiếng Anh**. Spec, đề bài milestone và runbook viết tiếng Việt.
- Tránh abstraction sớm khi mới có một implementation.

### Quy tắc tương thích Windows (spec §4.6)
- **Đường dẫn:** chỉ dùng `pathlib.Path`; không ghép chuỗi bằng `/` hay `\`; không hardcode ổ đĩa; giữ đường dẫn ngắn.
- **Tên file:** không chứa `: * ? " < > |`. ID như `chunk_id` (có `::`) không bao giờ được dùng làm tên file; file cache đặt tên theo hash.
- **Encoding:** luôn ghi `encoding="utf-8"` khi đọc/ghi text; JSON dùng `ensure_ascii=False`. Không dựa vào encoding mặc định của hệ thống.
- **Xuống dòng:** giữ `.gitattributes` (`eol=lf`, `evalsets/** -text`); checksum tính trên bytes; bước clean chuẩn hóa `\r\n`.
- **Tiến trình con:** `subprocess` với danh sách tham số, không `shell=True`; không gọi lệnh Unix từ code.
- **Timeout:** không dùng `signal.SIGALRM`; đặt timeout ở client HTTP hoặc `concurrent.futures`.
- **Multiprocessing:** entry point đặt trong `if __name__ == "__main__":`; `num_workers=0` mặc định.
- **Ghi file:** ghi ra file tạm rồi `os.replace`, có retry ngắn khi gặp `PermissionError`.

## 6. Testing
- Viết test song song với code cho: parser, chunker (kể cả `char_start`/`char_end`), RRF, span-relevance, metric, bootstrap, guardrail, API schema, LLM adapter, leakcheck, và round-trip UTF-8 với chuỗi tiếng Việt có dấu.
- Test không phụ thuộc hệ điều hành: dùng `tmp_path`, không hardcode đường dẫn hay ký tự xuống dòng. CI chạy trên `windows-latest` và `ubuntu-latest`; test dùng Qdrant `:memory:`.
- **Unit test không gọi dịch vụ thật** (LLM, Europe PMC). Dùng provider `fake`, mock HTTP, và fixture đã ghi sẵn. Test gọi API thật đánh dấu `@pytest.mark.integration`, mặc định bị skip, và **không chạy trong dự án này**.
- Trước khi báo task hoàn thành: `uv run poe check`. Khi đã có evalset thì chạy thêm `antibros evalset verify && antibros evalset leakcheck`.

## 7. Quy tắc quan trọng
1. **Không bao giờ bịa số liệu.** Mọi con số trong README, report hay CV phải đến từ một thư mục trong `experiments/`. Chưa chạy thì ghi `TBD`. Ước tính (VRAM, thời gian) phải được đánh dấu rõ là ước tính.
2. **Citation là bất biến cốt lõi:** mọi claim phải trỏ tới chunk có thật trong context. Đánh số citation bằng code, không để LLM tự đặt.
3. **Không đủ bằng chứng thì abstain.**
4. **Guardrail y tế:** nhánh red-flag không gọi retrieval và không trả lời chuyên môn.
5. **Evalset do Claude Code tự soạn, và sau khi freeze thì bất khả xâm phạm:**
   - Soạn theo đúng quy trình ở spec §6.2: rubric → nháp kèm đoạn trích nguyên văn → kiểm tra bằng code → 2 verifier khác họ → review log → data card → freeze. Không bỏ bước nào. Không chờ người review.
   - Không sửa `evalsets/v1/` sau khi freeze. Thay đổi đồng nghĩa với tạo `v2`, và kết quả cũ không còn so sánh được.
   - Không dùng câu trong evalset làm few-shot, đưa vào prompt, code hay config. `leakcheck` phải pass.
   - Khi viết hoặc sửa prompt, không mở `golden.jsonl` hay `safety.jsonl`. Chỉ xem metric tổng hợp và ví dụ lỗi của **dev**.
   - Không hardcode bất cứ thứ gì để qua eval.
6. **Kỷ luật tập test:**
   - Split `test` chỉ chạy qua `antibros eval final --plan configs/final_eval_plan.yaml`, ở M5, sau khi hỏi chủ dự án.
   - Mỗi cặp (cấu hình, bộ) chỉ chạy một lần.
   - Diễn tập dùng dev với `--limit`.
7. **Kế hoạch đánh giá cuối là khai báo trước:**
   - Chốt và commit `final_eval_plan.yaml` cuối M4.
   - Không đổi giả thuyết, cấu hình `full`/`baseline` hay quy tắc chọn guard sau khi đã thấy kết quả test.
   - Chạy lại chỉ khi có sự cố hạ tầng, tối đa một lần, và ghi vào report như một sai lệch.
8. **Ngân sách thời gian eval** (spec §6.8): dev ≤ 10 phút, test ≤ 20 phút mỗi lệnh. Nếu vượt, xử lý theo thứ tự đã khai báo, ghi ADR. Không thu nhỏ evalset.
9. **Báo kết quả trung thực:**
   - Luôn nêu khoảng tin cậy; nói rõ khi chênh lệch không có ý nghĩa thống kê.
   - Ghi rõ các hạn chế: corpus một ngách 12 bài; test set nhỏ; evalset do Claude soạn và model kiểm chứng, không có người review; verify do chính generator thực hiện.
10. **Dữ liệu:** chỉ dùng tài liệu open access; luôn lưu `license`. Không commit `data/`. Có commit `experiments/`.
11. **Secrets** chỉ đọc từ env. Khi thêm biến mới, cập nhật `.env.example`.
12. Khi thiết kế cần lệch khỏi spec: nêu lý do, đề xuất cập nhật `PROJECT_SPEC.md`, và tạo ADR.

## 8. Git
- Chỉ dùng nhánh `main`. Tag `m0` … `m6`.
- Commit nhỏ, theo Conventional Commits (`feat:`, `fix:`, `test:`, `docs:`, `refactor:`, `chore:`, `data:` cho evalset, `exp:` cho kết quả thí nghiệm).
- Hỏi trước khi push, tạo tag, tạo release, hay force-push.
