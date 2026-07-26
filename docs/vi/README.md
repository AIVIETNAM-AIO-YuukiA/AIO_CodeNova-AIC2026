# CodeNova — Pipeline truy hồi video (HCMC AI Challenge 2026)

Hệ thống truy hồi video chạy-tiếp-được (resumable): video được tách thành shot,
lấy keyframe, embed bằng **BEiT-3 large** (384px, dim 1024, fine-tune
COCO-retrieval) và index vào Qdrant. Chữ trên màn hình (OCR) và lời nói (ASR)
được index vào Elasticsearch. Một agent LLM (Qwen3.5-4B, chạy qua Docker) trả
lời câu hỏi VQA và điều khiển vòng chat thu hẹp kết quả tìm kiếm.

> Bản tiếng Anh: [../../README.md](../../README.md)

## Pipeline

```
OFFLINE (đánh index)
  ingest → detect-shots → extract-frames → embed-frames → build-index
                                          (vector index: BEiT-3 large → Qdrant)
  extract-text  (bước riêng: OCR theo keyframe + ASR theo video → Elasticsearch;
                 cần `make atlas-index-up` + `make elasticsearch-up`)

ONLINE (truy hồi)
  câu truy vấn → [LLM dịch/mở rộng] → embed (BEiT-3) → tìm trên Qdrant
    → temporal search (đi lan frame-to-frame → segment) → shot validation
    → định dạng theo track (Textual KIS / VQA / TRAKE)
    → [VQA] agent trả lời  /  [chat] vòng lặp agent tương tác
```

Mọi model sinh (LLM/VLM) đều **chạy qua Docker, giao tiếp bằng HTTP chuẩn
OpenAI** — không load checkpoint nào trong process Python cho agent,
captioning, OCR hay xử lý query. Các model embed/rerank (BEiT-3, SigLIP2,
BLIP-2) chạy in-process vì là batch encoder.

## Model

| Vai trò | Model | Chạy ở đâu |
|---------|-------|------------|
| Embed ảnh (mặc định) | BEiT-3 `beit3_large_patch16_384_coco_retrieval` (dim 1024) | in-process, tự tải về `external/BEiT3/checkpoints/` |
| Embed ảnh (tùy chọn) | SigLIP2 `google/siglip2-so400m-patch14-384` (dim 1152) | in-process |
| Embed caption tiếng Việt (tùy chọn) | `AITeamVN/Vietnamese_Embedding_v2` trên caption do VLM sinh | caption qua Docker VLM; embed in-process |
| Reranker (tùy chọn, khi chạy nhiều model) | BLIP-2 ITM `Salesforce/blip2-itm-vit-g` | in-process |
| Captioning + OCR (index & tool agent) | `nvidia/Qwen3.6-35B-A3B-NVFP4` | Docker `atlas-index` (Atlas, **chỉ GB10**), cổng 8881 |
| LLM agent (VQA, chat, mở rộng query) | **Qwen3.5-4B 4-bit** | Docker cổng 8888 — Atlas (NVFP4) trên DGX Spark/GB10, llama.cpp (GGUF Q4_K_M) trên GPU khác |
| ASR | gipformer-65M-rnnt + Silero-VAD | subprocess vào `external/gipformer/` |
| Tách shot | TransNetV2 (PyTorch) | in-process |

`atlas-agent` (4B, cổng 8888) và `atlas-index` (35B-A3B, cổng 8881) là 2
container Atlas độc lập, cổng riêng để chạy đồng thời mà không tranh chấp bộ
nhớ hợp nhất của GB10. Captioning/OCR **không có fallback cho máy khác GB10**
— `atlas-index` bắt buộc chạy Atlas, chỉ hỗ trợ GB10. `make agent-up` chỉ tự
chọn engine cho LLM agent (Atlas hay llama.cpp), không liên quan captioning/OCR.

`make agent-up` tự chọn Atlas hay llama.cpp dựa vào `nvidia-smi` (GB10 →
Atlas). Đổi model qua `ATLAS_MODEL` / `LLAMACPP_MODEL` trong `.env`.

## Agent

Hai đường agent, chung một backend LLM (cổng 8888):

- **Trả lời VQA** (`agent/react.py`): vòng ReAct (tối đa 5 bước) với tool
  `caption`/`ocr` gọi Docker VLM cổng 8881. Caption đã cache lúc index được
  truyền kèm làm ngữ cảnh, nên LLM text-only vẫn trả lời được kể cả khi VLM
  không chạy.
- **Chat tìm kiếm tương tác** (`agent/interactive.py`, `POST /api/agent/chat`
  + khung chat trong UI): vòng lặp thu hẹp kiểu AIC_2025 — tool `search_kis`,
  `search_asr`, `search_ocr`, `subagent_summarize`, `ask_user`; tối đa 6 vòng
  tool mỗi lượt; stateless (trình duyệt giữ hội thoại).

Xử lý query (`retrieval/query_processor.py`) dùng cùng LLM để dịch tiếng Việt
→ tiếng Anh và trích keyword OCR/ASR; tự động rơi về pass-through khi server
không chạy và tự tắt trong phiên sau lần lỗi đầu tiên (bài thi không được
phép treo vì một service phụ).

## Bố cục dự án

```
src/
  cli/            # giao diện dòng lệnh (xem `make help`)
  config/         # settings, đặt tên experiment, nạp .env
  core/           # logging, errors, các record có kiểu
  video/          # quét video, tách shot, trích frame (OpenCV/TransNetV2)
  indexing/       # các bước pipeline offline + manifest + job state SQLite
  retrieval/      # tìm kiếm online: Retriever, SRRF fusion, temporal search, tracks, VQA/TRAKE
  modules/        # backend model: embedding (beit3/siglip/vietnamese),
                  #   reranker (blip2_itm/vietnamese), captioning+ocr (vLLM), asr (gipformer)
  agent/          # brain (LLM Docker), vòng ReAct VQA, agent chat tương tác, tools
  stores/
    vector/       # vector index Qdrant
    text/         # Elasticsearch BM25 (OCR/ASR), nối vào qua `extract-text`
  repository/     # truy cập dữ liệu manifest (frames, videos, captions)
  prompts/        # prompt template LLM/VLM (captioning, ocr, agent)
  ui/             # UI trình duyệt local (các track + chat agent)
tests/unit/       # unit test
docs/vi/          # tài liệu tiếng Việt
```

## Yêu cầu

- Python ≥ 3.13, quản lý bằng [uv](https://docs.astral.sh/uv/)
- GPU NVIDIA + CUDA (embedder + TransNetV2 cần CUDA khi `--device auto`)
- Docker (Qdrant, Elasticsearch, toàn bộ LLM/VLM serving)
- Trọng số TransNetV2 tại
  `external/TransNetV2/inference-pytorch/transnetv2-pytorch-weights.pth`
- Cho ASR: chạy 1 lần `cd external/gipformer && uv sync` (repo+venv cô lập),
  và `uv pip install onnxruntime` (cố ý KHÔNG đưa vào `pyproject.toml` — thêm
  vào đó sẽ âm thầm hạ cấp bản torch `+cu128` đã pin)

## Cài đặt

```bash
uv sync                       # cài dependency
cp .env.example .env          # cấu hình endpoint (mặc định chạy được local)
make qdrant-up qdrant-health  # vector DB
make agent-up agent-health    # LLM agent (tự chọn Atlas hay llama.cpp)
```

## Chạy pipeline

`EXP` là tên experiment (run); chạy `make help` để xem đủ lệnh.

```bash
# Toàn bộ pipeline offline (chỉ vector index)
make pipeline EXP=demo INPUT=data/raw_videos

# Hoặc từng bước
make ingest         EXP=demo INPUT=data/raw_videos
make detect-shots   EXP=demo
make extract-frames EXP=demo
make embed-frames   EXP=demo      # BEiT-3 large; lần đầu tự tải checkpoint ~2GB
make build-index    EXP=demo      # cần Qdrant đang chạy

# Nhánh OCR/ASR (riêng; cần atlas-index + Elasticsearch — chỉ GB10)
make atlas-index-up elasticsearch-up
make extract-text EXP=demo
make export-text  EXP=demo        # xuất ES -> manifests/text.jsonl (chia sẻ được)
make import-text  EXP=demo        # nạp text.jsonl ngược vào ES

# Tìm kiếm / UI
make search   EXP=demo QUERY="a person riding a motorbike"
make serve-ui EXP=demo            # http://127.0.0.1:7860 (các track + chat agent)
```

Mỗi bước ghi tiến độ vào `runs/<EXP>/jobs.sqlite`; chạy lại sẽ bỏ qua phần đã
xong. Dùng `--force` (CLI gốc) để làm lại một bước.

### Re-index khi đổi model embedding

Đổi `--embedding-models` là đổi không gian embedding, nên phải chạy lại:

```bash
uv run codenova embed-frames --experiment-name <EXP> --embedding-models beit3 --force
uv run codenova build-index  --experiment-name <EXP> --embedding-models beit3
```

`build-index` xóa và tạo lại collection Qdrant từ đầu nên không còn vector cũ
sót lại.

## Cấu hình

Hai tầng — đừng trộn lẫn:

- **`.env`** — hạ tầng: endpoint service, cổng, ID model được host. Theo từng
  máy, git-ignore, nạp tự động (xem `.env.example` có chú thích đầy đủ).
- **CLI flags** — thiết lập theo experiment, ghi vào `runs/<exp>/config.json`.
  Đổi flag là đổi danh tính experiment.

Tùy chọn pipeline chính (CLI flag, giá trị mặc định):

| Flag | Mặc định | Ý nghĩa |
|------|----------|---------|
| `--embedding-models` | `beit3` | Danh sách embedder phân tách bằng phẩy (`beit3`, `siglip2`, `vietnamese-embedding`); >1 model bật SRRF fusion + rerank BLIP-2 |
| `--frame-sampling` | `shot-percentile` | Chiến lược lấy keyframe |
| `--keyframe-percentiles` | `0.15,0.5,0.85` | Vị trí lấy keyframe trong mỗi shot |
| `--index-backend` | `qdrant` | Backend vector index |
| `--top-k` | `20` | Số kết quả |
| `--device` | `auto` | Thiết bị torch (`auto` cần CUDA) |

`serve-ui` có thêm: `--reranker-model` / `--reranker-top-k` để bật bước rerank
BLIP-2 trong UI.

## Backend lưu trữ

- **`stores/vector`** — Qdrant. Embedding chuẩn hóa L2, khoảng cách cosine,
  mỗi model một named vector. Mỗi experiment một collection:
  `{QDRANT_COLLECTION}__{experiment}`. Metadata được gắn từ manifest lúc query.
- **`stores/text`** — Elasticsearch (BM25) cho document OCR/ASR, một index
  chung, field `source` phân biệt ocr/asr. Được nạp bởi `make extract-text`.

## Phát triển

```bash
make lint        # ruff check
make format      # ruff format
make check       # lint + kiểm tra format
make test        # pytest
make precommit   # toàn bộ pre-commit hook
```

## Sản phẩm của một run

```
runs/<experiment>/
  config.json
  jobs.sqlite
  logs/{pipeline.log, errors.log}
  manifests/{videos,shots,frames,embeddings}.jsonl  (+captions.jsonl, text.jsonl)
  frames/<video_id>/*.jpg
  embeddings/{frames.npz, frame_ids.json}           (file theo model khi chạy nhiều model)
```

Vector index nằm trong Qdrant (`qdrant_storage/`), text nằm trong
Elasticsearch (Docker volume). `runs/`, `data/`, `external/`,
`qdrant_storage/` và `.env` đều git-ignore.
