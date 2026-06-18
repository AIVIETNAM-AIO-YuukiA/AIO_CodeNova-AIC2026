# CodeNova (Tiếng Việt)

CodeNova là pipeline truy hồi video: tách video thành shot, lấy keyframe, embed bằng CLIP
và index vào Qdrant; truy vấn bằng text để tìm đúng khoảnh khắc (known-item search, KIS).

Pipeline chạy tiếp được sau khi gián đoạn: mỗi bước ghi tiến độ (vào `jobs.sqlite` và các
manifest), nên khi chạy lại sẽ bỏ qua phần đã hoàn tất thay vì làm lại từ đầu.

> Bản tiếng Anh: [../../README.md](../../README.md)

## Pipeline

```
OFFLINE (indexing)
  ingest → detect-shots → extract-frames → embed-frames → build-index

ONLINE (retrieval)
  text query → CLIP embed → Qdrant search → hydrate metadata → kết quả
```

## Cấu trúc thư mục

```
src/
  cli/            # giao diện dòng lệnh
  config/         # cấu hình, đặt tên experiment, nạp .env
  core/           # logging, errors, kiểu dữ liệu
  video/          # quét video, phát hiện shot, trích keyframe (OpenCV/TransNetV2)
  indexing/       # các stage offline + manifest + trạng thái job (SQLite)
  retrieval/      # search online: Retriever, hydrate metadata, contest tracks
  modules/        # model AI: embedding (CLIP) + stub (asr/ocr/captioning/detection/reranker)
  stores/
    vector/       # Qdrant vector index (interface + backend + factory)
    text/         # Elasticsearch full-text (interface + backend, chưa nối vào luồng)
  repository/     # tầng truy cập dữ liệu trên manifest
  prompts/        # prompt template cho LLM/VLM (stub)
  ui/             # UI trình duyệt cục bộ
tests/unit/       # unit test
docs/             # tài liệu tiếng Anh + docs/vi tiếng Việt
```

## Yêu cầu

- Python ≥ 3.13, quản lý bằng [uv](https://docs.astral.sh/uv/)
- GPU NVIDIA + CUDA (CLIP và TransNetV2 cần CUDA khi `--device auto`)
- Docker (cho Qdrant)
- Weights TransNetV2 PyTorch (xem [transnetv2.md](transnetv2.md))

## Cài đặt

```bash
uv sync                       # cài dependencies
cp .env.example .env          # cấu hình Qdrant / API keys
make qdrant-up                # bật Qdrant (docker compose)
make qdrant-health            # -> healthz check passed
```

## Chạy pipeline

Mọi lệnh chạy qua `uv`. `Makefile` gói các lệnh thường dùng — chạy `make help` để xem.
`EXP` là tên experiment (run).

```bash
# Toàn bộ pipeline offline
make pipeline EXP=demo INPUT=data/raw_videos

# Hoặc từng bước
make ingest         EXP=demo INPUT=data/raw_videos
make detect-shots   EXP=demo
make extract-frames EXP=demo
make embed-frames   EXP=demo
make build-index    EXP=demo      # cần Qdrant đang chạy

# Tìm kiếm / UI
make search   EXP=demo QUERY="a person riding a motorbike"
make serve-ui EXP=demo            # http://127.0.0.1:7860
```

Mỗi bước ghi tiến độ vào `runs/<EXP>/jobs.sqlite` và manifest, nên chạy lại sẽ bỏ qua phần
đã xong. Thêm `--force` (trên CLI gốc) để làm lại một bước.

## Cấu hình

Cấu hình chia hai loại:

- **`.env`** — hạ tầng: endpoint service, port, credential. Theo máy, tự động nạp (xem `.env.example`).
- **Cờ CLI** — cấu hình theo experiment (model, keyframe, top-k). Ghi vào `runs/<exp>/config.json`
  để mỗi run tái lập được. Các thứ này **cố ý không** ở `.env`.

`.env` (tự động nạp):

```bash
# Vector DB (Qdrant)
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=codenova_frames
QDRANT_API_KEY=                    # để trống khi local; điền khi dùng Qdrant Cloud

# Full-text index (Elasticsearch — cho OCR/ASR, chưa nối vào luồng)
ELASTIC_URL=http://localhost:9200
ELASTIC_INDEX=codenova_text
ELASTIC_API_KEY=

# Mặc định cho UI
CODENOVA_UI_HOST=127.0.0.1
CODENOVA_UI_PORT=7860
```

Các tùy chọn pipeline chính (cờ CLI, kèm mặc định):

| Cờ | Mặc định | Ý nghĩa |
|------|---------|---------|
| `--clip-model` | `clip-vit-b-32` | Model CLIP để embed |
| `--frame-sampling` | `shot-percentile` | Chiến lược lấy keyframe |
| `--keyframe-percentiles` | `0.15,0.5,0.85` | Vị trí trong shot để lấy keyframe |
| `--index-backend` | `qdrant` | Backend vector index |
| `--top-k` | `20` | Số kết quả |
| `--device` | `auto` | Thiết bị torch (`auto` cần CUDA) |

### Lấy keyframe theo percentile

Mỗi shot được lấy mẫu tại các percentile cấu hình: vị trí frame `= start + round(span * p)`
cho từng percentile `p`. Mặc định `0.15, 0.5, 0.85` cho ra ba keyframe mỗi shot (gần đầu,
giữa, gần cuối). Với shot rất ngắn, các index trùng nhau sẽ gộp lại còn một keyframe.

## Storage backend

Storage nằm trong `stores`, mỗi backend nằm sau một interface để thêm backend mới
mà không phải sửa pipeline:

- **`stores/vector`** — Qdrant. Embedding đã L2-normalize nên cosine distance xếp hạng như
  inner product. Mỗi experiment dùng collection riêng: `{QDRANT_COLLECTION}__{experiment}`.
  Lưu vector + `frame_id`; metadata được hydrate từ manifest lúc truy vấn.
- **`stores/text`** — Elasticsearch (BM25) cho text OCR/ASR. Đã có interface + backend nhưng
  **chưa nối vào pipeline** (chưa sinh text OCR/ASR). Cài qua extra `text`:
  `uv pip install -e '.[text]'`.

Không dùng MongoDB: Qdrant payload + manifest đã đủ cho metadata.

## Phát triển

```bash
make lint        # ruff check
make format      # ruff format
make check       # lint + kiểm tra format
make test        # pytest
make precommit   # tất cả pre-commit hooks
```

## Artifact của một run

```
runs/<experiment>/
  config.json
  jobs.sqlite
  logs/{pipeline.log, errors.log}
  manifests/{videos,shots,frames,embeddings}.jsonl
  frames/<video_id>/*.jpg
  embeddings/{frames.npz, frame_ids.json}
```

Vector index không nằm trên đĩa — nó ở trong Qdrant (`qdrant_storage/`).
`runs/`, `data/`, `external/`, `qdrant_storage/`, và `.env` đều bị git ignore.

## Tài liệu

- [qdrant.md](qdrant.md) — sử dụng Qdrant
- [transnetv2.md](transnetv2.md) — chuẩn bị weights TransNetV2
