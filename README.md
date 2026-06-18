# AIO_CodeNova-AIC2026

## Mục tiêu dự án

Dự án xây dựng pipeline truy xuất video bằng ngôn ngữ tự nhiên:

```text
query text
  -> CLIP text embedding
  -> tìm frame/keyframe gần nhất trong vector index
  -> trả về video, timestamp, frame, score
```

Pipeline ban đầu được thiết kế theo hướng:

- dùng TransNetV2 để tách video thành shot;
- trích xuất keyframe đại diện cho từng shot;
- dùng CLIP để embed keyframe và query;
- lưu embedding vào vector index;
- hỗ trợ chạy dài ngày bằng manifest + SQLite job state;
- mỗi lần chạy có experiment name hợp lệ, ổn định và dễ so sánh.

## Cấu trúc chính

```text
src/
  config/       # cấu hình và experiment naming
  core/         # logging, errors, typed records
  video/        # video discovery, shot detection, frame extraction
  indexing/     # offline pipeline: ingest -> shots -> frames -> embeddings -> build_index
  retrieval/    # online: search, metadata hydration, contest tracks
  modules/      # AI models: embedding (CLIP) + stub asr/ocr/captioning/detection/reranker
  index/        # vector index interface + Qdrant backend + factory
  repository/   # data access layer (đọc records từ manifest)
  prompts/      # prompt templates cho LLM/VLM (stub)
  cli/          # command-line interface
```

Các backend nặng được tách sau interface riêng:

- TransNetV2: chạy qua PyTorch implementation khi truyền `--transnetv2-module-dir` và `--transnetv2-weights`;
- OpenCV: đọc video và xuất keyframe;
- CLIP: dùng Hugging Face Transformers + PyTorch CUDA;
- Qdrant: vector index chạy như một service riêng (xem [docs/qdrant.md](docs/qdrant.md)).

## Experiment naming

Experiment name phải:

- chỉ dùng chữ thường, số, `_`, `-`;
- không có khoảng trắng hoặc path separator;
- dài 3-120 ký tự;
- có config hash để tránh nhầm lẫn giữa các cấu hình.

Tạo tên mặc định:

```bash
uv run codenova name-experiment
```

Ví dụ:

```text
20260612_retrieval_clip-vit-b-32_shot-midpoint_qdrant_506f72d1
```

Kiểm tra tên:

```bash
uv run codenova validate-experiment-name 20260612_retrieval_clip-vit-b-32_shot-midpoint_qdrant_506f72d1
```

## Kiểm tra GPU

Một số package là GPU-capable, một số chỉ là package hỗ trợ CPU:

| Package | GPU? | Ghi chú |
|---------|------|---------|
| `torch`, `torchvision` | Có | Cài CUDA-enabled wheel bằng `uv` |
| CLIP qua `transformers` | Có | Model chạy trên CUDA thông qua PyTorch |
| TransNetV2 PyTorch | Có | Code yêu cầu CUDA khi `--device auto` |
| `qdrant-client` | N/A | Client kết nối Qdrant service; tăng tốc nằm ở server |
| `opencv-python` | Không chính | Dùng để decode video và ghi ảnh |
| `pillow`, `numpy` | Không chính | Dùng cho load ảnh và array CPU |

Kiểm tra CUDA:

```bash
uv run python - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch cuda runtime:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

Kết quả mong đợi nếu máy có GPU NVIDIA và CUDA hoạt động:

```text
cuda available: True
device count: <number-of-gpus>
device: <gpu-name>
```

`nvidia-smi` kiểm tra driver NVIDIA. `torch.cuda.is_available()` kiểm tra Python/PyTorch có khởi tạo CUDA được trong environment hiện tại không. Cả hai nên hoạt động.

## Chuẩn bị TransNetV2

TransNetV2 không nằm trực tiếp trong repo chính. Clone vào thư mục `external/`:

```bash
mkdir -p external
git clone https://github.com/soCzech/TransNetV2 external/TransNetV2
```

Weights của TransNetV2 dùng Git LFS. Nếu thấy lỗi `Wire format was corrupt` khi convert weights, thường là do file weights chỉ là LFS pointer. Trên Ubuntu/Debian, cài Git LFS và pull file thật:

```bash
sudo apt install git-lfs
git lfs install

cd <repo-root>
git -C external/TransNetV2 lfs pull
```

Kiểm tra file weights không còn là file rất nhỏ:

```bash
ls -lh external/TransNetV2/inference/transnetv2-weights/saved_model.pb
ls -lh external/TransNetV2/inference/transnetv2-weights/variables/
```

Convert TensorFlow weights sang PyTorch weights bằng Python 3.10 tạm thời qua `uv`:

```bash
cd external/TransNetV2/inference-pytorch

uv run --no-project --python 3.10 \
  --with torch \
  --with tensorflow \
  --with numpy \
  python convert_weights.py \
  --tf_weights ../inference/transnetv2-weights

cd <repo-root>
```

File cần có sau khi convert:

```text
external/TransNetV2/inference-pytorch/transnetv2-pytorch-weights.pth
```

Các warning TensorFlow như `Could not find cuda drivers` trong bước convert không quan trọng. Bước convert có thể chạy CPU. Pipeline chính dùng PyTorch CUDA sau đó.

## Chạy pipeline retrieval

Đặt thư mục video input. Mặc định nên dùng:

```text
data/raw_videos
```

Tạo experiment name hợp lệ:

```bash
export EXPERIMENT=$(uv run codenova name-experiment)
echo "$EXPERIMENT"
```

### 1. Ingest video

```bash
uv run codenova ingest \
  --input data/raw_videos \
  --experiment-name "$EXPERIMENT" \
  --resume
```

### 2. Detect shots

```bash
uv run codenova detect-shots \
  --experiment-name "$EXPERIMENT" \
  --transnetv2-module-dir external/TransNetV2/inference-pytorch \
  --transnetv2-weights external/TransNetV2/inference-pytorch/transnetv2-pytorch-weights.pth
```

Nếu gặp CUDA fragmentation/OOM:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run codenova detect-shots \
  --experiment-name "$EXPERIMENT" \
  --transnetv2-module-dir external/TransNetV2/inference-pytorch \
  --transnetv2-weights external/TransNetV2/inference-pytorch/transnetv2-pytorch-weights.pth
```

Code hiện chạy TransNetV2 theo window 100 frames và giữ center 50 predictions, giống official inference wrapper, để tránh đưa toàn bộ video lên GPU một lần.

### 3. Extract keyframes

```bash
uv run codenova extract-frames \
  --experiment-name "$EXPERIMENT"
```

Kiểm tra số frame:

```bash
wc -l "runs/$EXPERIMENT/manifests/frames.jsonl"
```

Với `keyframes_per_shot=1`, số dòng dự kiến xấp xỉ số shot.

### 4. Embed frames bằng CLIP

Bắt đầu bằng batch size nhỏ nếu GPU có VRAM hạn chế:

```bash
uv run codenova embed-frames \
  --experiment-name "$EXPERIMENT" \
  --batch-size 8
```

Nếu CUDA OOM, giảm `--batch-size` xuống `4` hoặc `2`.

### 5. Build Qdrant vector index

Index dùng **Qdrant** (chạy như một service riêng). Bật Qdrant trước khi build:

```bash
docker compose up -d qdrant
curl http://localhost:6333/healthz   # -> healthz check passed
```

Connection lấy từ `.env` ở thư mục gốc (xem `.env.example`). Chi tiết: [docs/qdrant.md](docs/qdrant.md).

```bash
uv run codenova build-index \
  --experiment-name "$EXPERIMENT"
```

### 6. Search

```bash
uv run codenova search \
  --experiment-name "$EXPERIMENT" \
  "a person riding a motorbike"
```

### 7. Serve local UI

The UI shows a query form, retrieval-track selector, and result frame images:

```bash
uv run codenova serve-ui \
  --experiment-name "$EXPERIMENT" \
  --host 127.0.0.1 \
  --port 7860
```

Open:

```text
http://127.0.0.1:7860
```

Supported UI tracks:

| Track | Current behavior |
|-------|------------------|
| Textual KIS | Uses the query/context text to search CLIP frame embeddings |
| VQA | Combines scene context + question + query, then searches CLIP frame embeddings |
| Question Answering | Combines question + context + query, then searches CLIP frame embeddings |
| Visual KIS | Uses query/context text until image-query support is added |

Example Textual KIS query:

```text
A sequence of shots taken from a moving motorbike. In the first shot we see a
view under the rider's left arm, the handlebar, both mirrors and the rider's hands
are visible.
```

Example VQA context/question:

```text
Context:
Two women are having a conversation over the telephone. One is trying to hang
pictures from the wall and is asking the other for advice. The video cuts back
and forth between them. The woman with the pictures then drives to a hardware store.

Question:
What are the names of the two women?
```

Important: the UI is track-aware, but the backend currently routes every track through CLIP text-to-frame retrieval. It is useful for inspecting evidence frames now; it is not yet a full VQA answer generator.

Search results include submission-friendly metadata:

```text
video_name      # original file name, e.g. sample.mp4
video_path      # original file path from videos.jsonl
frame_index     # frame number in the source video
timestamp_sec   # timestamp in seconds
shot_id         # detected shot id
frame_id        # internal keyframe id
score           # CLIP/Qdrant cosine similarity score
```

## Feature status

| Feature | Status | Notes |
|---------|--------|-------|
| Video ingest | Done | Discovers videos and writes `videos.jsonl` |
| Shot detection | Done | Uses TransNetV2 PyTorch with windowed inference |
| Keyframe extraction | Done | Uses OpenCV, currently one or more frames per shot |
| CLIP embeddings | Done | Uses Transformers + PyTorch CUDA |
| Qdrant search | Done | Searches existing embedded keyframes via Qdrant |
| Local UI | Done | Supports track selector, query fields, result images, and video/frame metadata |
| Query result persistence | Not yet | UI searches are temporary and are not saved automatically |
| Batch query testing | Not yet | Needs a `batch-search` command |
| VQA answer generation | Not yet | Current VQA mode retrieves evidence frames only |
| Visual query upload | Not yet | Current Visual KIS mode still uses text fields |
| OCR/ASR indexing | Not yet | Needed for names, signs, dialogue, and text-heavy questions |
| Incremental index update | Not yet | Current safe path is to rebuild the Qdrant collection |

## Large query testing

The current CLI and UI can run many queries manually, but they are not optimized for large query sets. For hundreds or thousands of queries, add a dedicated batch runner that loads CLIP and metadata once (reusing a single `Retriever`), then loops through all queries.

Proposed input format:

```jsonl
{"query_id":"q001","track":"textual_kis","query":"a person riding a motorbike","top_k":20}
{"query_id":"q002","track":"vqa","context":"Two women talk on the phone...", "question":"What are the names of the two women?", "top_k":20}
```

Proposed command:

```bash
uv run codenova batch-search \
  --experiment-name "$EXPERIMENT" \
  --queries queries/sample_queries.jsonl \
  --output "runs/$EXPERIMENT/query_results/results.jsonl" \
  --top-k 20
```

Proposed output format:

```jsonl
{"query_id":"q001","track":"textual_kis","retrieval_text":"a person riding a motorbike","results":[...]}
{"query_id":"q002","track":"vqa","retrieval_text":"Two women talk on the phone... What are the names of the two women?","results":[...]}
```

The batch runner should support:

- resume by skipping completed `query_id`;
- per-query errors without stopping the whole run;
- JSONL output for easy evaluation;
- optional metrics when labels are available;
- loading CLIP/metadata once per process (Qdrant runs as a shared service).

## Backend work needed for multiple contest tracks

The UI keeps track-specific fields separate so the backend can evolve without changing the browser contract. The main backend gaps are:

- Add a `TrackQuery`/result schema to persisted search logs, not only UI requests.
- Add video/shot-level aggregation so Textual KIS can return ranked video segments, not only individual frames.
- Add VQA answer generation after evidence retrieval, likely using an image/video-language model over top frames or shot clips.
- Add OCR/ASR metadata indexes for questions involving names, signs, spoken dialogue, or on-screen text.
- Add visual-query support for Visual KIS, including image upload, image embedding, and image-to-frame search.
- Add reranking that can combine CLIP score, temporal continuity, OCR/ASR hits, and track-specific signals.
- Add export formats expected by the contest submission system.
- Add evaluation scripts per track, for example Recall@K for KIS and answer accuracy/evidence recall for VQA.

## Backend performance and scaling notes

### ONNX and TensorRT

Converting models to ONNX/TensorRT is a useful benchmark path, but it should be treated as an experiment, not assumed faster upfront.

Best candidates:

- CLIP image/text encoders: likely useful for faster embedding and query encoding.
- TransNetV2: possible, but the current PyTorch windowed inference may already be acceptable compared with video decoding cost.

Recommended plan:

1. Export each model separately to ONNX.
2. Validate numerical similarity against PyTorch outputs.
3. Benchmark throughput and latency on the target GPU.
4. Convert ONNX to TensorRT only if ONNX/runtime benchmarks justify it.
5. Keep PyTorch as the reference backend for correctness.

### Large datasets and storage

Large datasets can create storage pressure because the pipeline saves:

- extracted JPEG keyframes;
- CLIP embeddings;
- manifests and logs.

(Vector index nằm trong Qdrant — `qdrant_storage/` — không phải trong `runs/`.)

Main risks:

- too many keyframes per shot;
- high JPEG quality or duplicate frames;
- rebuilding experiments instead of resuming;
- storing every sweep/run separately without cleanup.

Mitigations:

- keep `keyframes_per_shot=1` as the baseline;
- store frames under `runs/<experiment>/frames`;
- keep manifests as source of truth for resumability;
- periodically remove failed/obsolete runs;
- for very large data, consider sharded experiments and indexes.

### Adding new data

`build-index` đọc embeddings đã lưu rồi recreate + upsert vào Qdrant. Khi thêm video/keyframe mới, workflow an toàn nhất là:

```text
ingest new videos -> detect shots -> extract frames -> embed frames -> rebuild index
```

`build-index` luôn recreate collection nên việc rebuild là idempotent. Sau này có thể thêm incremental updater chỉ upsert phần embeddings mới thay vì rebuild toàn bộ.

### Vector database backend

Index dùng **Qdrant** qua interface `VectorIndex` ([src/index/base.py](src/index/base.py)):

```text
VectorIndex
  build(embeddings, frame_ids)
  search(query_embedding, top_k)
```

`QdrantVectorIndex` ([src/index/qdrant_index.py](src/index/qdrant_index.py)) là implementation hiện tại; `build_vector_index()` ([src/index/factory.py](src/index/factory.py)) dựng nó từ config + biến môi trường. Muốn thêm backend khác (Milvus, LanceDB, ...) chỉ cần implement `VectorIndex` và mở rộng factory — phần `indexing/` và `retrieval/` không phải đổi.

Embeddings được L2-normalize nên Qdrant dùng cosine distance (tương đương inner-product). Mỗi experiment dùng collection riêng: `{QDRANT_COLLECTION}__{experiment_name}`.

## Artifacts của một run

Kết quả được lưu theo từng experiment:

```text
runs/<experiment-name>/
  config.json
  jobs.sqlite
  logs/
    pipeline.log
    errors.log
  manifests/
    videos.jsonl
    shots.jsonl
    frames.jsonl
    embeddings.jsonl
  frames/
  embeddings/
    frames.npz
    frame_ids.json
```

Vector index không nằm trong `runs/` mà trong Qdrant (collection `{QDRANT_COLLECTION}__<experiment-name>`, lưu tại `qdrant_storage/`).

`runs/`, `data/`, `external/`, `qdrant_storage/`, và `.env` đều được ignore khỏi git.

## Resume và chạy lại stage

```bash
uv run codenova ingest \
  --input data/raw_videos \
  --experiment-name "$EXPERIMENT" \
  --resume
```

Các stage có `--force` để chạy lại nếu artifact/state đã tồn tại:

```bash
uv run codenova extract-frames \
  --experiment-name "$EXPERIMENT" \
  --force
```

## Troubleshooting

### `No frames found to embed`

Nghĩa là chưa có hoặc chưa ghi được:

```text
runs/<experiment-name>/manifests/frames.jsonl
```

Chạy theo đúng thứ tự:

```text
ingest -> detect-shots -> extract-frames -> embed-frames -> build-index -> search
```

Kiểm tra:

```bash
wc -l "runs/$EXPERIMENT/manifests/shots.jsonl"
wc -l "runs/$EXPERIMENT/manifests/frames.jsonl"
```

### `Wire format was corrupt` khi convert TransNetV2

Nguyên nhân thường là chưa pull Git LFS weights. Chạy:

```bash
git -C external/TransNetV2 lfs pull
```

Nếu `git: 'lfs' is not a git command`, cài:

```bash
sudo apt install git-lfs
git lfs install
```

### TensorFlow báo không thấy CUDA khi convert

Bỏ qua được. Bước convert weights chạy CPU vẫn ổn. Pipeline inference chính dùng PyTorch CUDA.

### TransNetV2 CUDA OOM

Code đã dùng windowed inference để giảm VRAM. Nếu vẫn lỗi, chạy:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run codenova detect-shots \
  --experiment-name "$EXPERIMENT" \
  --transnetv2-module-dir external/TransNetV2/inference-pytorch \
  --transnetv2-weights external/TransNetV2/inference-pytorch/transnetv2-pytorch-weights.pth
```

### CLIP CUDA OOM

Giảm batch size:

```bash
uv run codenova embed-frames \
  --experiment-name "$EXPERIMENT" \
  --batch-size 4
```

### `nvidia-smi` chạy được nhưng `torch.cuda.is_available()` là `False`

`nvidia-smi` chỉ chứng minh driver thấy GPU. PyTorch còn cần CUDA runtime trong Python environment hiện tại. Kiểm tra bằng lệnh ở mục "Kiểm tra GPU".

## Yêu cầu

- Python >= 3.13
- [uv](https://docs.astral.sh/uv/) — quản lý môi trường và package

---

## Cài đặt lần đầu

```bash
# 1. Clone repo
git clone <repo-url>
cd AIO_CodeNova-AIC2026

# 2. Cài uv (nếu chưa có)
# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 3. Tạo môi trường và cài dependencies
uv sync

# 4. Cài pre-commit hooks
uv run pre-commit install
```

---

## Quản lý dependencies với uv

```bash
# Thêm package mới
uv add <package-name>

# Thêm package chỉ dùng khi dev (lint, test,...)
uv add --dev <package-name>

# Xóa package
uv remove <package-name>

# Đồng bộ môi trường theo uv.lock (dùng khi pull code mới về)
uv sync

# Chạy script/lệnh trong môi trường
uv run python main.py
```

> Sau khi `uv add` / `uv remove`, nhớ commit cả `pyproject.toml` và `uv.lock`.

---

## Git Workflow

### Quy tắc đặt tên branch

```
feature/<tên-tính-năng>     # thêm tính năng mới
fix/<tên-lỗi>               # sửa bug
chore/<công-việc>           # cập nhật config, docs,...
```

Ví dụ: `feature/data-preprocessing`, `fix/model-output-error`

### Các bước làm việc

```bash
# 1. Luôn cập nhật branch main trước
git checkout main
git pull origin main

# 2. Tạo branch mới từ main
git checkout -b feature/<tên-tính-năng>

# 3. Làm việc, sau đó commit
git add .
git commit -m "feat: mô tả ngắn thay đổi"

# 4. Push branch lên remote
git push origin feature/<tên-tính-năng>

# 5. Tạo Pull Request trên GitHub để merge vào main
```

### Quy tắc commit message

| Prefix | Dùng khi |
|--------|----------|
| `feat:` | Thêm tính năng mới |
| `fix:` | Sửa bug |
| `docs:` | Cập nhật tài liệu |
| `chore:` | Thay đổi config, dependencies |
| `refactor:` | Refactor code |

---

## Makefile shortcuts

```bash
make install      # uv sync — cài/đồng bộ dependencies
make lint         # kiểm tra lỗi code
make format       # tự động format code
make pre-commit   # chạy tất cả pre-commit hooks
make test         # chạy tests
```
