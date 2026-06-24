# Video Retrieval Pipeline

Resumable video retrieval pipeline using shot decomposition and CLIP-style semantic search.

Videos are split into shots, sampled into keyframes, embedded with CLIP, and indexed
in Qdrant. Text queries are embedded and matched against keyframes for known-item
search (KIS).

> 🇻🇳 Tài liệu tiếng Việt: [docs/vi/README.md](docs/vi/README.md)

## Pipeline

```
OFFLINE (indexing)
  ingest → detect-shots → extract-frames → embed-frames → build-index

ONLINE (retrieval)
  text query → CLIP embed → Qdrant search → hydrate metadata → results
```

## Project layout

```
src/
  cli/            # command-line interface
  config/         # settings, experiment naming, .env loading
  core/           # logging, errors, typed records
  video/          # video discovery, shot detection, frame extraction (OpenCV/TransNetV2)
  indexing/       # offline pipeline stages + manifests + SQLite job state
  retrieval/      # online search: Retriever, metadata hydration, contest tracks
  modules/        # AI models: embedding (CLIP) + stubs (asr/ocr/captioning/detection/reranker)
  stores/
    vector/       # Qdrant vector index (interface + backend + factory)
    text/         # Elasticsearch full-text index (interface + backend, not yet wired in)
  repository/     # data access over run manifests
  prompts/        # LLM/VLM prompt templates (stub)
  ui/             # local browser UI
tests/unit/       # unit tests
docs/             # documentation (English) + docs/vi (Vietnamese)
```

## Requirements

- Python ≥ 3.13, managed with [uv](https://docs.astral.sh/uv/)
- NVIDIA GPU + CUDA (CLIP and TransNetV2 require CUDA when `--device auto`)
- Docker (for Qdrant)
- TransNetV2 PyTorch weights (see [docs/transnetv2.md](docs/transnetv2.md))

## Setup

```bash
uv sync                       # install dependencies
cp .env.example .env          # configure Qdrant / API keys
make qdrant-up                # start Qdrant (docker compose)
make qdrant-health            # -> healthz check passed
```

Verify CUDA:

```bash
uv run python - <<'PY'
import torch
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

## Running the pipeline

Every step runs through `uv`. The `Makefile` wraps the common commands — run `make help`
to list them. `EXP` selects the experiment (run) name.

```bash
# Full offline pipeline (ingest → ... → build-index)
make pipeline EXP=demo INPUT=data/raw_videos

# Or step by step
make ingest         EXP=demo INPUT=data/raw_videos
make detect-shots   EXP=demo
make extract-frames EXP=demo
make embed-frames   EXP=demo
make build-index    EXP=demo      # needs Qdrant running

# Search / UI
make search   EXP=demo QUERY="a person riding a motorbike"
make serve-ui EXP=demo            # http://127.0.0.1:7860
```

Each stage records progress in `runs/<EXP>/jobs.sqlite` and the manifests, so re-running a
stage skips completed work. Pass `--force` (on the raw CLI) to redo a stage.

## Configuration

Config splits in two:

- **`.env`** — infrastructure: service endpoints, ports, credentials. Per-machine, loaded
  automatically (see `.env.example`).
- **CLI flags** — per-experiment settings (model, keyframes, top-k). Recorded in
  `runs/<exp>/config.json` so each run is reproducible. These are intentionally *not* in `.env`.

`.env` (loaded automatically):

```bash
# Vector DB (Qdrant)
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=codenova_frames
QDRANT_API_KEY=                    # empty for local; set for Qdrant Cloud

# Full-text index (Elasticsearch — for OCR/ASR, not yet wired in)
ELASTIC_URL=http://localhost:9200
ELASTIC_INDEX=codenova_text
ELASTIC_API_KEY=

# Retrieval UI defaults
CODENOVA_UI_HOST=127.0.0.1
CODENOVA_UI_PORT=7860
```

Key pipeline options (CLI flags, defaults shown):

| Flag | Default | Meaning |
|------|---------|---------|
| `--clip-model` | `clip-vit-b-32` | CLIP model for embeddings |
| `--frame-sampling` | `shot-percentile` | Keyframe sampling strategy |
| `--keyframe-percentiles` | `0.15,0.5,0.85` | Where in each shot to sample keyframes |
| `--index-backend` | `qdrant` | Vector index backend |
| `--top-k` | `20` | Number of results |
| `--device` | `auto` | Torch device (`auto` requires CUDA) |

### Keyframe sampling

Each shot is sampled at the configured percentiles: frame index `= start + round(span * p)`
for each percentile `p`. The default `0.15, 0.5, 0.85` yields three keyframes per shot
(near-start, middle, near-end). Duplicate indices for very short shots collapse to a single
keyframe.

## Storage backends

Storage lives under `stores`, each backend behind an interface so new ones can be
added without touching the pipeline:

- **`stores/vector`** — Qdrant. Embeddings are L2-normalized so cosine distance ranks like
  inner product. Each experiment uses its own collection: `{QDRANT_COLLECTION}__{experiment}`.
  Vector + `frame_id` are stored; metadata is hydrated from manifests at query time.
- **`stores/text`** — Elasticsearch (BM25) for OCR/ASR text. Interface + backend exist but
  are **not yet wired into the pipeline** (no OCR/ASR text is produced yet). Install with
  the `text` extra: `uv pip install -e '.[text]'`.

## Development

```bash
make lint        # ruff check
make format      # ruff format
make check       # lint + format-check
make test        # pytest
make precommit   # all pre-commit hooks
```

## Run artifacts

```
runs/<experiment>/
  config.json
  jobs.sqlite
  logs/{pipeline.log, errors.log}
  manifests/{videos,shots,frames,embeddings}.jsonl
  frames/<video_id>/*.jpg
  embeddings/{frames.npz, frame_ids.json}
```

The vector index is not on disk — it lives in Qdrant (`qdrant_storage/`).
`runs/`, `data/`, `external/`, `qdrant_storage/`, and `.env` are git-ignored.

## Documentation

- [docs/qdrant.md](docs/qdrant.md) — Qdrant usage
- [docs/transnetv2.md](docs/transnetv2.md) — preparing TransNetV2 weights
- [docs/vi/](docs/vi/) — Vietnamese documentation
