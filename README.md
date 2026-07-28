# CodeNova — Video Retrieval Pipeline (HCMC AI Challenge 2026)

Resumable video retrieval system: videos are split into shots, sampled into
keyframes, embedded with **BEiT-3 large** (384px, dim 1024, COCO-retrieval
fine-tune), and indexed in Qdrant. OCR/ASR text goes into Elasticsearch. An
LLM agent (Qwen3.5-4B, Docker-served) answers VQA questions and drives an
interactive narrowing-loop search chat.

> 🇻🇳 Tài liệu tiếng Việt: [docs/vi/README.md](docs/vi/README.md)

## Pipeline

```
OFFLINE (indexing)
  ingest → detect-shots → extract-frames → embed-frames → build-index
                                          (vector index: BEiT-3 large → Qdrant)
  extract-text  (separate stage: OCR per keyframe + ASR per video → Elasticsearch;
                 needs `make vllm-index-up` + `make elasticsearch-up`)

ONLINE (retrieval)
  text query → [LLM translate/expand] → embed (BEiT-3) → Qdrant search
    → temporal search (frame-to-frame walk → segments) → shot validation
    → track shaping (textual KIS / VQA / TRAKE)
    → [VQA] agent answer  /  [chat] interactive agent loop
```

All generative models (LLM/VLM) are **Docker-served over OpenAI-compatible
HTTP** — no model checkpoint is loaded inside the Python process for agent,
captioning, OCR, or query processing. Embedders/rerankers (BEiT-3, SigLIP2,
BLIP-2) run in-process as batch encoders.

## Models

| Role | Model | Where it runs |
|------|-------|---------------|
| Visual embedding (default) | BEiT-3 `beit3_large_patch16_384_coco_retrieval` (dim 1024) | in-process, TensorRT FP16 (~27x vs PyTorch on GB10) |
| Visual embedding (opt-in) | SigLIP2 `google/siglip2-so400m-patch14-384` (dim 1152) | in-process, TensorRT FP16 (~3.5x vs PyTorch on GB10) |
| Caption-text embedding (opt-in) | `AITeamVN/Vietnamese_Embedding_v2` over VLM captions | captions via Docker VLM; embedding in-process |
| Reranker (opt-in, multi-model runs) | BLIP-2 ITM `Salesforce/blip2-itm-vit-g` | in-process |
| Captioning + OCR (indexing & agent tools) | `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit` | Docker `vllm-index` (vLLM, AWQ/Marlin, **GB10 only**), port 8881 |
| Agent LLM (VQA answer, chat, query expand) | **Qwen3.5-4B 4-bit** | Docker port 8884 — vLLM (AWQ) on DGX Spark/GB10, llama.cpp (GGUF Q4_K_M) elsewhere |
| ASR | gipformer-65M-rnnt + Silero-VAD | subprocess into `external/gipformer/` |
| Shot detection | TransNetV2 (PyTorch) | in-process |

`vllm-agent` (4B, port 8884) and `vllm-index` (35B-A3B, port 8881) are two
independent vLLM containers — separate ports so both run at once without
evicting each other's weights/KV cache from GB10's unified memory pool.
Captioning/OCR has **no non-GB10 fallback**: `vllm-index` needs GB10's
Blackwell (SM121) GPU for its AWQ/Marlin kernel. `make agent-up` picks vLLM
vs llama.cpp automatically from `nvidia-smi` (GB10 → vLLM) for the *agent*
LLM only. Override models via `VLLM_AGENT_MODEL` / `LLAMACPP_MODEL` /
`VLLM_INDEX_MODEL` in `.env`.

### TensorRT-accelerated embedding

BEiT-3 and SigLIP2 spend nearly all their time in the vision tower's forward
pass — measured on GB10, PyTorch eager mode takes ~400-860ms/image, which
would put a 283K-frame corpus at 1-2 days. `modules/embedding/tensorrt_runtime.py`
exports each model's vision tower to ONNX and builds a TensorRT FP16 engine
the first time `embed-frames` runs (a few minutes, one-time), caching it
under `weights/<model>/`. Every later run just loads the cached engine —
verified cosine similarity >=0.9999 against the PyTorch output, ~27x faster
for BEiT-3 and ~3.5x for SigLIP2. Text queries stay on PyTorch (a single
short sequence per call has no batching gain to capture). Disable per-model
with `BEIT3_USE_TENSORRT=0` / `SIGLIP2_USE_TENSORRT=0` in `.env`.

## Agent

Two agent paths, one LLM backend (port 8888):

- **VQA answering** (`agent/react.py`): ReAct loop (max 5 steps) with
  `caption`/`ocr` tools that call the Docker VLM on port 8881. Cached
  index-time captions are passed as context, so the text-only LLM can answer
  even when the VLM service is down.
- **Interactive search chat** (`agent/interactive.py`, `POST /api/agent/chat`
  + chat panel in the UI): the AIC_2025-style narrowing loop — tools
  `search_kis`, `search_asr`, `search_ocr`, `subagent_summarize`, `ask_user`;
  max 6 tool rounds per turn; stateless (the browser keeps the conversation).

Query processing (`retrieval/query_processor.py`) uses the same LLM to
translate Vietnamese → English and extract OCR/ASR keywords; it silently
degrades to pass-through when the server is down and disables itself for the
session after the first failure (a competition run must never block on it).

## Project layout

```
src/
  cli/            # command-line interface (see `make help`)
  config/         # settings, experiment naming, .env loading
  core/           # logging, errors, typed records
  video/          # video discovery, shot detection, frame extraction (OpenCV/TransNetV2)
  indexing/       # offline pipeline stages + manifests + SQLite job state
  retrieval/      # online search: Retriever, SRRF fusion, temporal search, tracks, VQA/TRAKE
  modules/        # model backends: embedding (beit3/siglip/vietnamese),
                  #   reranker (blip2_itm/vietnamese), captioning+ocr (vLLM), asr (gipformer)
  agent/          # brain (Docker LLM), ReAct VQA loop, interactive chat agent, tools
  stores/
    vector/       # Qdrant vector index
    text/         # Elasticsearch BM25 (OCR/ASR), wired in via `extract-text`
  repository/     # data access over run manifests (frames, videos, captions)
  prompts/        # LLM/VLM prompt templates (captioning, ocr, agent)
  ui/             # local browser UI (tracks + agent chat)
tests/unit/       # unit tests
docs/vi/          # Vietnamese documentation
```

## Requirements

- Python ≥ 3.13, managed with [uv](https://docs.astral.sh/uv/)
- NVIDIA GPU + CUDA (embedders + TransNetV2 need CUDA when `--device auto`)
- Docker (Qdrant, Elasticsearch, all LLM/VLM serving)
- TransNetV2 PyTorch weights at
  `external/TransNetV2/inference-pytorch/transnetv2-pytorch-weights.pth`
- For ASR: one-time `cd external/gipformer && uv sync` (isolated repo+venv),
  and `uv pip install onnxruntime` (deliberately NOT in `pyproject.toml` —
  adding it there silently downgrades the pinned `+cu128` torch build)
- For TensorRT-accelerated embedding (optional, on by default — see Models
  table below): `uv pip install onnx onnxscript tensorrt` (same reasoning as
  onnxruntime above — not in `pyproject.toml`). Set `BEIT3_USE_TENSORRT=0` /
  `SIGLIP2_USE_TENSORRT=0` in `.env` to fall back to PyTorch if these aren't
  installed or an engine build fails on unusual hardware.

## Setup

```bash
uv sync                       # install dependencies
cp .env.example .env          # configure endpoints (defaults work locally)
make qdrant-up qdrant-health  # vector DB
make agent-up agent-health    # agent LLM (auto-picks vLLM vs llama.cpp)
```

## Running the pipeline

`EXP` selects the experiment (run) name; run `make help` for all targets.

```bash
# Full offline pipeline (vector index only)
make pipeline EXP=demo INPUT=data/raw_videos

# Or step by step
make ingest         EXP=demo INPUT=data/raw_videos
make detect-shots   EXP=demo
make extract-frames EXP=demo
make embed-frames   EXP=demo      # BEiT-3 large; ~2GB checkpoint auto-downloads on first run
make build-index    EXP=demo      # needs Qdrant running

# OCR/ASR text branch (separate; needs vllm-index + Elasticsearch — GB10 only)
make vllm-index-up elasticsearch-up
make extract-text EXP=demo
make export-text  EXP=demo        # dump ES -> manifests/text.jsonl (shareable)
make import-text  EXP=demo        # load text.jsonl back into ES

# Search / UI
make search   EXP=demo QUERY="a person riding a motorbike"
make serve-ui EXP=demo            # http://127.0.0.1:7860 (tracks + agent chat)
```

Each stage records progress in `runs/<EXP>/jobs.sqlite`; re-running a stage
skips completed work. Pass `--force` (raw CLI) to redo a stage.

### Re-indexing after an embedding-model change

Changing `--embedding-models` changes the embedding space, so re-run:

```bash
uv run codenova embed-frames --experiment-name <EXP> --embedding-models beit3-large --force
uv run codenova build-index  --experiment-name <EXP> --embedding-models beit3-large
```

`build-index` recreates the Qdrant collection from scratch (delete +
recreate), so no stale vectors survive.

## Configuration

Two layers — don't blur them:

- **`.env`** — infrastructure: service endpoints, ports, hosted-model IDs.
  Per-machine, git-ignored, loaded automatically (see `.env.example` for the
  full annotated list).
- **CLI flags** — per-experiment settings, recorded in
  `runs/<exp>/config.json`. Changing them changes the experiment identity.

Key pipeline options (CLI flags, defaults shown):

| Flag | Default | Meaning |
|------|---------|---------|
| `--embedding-models` | `beit3-large` | Comma-separated embedders (`beit3`, `siglip2`, `vietnamese-embedding`); >1 model enables SRRF fusion + BLIP-2 rerank |
| `--frame-sampling` | `shot-percentile` | Keyframe sampling strategy |
| `--keyframe-percentiles` | `0.15,0.5,0.85` | Where in each shot to sample keyframes |
| `--index-backend` | `qdrant` | Vector index backend |
| `--top-k` | `20` | Number of results |
| `--device` | `auto` | Torch device (`auto` requires CUDA) |

`serve-ui` extras: `--reranker-model` / `--reranker-top-k` enable the BLIP-2
cross-encoder rerank stage in the UI.

## Storage backends

- **`stores/vector`** — Qdrant. L2-normalized embeddings, cosine distance,
  named vector per embedding model. One collection per experiment:
  `{QDRANT_COLLECTION}__{experiment}`. Metadata is hydrated from manifests at
  query time.
- **`stores/text`** — Elasticsearch (BM25) for OCR/ASR documents, one index,
  `source` field distinguishes ocr/asr. Populated by `make extract-text`.

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
  manifests/{videos,shots,frames,embeddings}.jsonl  (+captions.jsonl, text.jsonl)
  frames/<video_id>/*.jpg
  embeddings/{frames,frame_ids}__<model>.{npz,json} (one pair per configured model)
```

The vector index lives in Qdrant (`qdrant_storage/`), text in Elasticsearch
(named Docker volume). `runs/`, `data/`, `external/`, `qdrant_storage/`, and
`.env` are git-ignored.
