# CodeNova — Video Retrieval Pipeline (HCMC AI Challenge 2026)

Resumable video retrieval system: videos are split into shots, sampled into
keyframes, embedded with one or more configured models (Jina CLIP v2 by
default), and indexed in Qdrant. OCR/ASR text goes into Elasticsearch.

> 🇻🇳 Tài liệu tiếng Việt: [docs/vi/README.md](docs/vi/README.md)

## Pipeline

```
OFFLINE (indexing)
  preflight/approval
    → ingest → detect-shots → extract-frames → embed-frames → build-index
    → validate-index → readiness.json (READY/DEGRADED/INVALID)
  extract-text  (separate stage: OCR per keyframe + ASR per video → Elasticsearch;
                 needs `make vllm-index-up` + `make elasticsearch-up`)

ONLINE (retrieval)
  text query → [LLM translate/expand] → embed (BEiT-3) → Qdrant search
    → temporal search (frame-to-frame walk → segments) → shot validation
    → track shaping (textual KIS / VQA / TRAKE)
    → [VQA] agent answer  /  [chat] interactive agent loop
```

Generative models (LLM/VLM) are **Docker-served over OpenAI-compatible
HTTP** — no model checkpoint is loaded inside the Python process for
captioning, OCR, or query processing. Embedders/rerankers (BEiT-3, SigLIP2,
BLIP-2) run in-process as batch encoders. The agent LLM itself is not
served by this repo's `docker-compose.yml` (point `.env` at whatever
OpenAI-compatible endpoint you run for it).

## Models

| Role | Model | Where it runs |
|------|-------|---------------|
| Visual embedding (default) | BEiT-3 `beit3_large_patch16_384_coco_retrieval` (dim 1024) | in-process, TensorRT FP16 (~27x vs PyTorch, benchmarked on GB10) |
| Visual embedding (opt-in) | SigLIP2 `google/siglip2-so400m-patch14-384` (dim 1152) | in-process, TensorRT FP16 (~3.5x vs PyTorch, benchmarked on GB10) |
| Caption-text embedding (opt-in) | `AITeamVN/Vietnamese_Embedding_v2` over VLM captions | captions via Docker VLM; embedding in-process |
| Reranker (opt-in, multi-model runs) | BLIP-2 ITM `Salesforce/blip2-itm-vit-g` | in-process |
| Captioning + OCR (indexing & agent tools) | `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit` | Docker `vllm-index` (vLLM, AWQ/Marlin, **GB10 only**), port 8881 |
| Agent LLM (VQA answer, chat, query expand) | **Qwen3.5-4B 4-bit** | not served by this repo's docker-compose — point `.env` at your own OpenAI-compatible endpoint |
| ASR | gipformer-65M-rnnt + Silero-VAD | subprocess into `external/gipformer/` |
| Shot detection | TransNetV2 (PyTorch) | in-process |

`vllm-index` (35B-A3B, port 8881) is the only vLLM container this repo's
`docker-compose.yml` runs. Captioning/OCR has **no non-GB10 fallback**:
`vllm-index` needs GB10's Blackwell (SM121) GPU for its AWQ/Marlin kernel.
Override its model via `VLLM_INDEX_MODEL` in `.env`.

### TensorRT-accelerated embedding

BEiT-3 and SigLIP2 spend nearly all their time in the vision tower's forward
pass — measured on this project's GB10 dev machine, PyTorch eager mode takes
~400-860ms/image, which would put a 283K-frame corpus at 1-2 days on that
hardware (actual timing varies by GPU). `modules/embedding/tensorrt_runtime.py`
exports each model's vision tower to ONNX and builds a TensorRT FP16 engine
the first time `embed-frames` runs (a few minutes, one-time), caching it
under `weights/<model>/`. Every later run just loads the cached engine —
verified cosine similarity >=0.9999 against the PyTorch output, ~27x faster
for BEiT-3 and ~3.5x for SigLIP2. Text queries stay on PyTorch (a single
short sequence per call has no batching gain to capture). Disable per-model
with `BEIT3_USE_TENSORRT=0` / `SIGLIP2_USE_TENSORRT=0` in `.env`.

## Agent

Two agent paths, one LLM backend (endpoint set via `.env`, not docker-compose):

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
- `git` and `uv` on PATH — `detect-shots` uses them to fetch and convert the
  TransNetV2 weights on first run (see [TransNetV2 weights](#transnetv2-weights))
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
make setup                    # deps + GPU extras + external repos/weights
cp .env.example .env          # configure endpoints (defaults work locally)
make qdrant-up qdrant-health  # vector DB
make elasticsearch-up         # OCR/ASR
```

`make setup` runs `uv sync`, installs the optional GPU extras (onnxruntime,
onnx, TensorRT — skipped without failing the build if they don't resolve on
this platform), and provisions the two external repos: TransNetV2 (clone +
weight conversion) and gipformer (clone + isolated venv). Each step is
idempotent, so re-running it is cheap. `uv sync` alone still works — the
external repos are then provisioned lazily on first use.

### TransNetV2 weights

`make detect-shots` provisions these itself on first run: it clones
[soCzech/TransNetV2](https://github.com/soCzech/TransNetV2) into `external/`
and converts the bundled TensorFlow SavedModel into
`external/TransNetV2/inference-pytorch/transnetv2-pytorch-weights.pth`.

The conversion runs in a throwaway `uv` venv (`tensorflow==2.16.*` + `torch`)
that is deleted afterwards, so its pins never touch the project's own
`.venv` and its `torch==2.11.0+cu128` build. Requires `git` and `uv` on PATH.
Pass `--transnetv2-weights` to point at an existing `.pth` and skip all of it.

## Running the pipeline

`EXP` selects the experiment (run) name; run `make help` for all targets.

```bash
# Full guarded offline pipeline (approved preflight + final quality gate)
make pipeline EXP=demo INPUT=data/raw_videos

# Resume the same experiment; completed per-video work is skipped
make pipeline EXP=demo INPUT=data/raw_videos RESUME=1

# Include OCR/ASR in the same execution (services must already be running)
make pipeline EXP=demo INPUT=data/raw_videos WITH_TEXT=1

# Or step by step
make preflight-index EXP=demo INPUT=data/raw_videos APPROVE=1
make ingest         EXP=demo INPUT=data/raw_videos
make detect-shots   EXP=demo
make extract-frames EXP=demo
make embed-frames   EXP=demo      # BEiT-3 large; ~2GB checkpoint auto-downloads on first run
make build-index    EXP=demo      # needs Qdrant running
make validate-index EXP=demo      # writes readiness.json; non-zero unless READY

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
skips completed work. Pass `--force` (raw CLI) to redo a stage. `serve-ui`
refuses to start unless `readiness.json` is `READY` and its artifact hashes
still match, so run `validate-index` after any manual stage or repair.

`offline-index` is the recommended competition command. Without `--approve`,
the raw CLI only writes and prints a `PENDING_APPROVAL` plan. After review,
repeat the command with `--resume --approve`; it verifies the immutable
dataset/config fingerprint, runs the stages, always runs the final quality
gate, and returns a non-zero exit code on pipeline error or non-READY output.

To inspect a damaged JSONL manifest without changing it:

```bash
make repair-manifest EXP=demo MANIFEST=frames
make repair-manifest EXP=demo MANIFEST=frames APPLY=1  # atomic rewrite + .bak + audit JSON

# Legacy runs: inspect first, then explicitly convert paths to run-relative form
make migrate-frame-paths EXP=demo LEGACY_ROOT="$PWD"
make migrate-frame-paths EXP=demo LEGACY_ROOT="$PWD" APPLY=1
# APPLY updates frames.jsonl + frame partitions, creates backups/audit, and removes stale readiness
make validate-index EXP=demo
```

### Changing an artifact-defining option

An existing experiment is immutable with respect to `embedding_models`,
`frame_sampling`, `keyframe_percentiles`, `index_backend`, `pipeline`, and
`data_dir`. To change one of them, create a new experiment instead of rewriting
the old run:

```bash
EMBEDDING_MODELS=beit3-large make pipeline EXP=<NEW_EXP> INPUT=data/raw_videos
```

Passing an artifact option that differs from `config.json` to an existing-run
command fails with the field name plus persisted/requested values. `device`,
`top_k`, `runs_dir`, UI host/port and logging controls remain runtime options.
`--force` may regenerate artifacts only with the same persisted definition.

## Configuration

Two layers — don't blur them:

- **`.env`** — infrastructure: service endpoints, ports, hosted-model IDs.
  Per-machine, git-ignored, loaded automatically (see `.env.example` for the
  full annotated list).
- **CLI flags** — artifact settings are recorded in `runs/<exp>/config.json`
  schema v1 and restored whenever the run is reopened. Explicit mismatches are
  rejected. Runtime-only flags such as `--device` and `--top-k` may change.

Key pipeline options (CLI flags, defaults shown):

| Flag | Default | Meaning |
|------|---------|---------|
| `--embedding-models` | `jina-clip-v2` | Comma-separated embedders (`jina`, `beit3`, `siglip2`, `vietnamese-embedding`); >1 model enables SRRF fusion + BLIP-2 rerank |
| `--frame-sampling` | `shot-percentile` | Keyframe sampling strategy |
| `--keyframe-percentiles` | `0.15,0.5,0.85` | Where in each shot to sample keyframes |
| `--index-backend` | `qdrant` | Vector index backend |
| `--top-k` | `20` | Number of results |
| `--device` | `auto` | Torch device (`auto` requires CUDA) |

Embedding backend resolution is strict. Supported short aliases are `jina`,
`jina-clip-v2`, `beit3`, `beit3-large`, `siglip2`, `siglip2-so400m`,
`siglip2-large`, `vietnamese-embedding`, and `vietnamese_embedding`. A custom
repository ID is accepted only when it contains the corresponding `jina`,
`siglip`, or `vietnamese-embedding` marker; BEiT-3 uses its explicit local
aliases. Unknown names and typos fail during
preflight, before embedding artifacts are created; they never fall back to
SigLIP. The resolved backend/model ID/revision/preprocessing identity is stored
in `manifests/embeddings.jsonl` and checked again before online retrieval.

Multi-model indexing and WSF fusion join vectors by `frame_id`, never by raw
row position. Every model must have the same unique frame-ID set; a missing,
extra or duplicate ID fails `build-index`/`validate-index`. Different row order
is safe and reported in readiness as `policy: join_by_frame_id` with
`same_row_order: false`.

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
  config.json  # versioned artifact_config + runtime_defaults + config_hash
  jobs.sqlite
  readiness.json
  plans/<preflight-plan>.json
  logs/{pipeline.log, errors.log, executions/*.log, repairs/*.json}
  manifests/{videos,shots,frames,embeddings}.jsonl  (+captions.jsonl, text.jsonl)
  manifests/partitions/<stage>/*.json
  frames/<video_id>/*.jpg
  embeddings/{frames,frame_ids}__<model>.{npz,json} (one pair per configured model)
```

The vector index lives in Qdrant (`qdrant_storage/`), text in Elasticsearch
(named Docker volume). `runs/`, `data/`, `external/`, `qdrant_storage/`, and
`.env` are git-ignored.
