# CodeNova — HCMC AI Challenge 2026 Video Retrieval Pipeline

## What this project is

A resumable, experiment-tracked video retrieval system for the HCMC AI Challenge
(image-level pipeline: video → shots → keyframes → embeddings, not raw video encoding —
cheaper to compute/store and avoids the long-video/short-moment signal dilution
problem). It targets four competition tracks:

| Track | Meaning | Status |
|---|---|---|
| **Textual KIS** | Find the unique moment matching a text description | ✅ implemented |
| **Video KIS** | Find the unique moment matching a sample video clip | ⚠️ label/stub only — no video encoder |
| **VQA** | Find a moment + answer a question about it | ✅ implemented (search → temporal → validate → agent) |
| **TRAKE** | Find an ordered sequence of events for a query | ✅ implemented (multi-segment temporal search) |

Competition constraints that shape the design: 5 minutes per KIS query, scored on
speed + rank; VQA answers must be short text; TRAKE needs temporally-ordered event
lists, not just top-K frames.

## Ground truth over docs

Two files in the repo root — `hcmc_ai_challenge_pipeline_analysis.md` and
`docs/pipeline_detail.txt` — are **research notes and partially aspirational design
docs**, not accurate descriptions of current code. Known stale claims in them:

- They reference `src/pipeline/vqa.py`, `src/pipeline/indexing.py`,
  `src/pipeline/clip_model.py` — **`src/pipeline/` does not exist.** The real code is
  `src/retrieval/vqa.py`, `src/retrieval/search.py`.
- They describe **CLIP (ViT-B/32)** as the embedder. **There is no CLIP implementation
  anywhere in `src/`.** The project moved to SigLIP2 + BEiT-3.
- `README.md`'s config table lists `--clip-model` — the actual flag is
  `--embedding-models` (comma-separated, default `siglip2-large`).
- "48 tests" is stale; current count is ~77 test functions across 17 files.

Treat `README.md` and this file as more current than the two docs above, but when in
doubt, **read the code** — `src/cli/main.py` for flags, `src/config/settings.py` for
defaults. `hcmc_ai_challenge_pipeline_analysis.md` is still useful as a *literature
survey* (SOTA model comparisons, competition paper analysis) — just don't trust it as
an implementation spec.

For detailed algorithms and formulas extracted from the reference competition papers
— temporal search variants, SRRF fusion, λ-decay, composite validation scoring,
landmark image-to-image reasoning, adaptive keyframe/subframe extraction — see:

- [`references/paper1_llandmark.md`](references/paper1_llandmark.md) — LLandMark
  (UIT+HCMUT, 77.40/88): landmark image-to-image search, min-score temporal
  intersection, weighted fusion agent, YOLOv9-e, Milvus, MongoDB metadata.
- [`references/paper2_cascaded_system.md`](references/paper2_cascaded_system.md) —
  Cascaded System (UIT+IU+HCMUT, 76.4/88): full formulas for beam search + λ-decay
  temporal scoring (additive exploration vs. multiplicative validation), SRRF,
  adaptive score fusion with agent-predicted weights, GPT-4o query expansion (N=4
  variants), experimental results/hyperparameters.
- [`references/paper3_vortex.md`](references/paper3_vortex.md) — Vortex (VNU-HCM
  Science, FocusOnFun, 79.6/88 — highest prelim score of the three): adaptive
  L2-norm keyframe filtering, plain RRF (no score-preservation), no cross-encoder
  reranker, Rocchio relevance feedback (like/dislike), additive temporal search
  with no time-gap penalty, human-in-the-loop philosophy (LLM only suggests query
  rewrites, never auto-applies them) — a useful counterpoint to paper1/paper2's
  full-automation approach.
- [`references/paper.md`](references/paper.md) — narrower, implementation-scoped
  research (not competition-architecture analysis like the three above):
  structured-captioning-for-embedding best practices (why the caption prompt in
  `prompts/captioning.py` is rigid, not free-form), AIC 2025 teams' OCR/rerank
  techniques and confirmation that AIC 2026 has no results yet as of 7/2026, and
  ASR chunking/overlap-deduplication practice behind `modules/asr/gipformer.py`.

Each entry in all three files is tagged Implemented / Not implemented against
current code; check that tag before assuming an upgrade idea already exists.

## Pipeline (as implemented)

```
OFFLINE (indexing) — src/cli/main.py subcommands, wrapped by `make pipeline`
  ingest → detect-shots → extract-frames → embed-frames → build-index
                                                        (extract-text runs separately)

  ingest          discover videos under INPUT, init run dir (runs/<exp>/)
  detect-shots    TransNetV2 (PyTorch) shot boundary detection
  extract-frames  keyframe sampling at percentiles [0.15, 0.5, 0.85] per shot
  embed-frames    embed keyframes with SigLIP2 / BEiT-3 / Vietnamese caption
                    embedding, per --embedding-models (any subset, any count)
  build-index     write vectors + frame_id into Qdrant (named vectors per model)
  extract-text    OCR (per-keyframe) + ASR (per-video) → Elasticsearch;
                    separate command, not part of `make pipeline` — needs
                    vLLM + Elasticsearch running, not just Qdrant

ONLINE (retrieval) — src/retrieval/
  text query → embed (same model(s) as index) → Qdrant search
    → [if >1 embedding model] SRRF fusion (retrieval/fusion.py)
    → [if >1 embedding model] BLIP-2 ITM rerank (modules/reranker/blip2_itm.py)
    → temporal_search (forward/backward frame-to-frame walk → segments)
    → ShotValidator (0.7*avg_clip_score + 0.3*temporal_consistency)
    → track-specific shaping (retrieval/tracks.py: textual_kis / vqa / trake)
    → [VQA/TRAKE only] agent.react ReAct loop → short text answer
```

**`VietnameseReranker` (`modules/reranker/vietnamese_reranker.py`) is not yet
wired into `Retriever.search()`** — it exists and is tested standalone, but
combining it with `Blip2ItmReranker`'s output was deliberately deferred (the
two rerankers need to run in parallel as independent result lists, not fused
into one score — see the two-rerankers decision below). Don't assume it's
active in the online search path yet; check `retrieval/search.py` before
relying on it.

Each stage records progress in `runs/<exp>/jobs.sqlite`; re-running a completed stage
is a no-op unless `--force` is passed. This resumability is load-bearing for a
multi-day indexing run over a large video corpus — don't "fix" it by making stages
always rerun.

## Layout

```
src/
  cli/main.py         # argparse CLI: ingest, detect-shots, extract-frames,
                       #   embed-frames, build-index, search, serve-ui,
                       #   name-experiment, validate-experiment-name
  config/settings.py   # PipelineConfig (dataclass), Experiment, experiment naming/hashing
  core/                # logging, typed records (FrameRecord, SearchResult), errors
  video/               # OpenCV discovery, TransNetV2 shot detection, frame extraction
  indexing/            # offline stage implementations + manifests + SQLite job state
  retrieval/
    search.py          # embed query → Qdrant search
    fusion.py           # SRRF (score-reflected reciprocal rank fusion) across models
    hydrator.py         # attach manifest metadata to raw vector-search hits
    query_processor.py  # PassThroughQueryProcessor / LlmQueryProcessor (Gemini translate+expand)
    temporal_search.py  # forward/backward frame-to-frame walk, find_segments,
                         #   gather_frame_s, ShotInput, ShotValidator
    tracks.py            # textual_kis / video_kis (stub) / vqa / trake track adapters
    vqa.py                # vqa_search(), trake_search() — full pipelines
  modules/
    _vllm_chat.py        # internal shared client: vLLM OpenAI-compatible /chat/completions
                         #   with an inlined base64 image — used by captioning/vllm.py
                         #   and ocr/vllm.py (same server, different prompts)
    embedding/
      siglip.py            # real, HF Transformers (SigLIP 2 checkpoints)
      beit3.py              # real, vendored (see _beit3_vendor/ note below)
      vietnamese.py         # real: VLM-captions each frame (cached to
                             #   manifests/captions.jsonl), then embeds the
                             #   caption text with a Vietnamese sentence
                             #   embedder — a drop-in Embedder even though it
                             #   embeds text, not pixels; see its docstring
      build_embedder() dispatches on "beit3"/"vietnamese-embedding" substrings,
        else SigLIP — any subset of the three can be configured together
    reranker/
      blip2_itm.py          # real BLIP-2 ITM cross-encoder (query text <-> image)
      vietnamese_reranker.py # real BGE-reranker-v2-m3-based cross-encoder
                              #   (query text <-> caption text) — needs
                              #   result.caption already hydrated; NOT yet
                              #   wired into Retriever.search() (see Pipeline
                              #   section above)
    captioning/vllm.py     # real: self-hosted vLLM VLM, structured Vietnamese
                            #   caption prompt (prompts/captioning.py)
    ocr/vllm.py             # real: same vLLM server, dedicated OCR-only prompt
                            #   (prompts/ocr.py) — deliberately separate from
                            #   captioning, feeds Elasticsearch not the
                            #   Vietnamese embedding branch
    asr/gipformer.py       # real: g-group-ai-lab/gipformer-65M-rnnt via
                            #   sherpa-onnx, called as a subprocess into its
                            #   own isolated repo+venv at external/gipformer/
                            #   (NOT a HuggingFace Transformers checkpoint).
                            #   Chunks long audio (30s windows, 1s overlap,
                            #   LCS-deduplicated) — see its docstring.
    detection/              # base.py interface ONLY — no subclass. NOT the
                            #   same code path as agent/tools.py (see below).
  stores/
    vector/             # Qdrant (only working backend; others raise NotImplementedError)
    text/                # Elasticsearch BM25 — wired in via `codenova extract-text`
                         #   (OCR + ASR documents); factory in stores/text/factory.py
  repository/           # data access over run manifests
                         #   (frame_repo, video_repo, caption_repo — captions.jsonl)
  agent/                # ReAct loop for VQA/TRAKE answer generation
    react.py             # Agent: thought → action → tool → observation, max 5 steps
    brain.py             # Gemini backend (gemini-2.5-flash-lite, text-only reasoning,
                          #   temperature=0.0)
    local.py             # Ollama backend (moondream/llava) — auto-detected if server running
    internvl.py          # InternVL3-2B-hf backend (temperature=0.0 / do_sample=False)
    tools.py             # caption/ocr (real, VLM-backed), detect (YOLOv8n via ultralytics,
                          #   optional dep), asr (Whisper, optional dep) — degrade to
                          #   placeholder text if optional deps aren't installed.
                          #   Separate implementation from modules/{captioning,ocr,asr}/ —
                          #   this one is for interactive agent tool-calling
                          #   (VQA/TRAKE answer generation), the modules/ one is
                          #   for offline batch indexing. Don't conflate the two.
  prompts/              # LLM/VLM prompt templates
    captioning.py         # structured Vietnamese news-keyframe caption prompt
                           #   (rigid 5-section format — see its docstring for why)
    ocr.py                 # dedicated on-screen-text-only extraction prompt
    qa.py, vqa.py, query_expansion.py   # answer-generation / query-processing prompts
  ui/server.py          # stdlib http.server app; serves textual_kis/video_kis/vqa/trake
tests/unit/             # pytest, ~77 tests across 17 files
external/
  TransNetV2/           # weights present at inference-pytorch/transnetv2-pytorch-weights.pth
  BEiT3/checkpoints/     # weights present: beit3_base_itc_patch16_224.pth, beit3.spm
  gipformer/             # cloned from github.com/ggroup-ai-lab/gipformer, own venv
                         #   (`cd external/gipformer && uv sync`) — isolated from the
                         #   main project's dependency stack (own sherpa-onnx pin).
                         #   infer_json.py is a JSONL-output wrapper added on top of
                         #   the upstream repo (infer_onnx.py's stdout format isn't
                         #   safe to parse programmatically) — don't remove it thinking
                         #   it's redundant with infer_onnx.py.
```

**`detection/` is the one remaining dead-interface stub** — `modules/detection/base.py`
has no subclass anywhere (unlike asr/captioning/ocr, which are now real). The
agent's own caption/OCR/detect/ASR tools in `src/agent/tools.py` remain a
*separate* implementation from `modules/{captioning,ocr,asr}/` — same
capability, different purpose (interactive agent vs. offline batch indexing)
and different code paths. Don't conflate them.

## Infra (docker-compose.yml)

| Service | Port | Purpose | Status |
|---|---|---|---|
| `qdrant` | 6333 (REST+UI), 6334 (gRPC) | vector index, backed by `qdrant_storage/` (bind mount) | active, required |
| `vllm-retrieval` | 8881 | Qwen3.5-9B-AWQ-4bit — online agent/retrieval tasks (VQA/TRAKE reasoning) | active, `profiles: [retrieval]` — `make vllm-retrieval` |
| `vllm-index` | 8881 | Qwen3.6-35B-A3B-AWQ-4bit — offline captioning/OCR indexing (`modules/captioning/vllm.py`, `modules/ocr/vllm.py`) | active, `profiles: [index]` — `make vllm-index` |
| `elasticsearch` | 8882→9200 | BM25 OCR/ASR full-text, data in the `elasticsearch_data` **named volume** | active — populated by `codenova extract-text` |
| `elasticvue` | 8883→8080 | Web UI for Elasticsearch (ES itself has no built-in UI, only a REST API) | active |

**`vllm-retrieval` and `vllm-index` share port 8881 and are mutually exclusive**
(same GPU, can't run both) — they're gated behind Compose `profiles` specifically so
a bare `docker compose up -d` starts neither; always start the one you need via
`make vllm-retrieval` or `make vllm-index` (`make vllm-down` stops whichever is up).
Because they share a port, no `.env` change is needed when switching — `VLLM_MODEL`
only matters for display/logging, the running container determines which model
actually answers requests. Both services' GPU memory fraction/max-model-len are
tuned for a specific machine (Tân's DGX Spark GB10) — don't change without checking
the target hardware.

**GB10-specific tuning — counterintuitive vs. discrete-GPU advice, don't "fix" it:**
`vllm-index` runs `--performance-mode throughput` with `--max-num-seqs 16` (matches
`CAPTION_WORKERS`); `vllm-retrieval` runs `--performance-mode interactivity` with
`--max-num-seqs 4`. This split is deliberate — GB10's unified LPDDR5x memory is
bandwidth-bound (~273GB/s), not compute-bound, so batch captioning (many concurrent
requests) genuinely benefits from higher concurrency while single-user agent chat
does not (per-token bandwidth tax outweighs continuous-batching gains above ~4
streams for interactive workloads). Neither service sets `--kv-cache-dtype fp8` —
unlike on H100-class discrete GPUs where FP8 KV-cache is usually a throughput win,
vLLM's own DGX Spark guidance says it can hurt predictability/perf on this hardware.
Don't add it "for speed" without re-benchmarking on this specific machine.

`elasticsearch` uses a **named Docker volume**, not a bind mount like `qdrant`. This
is deliberate: a bind mount (`./elasticsearch_storage:/usr/share/elasticsearch/data`)
breaks on a fresh machine whenever the host directory gets created with the wrong
owner (commonly root, if Docker creates it on first `up`) — the container runs as
UID 1000, can't write, and crash-loops on "failed to obtain node locks". A named
volume sidesteps this entirely since Docker manages its ownership internally. Don't
revert `elasticsearch` to a bind mount without also fixing the ownership problem some
other way.

## Config model — two layers, don't blur them

- **`.env`** — infrastructure only: Qdrant/Elasticsearch URLs, UI host/port, API keys
  (`GEMINI_API_KEY`, `OPENAI_API_KEY`), plus model identifiers for the
  self-hosted/HF-ID-configurable backends (`VLLM_BASE_URL`, `VLLM_MODEL`,
  `VIETNAMESE_EMBEDDING_MODEL`, `VIETNAMESE_RERANKER_MODEL`) — these count as
  infra, not per-experiment tuning, because swapping which VLM/embedder is
  *hosted* is a deployment concern, unlike which embedders an experiment
  *uses* (that's `--embedding-models`, a CLI flag, see below). gipformer (ASR)
  has no env var — it's not HF-ID swappable, it's a fixed subprocess call into
  `external/gipformer/`. Loaded automatically, per-machine, git-ignored.
- **CLI flags** — per-experiment settings (embedding models, keyframe percentiles,
  top-k, device). Recorded verbatim in `runs/<exp>/config.json` for reproducibility.
  **Never move one of these into `.env`** — the whole point is that changing them
  produces a new experiment name (see `PipelineConfig.config_hash()` /
  `default_experiment_name()` in `src/config/settings.py`), and `.env` values wouldn't
  be captured in that hash.

Key defaults (`src/config/settings.py::PipelineConfig`):
`embedding_models=("siglip2-large",)`, `frame_sampling="shot-percentile"`,
`index_backend="qdrant"`, `keyframe_percentiles=(0.15, 0.5, 0.85)`, `top_k=20`,
`device="auto"`.

## Query processing (Vietnamese-first)

`retrieval/query_processor.py` has two modes, auto-selected by whether
`GEMINI_API_KEY` is set:

- **Pass-through** (no key): query sent to the embedder as-is.
- **LLM mode** (key present): Gemini translates Vietnamese → English, enriches the
  prompt (layout/color/lighting detail) for better SigLIP/BEiT-3 matching, and extracts
  OCR/ASR keywords. Falls back to pass-through silently on any API/network error —
  never raises to the caller. Preserve this graceful-fallback behavior; a demo/
  competition run cannot hard-fail because of a flaky API call. Note this is a
  separate Vietnamese-handling path from `modules/embedding/vietnamese.py` /
  `modules/ocr/vllm.py` — this one processes the *query* at search time, those
  process *keyframes* at index time.

SigLIP2/BEiT-3 tokenizers truncate at their context limit automatically
(`truncation=True`) — long queries are cut, not crashed on.

## Development workflow

```bash
uv sync                                        # install deps (Python 3.13, uv-managed)
cp .env.example .env
make qdrant-up && make qdrant-health           # start + verify Qdrant

make pipeline EXP=demo INPUT=data/raw_videos   # full offline run (vector index only)
make search   EXP=demo QUERY="..."
make serve-ui EXP=demo                          # http://127.0.0.1:7860

# OCR/ASR branch — separate, needs vllm-index (not vllm-retrieval — see Infra
# section) + Elasticsearch running, not just Qdrant:
make vllm-index && make elasticsearch-up && make elasticsearch-health
make extract-text EXP=demo

make lint / make format / make check / make test / make precommit
```

`external/gipformer/` needs its own one-time setup (`cd external/gipformer && uv sync`)
before ASR extraction works — it's an isolated repo+venv, not part of the main
project's `uv sync`.

- Lint/format: `ruff` (line-length 100), config in `pyproject.toml`. `external/` and
  the vendored `src/modules/embedding/_beit3_vendor/` are excluded from ruff.
- Tests: `pytest`, `pythonpath = ["src"]`, run via `uv run pytest -q` or `make test`.
- Pre-commit: trailing-whitespace, end-of-file-fixer, check-yaml,
  check-added-large-files, ruff (`--fix --unsafe-fixes`), ruff-format.
- GPU is expected for SigLIP/BEiT-3/TransNetV2 when `--device auto` — verify with
  `torch.cuda.is_available()` before debugging "model too slow" reports.

## Conventions worth preserving

- **Resumable by design.** Every offline stage checks `jobs.sqlite` before doing work.
  Don't add code paths that silently skip this check — always gate re-execution behind
  the same job-state mechanism, or add an explicit `--force` like existing stages.
- **Frame-to-frame, not query-to-frame, for temporal search.** The forward/backward
  walk in `temporal_search.py` compares adjacent frame embeddings to each other to find
  segment boundaries — it does not compare frames against the query. This is
  intentional (finds shot-consistent context around a hit); don't "simplify" it into a
  query-similarity threshold.
- **Vendored BEiT-3.** `_beit3_vendor/` exists because of a `torchscale`/`timm` version
  conflict with the rest of the stack — it's a deliberate workaround, not leftover
  code. Don't try to replace it with a pip dependency without checking that conflict is
  resolved upstream.
- **Multi-model fusion is opt-in.** SRRF fusion and BLIP-2 reranking only activate when
  `--embedding-models` has more than one model. Single-model runs skip both — keep this
  short-circuit; it's a real latency/cost tradeoff for the 5-minute KIS clock, not an
  oversight.
- **Optional-dependency degradation.** `agent/tools.py`'s detect (YOLOv8) and asr
  (Whisper) tools degrade to placeholder output when the optional package isn't
  installed, rather than raising. Match this pattern for any new tool.
- No CLIP anywhere — if you see a reference to CLIP in a doc, prompt, or comment,
  treat it as stale unless the code says otherwise.
- **Deterministic generation everywhere for batch/agent tasks.** `temperature=0.0`
  (or `do_sample=False`) is used across the board for OCR (`modules/ocr/vllm.py`),
  captioning (`modules/captioning/vllm.py`), and agent reasoning (`agent/brain.py`,
  `agent/internvl.py`) — a deliberate choice, not the default of the underlying
  libraries. These are batch-indexing/tool-calling tasks feeding downstream
  embeddings or structured parsing, not creative writing; consistency matters
  more than fluency. Don't raise temperature back up "for better answers"
  without checking why it was set to 0 first.
- **Prompts live in `prompts/`, not inlined in the calling module.** `captioning.py`
  and `ocr.py` are separate prompt files even though both call the same vLLM
  server, because they serve different downstream consumers (Vietnamese
  embedding vs. Elasticsearch) and need independently tunable prompts. If you
  add a new VLM-backed capability, put its prompt in `prompts/`, not as a
  string literal inside the module that calls the model — `agent/tools.py`'s
  `CaptionTool`/`OCRTool` predate this convention and hardcode their prompts
  inline; that's a known inconsistency, not a pattern to copy.
- **Two rerankers, not fused.** `Blip2ItmReranker` (image-text) and
  `VietnameseReranker` (caption-text) are designed to run as independent,
  parallel signals — the user explicitly rejected combining them into one
  score. If you wire `VietnameseReranker` into `Retriever.search()`, keep the
  two result lists separate (e.g. return a dict keyed by reranker name)
  rather than merging scores.

## Where to look for competition-strategy ideas

`hcmc_ai_challenge_pipeline_analysis.md` is a literature review (LLandMark and
Cascaded-System competition papers) with a proposed "ideal pipeline" section — useful
for brainstorming next-model upgrades (ViSigLIP-OT for Vietnamese, LLM2CLIP,
adaptive/subframe keyframing, λ-decay temporal scoring, hybrid sparse+dense in Qdrant
to retire Elasticsearch) but **nothing in it should be assumed already implemented**.
Cross-check against `src/` before proposing it's "already there."
