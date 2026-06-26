# TRAKE Pipeline — Multi-Event Temporal Retrieval

## Overview

TRAKE (**TRA**cking **KE**y events) is a video retrieval track that finds videos containing **multiple sequential events** described in natural language. Unlike single-query retrieval (Textual KIS) or question-answering (VQA), TRAKE receives **N independent event descriptions** (E₁, E₂, ..., Eₙ), searches each independently, then intersects results to find videos containing **all events**, scoring them by rank quality and temporal order.

## Pipeline Architecture

```
Events [E₁, E₂, ..., Eₙ]
    │
    ├─► Event 1 ─► SigLIP search (top-100) ─► Qdrant ─► [frames with scores & ranks]
    ├─► Event 2 ─► SigLIP search (top-100) ─► Qdrant ─► [frames with scores & ranks]
    └─► Event N ─► SigLIP search (top-100) ─► Qdrant ─► [frames with scores & ranks]
    │
    ▼
Intersect: videos appearing in ALL events
    │
    ▼
For each common video:
  ┌─ Pick best (lowest-rank) frame per event
  ├─ Score = Σ e^(-0.02 × rankᵢ)
  └─ Check temporal order: E₁_time < E₂_time < ... < Eₙ_time
    │
    ▼
Apply temporal bonus: if valid → score × 2.0
    │
    ▼
Sort by score descending → return ALL videos
    │
    ▼
UI: results with clickable frames → shot-level preview modal
```

## Key Components

### 1. Search (`src/retrieval/vqa.py:224-249`)

Each event text is embedded via **SigLIP 2** (local model, no API key needed) and searched against **Qdrant** vector index for the **top-100 frames**.

```python
def _search_event(experiment, event_text, event_index, top_k=100):
    retriever = build_retriever(experiment)
    results = retriever.search(query=event_text, top_k=top_k)
    # Returns list of {event_index, rank, score, frame_id, video_id,
    #                  frame_path, timestamp_sec, shot_id, frame_index}
```

- **top_k is hard-coded at 100** in the backend (`src/ui/server.py:72`). The user cannot override this from the UI.
- This ensures comprehensive coverage for each event.

### 2. Fallback: Query Processing (`src/retrieval/query_processor.py`)

Before embedding, raw event text optionally goes through a Gemini LLM processor for translation/expansion. If the Gemini API key is missing, initialization fails, or the API call fails, the system falls back to **pass-through** — the original text goes directly to SigLIP unchanged.

```
GEMINI_API_KEY missing   → PassThroughQueryProcessor → raw → SigLIP
Gemini init fails        → LlmQueryProcessor fallback → raw → SigLIP
Gemini API call fails    → LlmQueryProcessor fallback → raw → SigLIP
```

This means **no API key is required** for TRAKE to work. Everything runs locally.

### 3. Intersection (`src/retrieval/vqa.py:323-328`)

After all events are searched, we find the set of **video IDs present in every event's top-100 results**:

```python
video_sets = [set(r["video_id"] for r in er) for er in event_results]
common_videos = video_sets[0]
for vs in video_sets[1:]:
    common_videos &= vs
```

Only videos appearing in **all** events proceed to scoring. If a video only appears in 2 out of 3 events, it is excluded.

### 4. Scoring (`src/retrieval/vqa.py:252-293`)

For each common video:

```python
def _best_event_frames(event_results, vid):
```

**Step A — Best frame selection:** Pick the frame with the **lowest rank** (best search result) for each event.

**Step B — Base score:**
```
score = e^(-0.02 × rank₁) + e^(-0.02 × rank₂) + ... + e^(-0.02 × rankₙ)
```

| Rank | e^(-0.02 × rank) |
|------|-------------------|
| 1    | 0.980             |
| 5    | 0.905             |
| 10   | 0.819             |
| 20   | 0.670             |
| 50   | 0.368             |
| 100  | 0.135             |

**Step C — Temporal check:** Are the timestamps strictly increasing?
```
E₁_time < E₂_time < ... < Eₙ_time
```

**Step D — Temporal bonus:** If temporal order is valid, apply a **2.0× multiplier**:
```
if temporal_ok:
    final_score = base_score × 2.0
else:
    final_score = base_score
```

**Step E — Sort all videos** by final score descending (temporal-valid videos dominate the top).

### 5. API Endpoint (`src/ui/server.py:63-85`)

```
POST /api/trake-search
Body: { "events": ["Event 1 text", "Event 2 text", ...] }
Response: {
  "videos": [
    {
      "video_id": "...",
      "video_name": "...",
      "score": 3.456,
      "temporal_order_valid": true,
      "events": [
        {
          "event_index": 0,
          "rank": 2,
          "frame_id": "...",
          "frame_path": "...",
          "image_url": "/frame?path=...",
          "timestamp_sec": 10.5,
          "shot_id": "...",
          "frame_index": 42
        },
        ...
      ]
    },
    ...
  ],
  "total_candidates": 15
}
```

### 6. Shot Data Endpoint (`src/ui/server.py:221-261`)

```
GET /api/video-shots?video_id=<id>
Response: {
  "video_id": "...",
  "shots": [
    {
      "shot_id": "...",
      "start_frame": 0,
      "end_frame": 86,
      "start_time_sec": 0.0,
      "end_time_sec": 3.44,
      "frames": [
        { "frame_id": "...", "frame_index": 13, "timestamp_sec": 0.52,
          "frame_path": "...", "image_url": "/frame?path=..." },
        { "frame_id": "...", "frame_index": 43, "timestamp_sec": 1.72,
          "frame_path": "...", "image_url": "/frame?path=..." }
      ]
    },
    ...
  ]
}
```

Data sources:
- **Shots manifest:** `runs/<experiment>/manifests/shots.jsonl` — shot boundaries per video
- **Frames manifest:** `runs/<experiment>/manifests/frames.jsonl` — all extracted frames

Each shot returns **up to 3 frames** evenly distributed by frame_index (start, middle, end).

## UI Features

### Main Results View

Each video is displayed as a card with:
- Video name and score
- Temporal order badge (✓ temporal / ✗ temporal)
- **N event thumbnails** side-by-side (up to 5 columns)
- Each thumbnail shows: event number, rank, timestamp
- Revert button appears on events with custom thumbnails

### Shot-Level Preview Modal

Click any event frame to open the modal:

| Element | Description |
|---------|-------------|
| **Header** | Shot info (e.g., "Shot 3/15") + timestamp (⏱ HH:MM:SS) + close button |
| **Main** | Full-size image + Prev/Next buttons (or ←/→ keys) |
| **Thumbnails** | 3 frames of current shot — click to switch frame within shot |
| **Footer** | Frame info + **"Làm thumbnail"** button + **"Revert"** button |
| **Keys** | ←/→ = change shot, ↑/↓ = change frame, Esc = close |

### Custom Thumbnail Workflow

1. Click event frame → modal opens at that event's shot
2. Navigate shots/frames until finding the best representative frame
3. Click **"Làm thumbnail"** → event thumbnail in results is updated
4. Badge **"CUSTOM"** appears on the updated event
5. **"Revert"** button appears — click to restore the original frame
6. Changes persist until the next search

### State Management

- `originalThumbnails[videoId][eventIdx]` — frame the user originally clicked (saved when modal opens)
- `customThumbnails[videoId][eventIdx]` — frame the user selected via "Làm thumbnail"
- Both are cleared on new search (`RESULTS.innerHTML = ""` implicitly)

## File Map

| File | Role |
|------|------|
| `src/retrieval/vqa.py` | Core pipeline: `_search_event()`, `_best_event_frames()`, `trake_search()` |
| `src/retrieval/search.py` | `Retriever` class: embed → search → hydrate |
| `src/retrieval/query_processor.py` | Query processing with Gemini fallback |
| `src/retrieval/temporal_search.py` | Shot detection, `load_temporal_data()`, `gather_frame_s()` |
| `src/retrieval/tracks.py` | Track enum: `SUPPORTED_TRACKS["trake"]` |
| `src/retrieval/hydrator.py` | `ResultHydrator` — enrich search results with metadata |
| `src/ui/server.py` | HTTP server: `/api/trake-search`, `/api/video-shots`, full UI |
| `config/settings.py` | Experiment configuration model |

## Dependencies

- **SigLIP 2** (Hugging Face Transformers) — local text/image embedding
- **Qdrant** — vector database (standalone or Docker)
- **No external API key required** — Gemini is optional and falls back gracefully

## Running

```bash
uv run codenova serve-ui --experiment-name "<name>" --device cpu
```

Then open `http://127.0.0.1:7860`, select **TRAKE** track, enter 2+ event descriptions, and click Search.
