# Qdrant usage

The vector index uses **Qdrant**, which runs as a separate service. Qdrant must be running
and `.env` must point at it before `build-index` or `search`.

> 🇻🇳 Bản tiếng Việt: [vi/qdrant.md](vi/qdrant.md)

## 1. Start Qdrant

The simplest way is Docker Compose (`docker-compose.yml` is included):

```bash
make qdrant-up                   # docker compose up -d qdrant
docker compose ps                # check the container
docker compose logs -f qdrant    # view logs
make qdrant-down                 # stop (data kept in ./qdrant_storage)
```

Data is persisted to `./qdrant_storage/` (git-ignored), so restarts keep the index.

Health check:

```bash
make qdrant-health               # curl http://localhost:6333/healthz
```

Dashboard: http://localhost:6333/dashboard

## 2. Configure `.env`

```bash
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=codenova_frames
QDRANT_API_KEY=                  # empty for local; set for Qdrant Cloud
```

Each experiment uses a collection named `{QDRANT_COLLECTION}__{experiment}`, so multiple
runs sharing one Qdrant never collide.

## 3. What Qdrant stores

Each point holds the CLIP embedding plus a payload with `frame_id`. Keyframe images stay on
disk (`runs/<exp>/frames/`); full metadata lives in the run manifests and is joined at query
time via `frame_id`. `build-index` recreates the collection then upserts all embeddings, so
rebuilds are idempotent.

## 4. Quick inspection

```bash
# List collections
curl http://localhost:6333/collections

# Inspect one collection (replace with your experiment)
curl http://localhost:6333/collections/codenova_frames__demo

# Delete a collection to rebuild from scratch
curl -X DELETE http://localhost:6333/collections/codenova_frames__demo
```

## 5. Qdrant Cloud

Create a cluster, then set in `.env`:

```bash
QDRANT_URL=https://<your-cluster>.qdrant.io:6333
QDRANT_API_KEY=<your-api-key>
```

No code change needed — `stores.vector.factory` reads these from the environment.

## 6. Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| `Connection refused` | Qdrant not running → `make qdrant-up` |
| `Install qdrant-client ...` | Missing dependency → `uv sync` |
| Empty search results | Index not built, or wrong `--experiment-name` |
| `Cannot build a Qdrant index with zero embeddings` | Run `embed-frames` first / no frames found |
