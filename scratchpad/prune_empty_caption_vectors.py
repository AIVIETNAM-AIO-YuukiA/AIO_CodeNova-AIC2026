"""One-off cleanup: drop vietnamese-embedding vectors built from an empty
caption (frame_id not present in captions.jsonl) so a later
`embed-frames --caption-missing` run recaptions and re-embeds them properly.

Root cause: OpenRouter key hit its monthly limit (HTTP 403) for ~23h; every
caption call during that window failed, embed_images() caught it, embedded
"" anyway, so those frames never made it into captions.jsonl but did make it
into frames__vietnamese-embedding.npz / frame_ids__vietnamese-embedding.json.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np

RUN_DIR = Path("runs/result")
EMB_DIR = RUN_DIR / "embeddings"
MODEL = "vietnamese-embedding"

vectors_path = EMB_DIR / f"frames__{MODEL}.npz"
ids_path = EMB_DIR / f"frame_ids__{MODEL}.json"
captions_path = RUN_DIR / "manifests" / "captions.jsonl"

captioned_ids = set()
with captions_path.open(encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        captioned_ids.add(row["frame_id"])

frame_ids = json.loads(ids_path.read_text(encoding="utf-8"))
vectors = np.load(vectors_path)["embeddings"]

assert len(frame_ids) == vectors.shape[0], "frame_ids/vectors length mismatch"

keep_mask = [fid in captioned_ids for fid in frame_ids]
kept_ids = [fid for fid, keep in zip(frame_ids, keep_mask) if keep]
kept_vectors = vectors[np.array(keep_mask, dtype=bool)]

dropped = len(frame_ids) - len(kept_ids)
print(f"total vectors: {len(frame_ids)}")
print(f"keeping (real caption): {len(kept_ids)}")
print(f"dropping (empty caption): {dropped}")

if dropped == 0:
    print("Nothing to prune, exiting without touching files.")
else:
    shutil.copy2(vectors_path, vectors_path.with_suffix(".npz.bak"))
    shutil.copy2(ids_path, ids_path.with_suffix(".json.bak"))

    np.savez_compressed(vectors_path, embeddings=kept_vectors.astype("float32"))
    ids_path.write_text(json.dumps(kept_ids), encoding="utf-8")

    print(f"Backups written: {vectors_path.with_suffix('.npz.bak')}, {ids_path.with_suffix('.json.bak')}")
    print(f"Pruned files written: {vectors_path}, {ids_path}")
    print(f"Now {len(kept_ids)} frames embedded; {dropped} will be picked up by --caption-missing")
