import json
from pathlib import Path

run_dir = Path("runs/20260625_retrieval_siglip2-large_shot-percentile_qdrant_3f9594ab")
video_id = "a633720856da35cb"

frames_path = run_dir / "manifests" / "frames.jsonl"
frames = []
with open(frames_path, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        frame = json.loads(line)
        if frame.get("video_id") == video_id:
            frames.append(frame)

print(f"Found {len(frames)} frames for {video_id}")
print(f"First frame_path: {frames[0]['frame_path']}")
resolved = Path(frames[0]["frame_path"]).resolve()
print(f"Resolved path: {resolved}")
print(f"Exists: {resolved.is_file()}")

frames_root = (run_dir / "frames").resolve()
print(f"frames_root: {frames_root}")
print(f"is_relative_to frames_root: {resolved.is_relative_to(frames_root)}")

shots_path = run_dir / "manifests" / "shots.jsonl"
shots = []
with open(shots_path, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        shot = json.loads(line)
        if shot.get("video_id") == video_id:
            shots.append(shot)
print(f"Found {len(shots)} shots for {video_id}")

frame_by_shot = {}
for f in frames:
    sid = f.get("shot_id")
    if sid:
        frame_by_shot.setdefault(sid, []).append(f)

for shot in shots[:3]:
    sid = shot["shot_id"]
    shot_frames = frame_by_shot.get(sid, [])
    first = shot_frames[0] if shot_frames else None
    if first:
        p = Path(first["frame_path"]).resolve()
        print(f"Shot {sid}: {len(shot_frames)} frames, first exists: {p.is_file()}")
    else:
        print(f"Shot {sid}: NO FRAMES FOUND!")
