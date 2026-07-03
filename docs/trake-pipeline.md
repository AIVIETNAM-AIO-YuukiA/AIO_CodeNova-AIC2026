# TRAKE Pipeline — Bidirectional Pair Join (BPJ)

## Overview

TRAKE (**TRA**cking **KE**y events) finds videos containing **N sequential events** described in natural language. The new **Bidirectional Pair Join (BPJ)** algorithm processes each adjacent pair (eᵢ, eᵢ₊₁) independently with forward and backward searches, then chains the best pairs together.

---

## Step-by-Step Walkthrough (bằng lời, không code)

Giả sử user nhập 3 events:

```
E₁ = "a person riding a motorbike"
E₂ = "a person falling off"
E₃ = "person lies on ground"
```

### Bước 1: Search tất cả events — mỗi event một lần

Hệ thống search từng event một cách độc lập trên toàn bộ database:

- **E₁** → vector search → lấy **300 frame** tốt nhất từ mọi video
- **E₂** → vector search → lấy **300 frame** tốt nhất từ mọi video
- **E₃** → vector search → lấy **300 frame** tốt nhất từ mọi video

Mỗi frame có: `frame_id`, `video_id`, `timestamp`, `score` (vector similarity với câu query).

**Kết quả:** 3 danh sách, mỗi danh sách 300 frame.

---

### Bước 2: Xử lý từng cặp liền kề (Adjacent Pairs)

Với N=3 events, có 2 cặp liền kề:
- **Pair A**: (E₁, E₂)
- **Pair B**: (E₂, E₃)

Mỗi cặp được xử lý độc lập qua 2 chiều: **Forward** và **Backward**.

---

#### Pair A (E₁, E₂)

##### Forward pass

Lấy 300 frame E₁ làm mồi. Với mỗi frame fᵢ của E₁:
- Tìm **top 30 frame** E₂ trong [timestamp(fᵢ), timestamp(fᵢ) + 300s] (cùng video_id)
- Dùng **in-memory cosine similarity** giữa embeddings trong window và query vector của E₂
- Chọn 30 frame E₂ có similarity cao nhất

Kết quả: tối đa 300 × 30 = 9000 cặp (E₁_frame, E₂_frame).

Ví dụ forward pairs:

| E₁ frame | E₁ ts | E₁ score | E₂ frame | E₂ ts | E₂ score |
|----------|-------|----------|----------|-------|----------|
| f_001    | 10.5s | 0.92     | f_201    | 55.0s | 0.91     |
| f_002    | 15.2s | 0.88     | f_201    | 55.0s | 0.91     |
| f_003    | 18.0s | 0.76     | f_201    | 55.0s | 0.91     |
| f_004    | 200.1s | 0.85    | f_202    | 210.0s | 0.87    |

##### Backward pass

Lấy 300 frame E₂ làm mồi. Với mỗi frame fⱼ của E₂:
- Tìm **top 30 frame** E₁ trong [timestamp(fⱼ) - 300s, timestamp(fⱼ)] (cùng video_id)
- Dùng **in-memory cosine similarity** giữa embeddings trong window và query vector của E₁
- Chọn 30 frame E₁ có similarity cao nhất

Kết quả: tối đa 300 × 30 = 9000 cặp (E₁_frame, E₂_frame).

Ví dụ backward pairs:

| E₂ frame | E₂ ts | E₂ score | E₁ frame | E₁ ts | E₁ score |
|----------|-------|----------|----------|-------|----------|
| f_201    | 55.0s | 0.91     | f_001    | 10.5s | 0.92     |
| f_202    | 210.0s | 0.87    | f_004    | 200.1s | 0.85    |
| f_203    | 65.0s | 0.70     | f_001    | 10.5s | 0.92     |

##### Merge

- Gộp forward pairs + backward pairs (tối đa 18000 raw pairs)
- Dedup bằng (frame_id_E₁, frame_id_E₂) — giữ cặp có pair score cao nhất
- Pair score = sim_i + sim_j (không temporal penalty)
- Giữ tối đa **300 pairs** cho cặp (E₁, E₂)

Pair score tính bằng:

```
pair_score = sim(fᵢ, eᵢ) + sim(fᵢ₊₁, eᵢ₊₁)
```

Ví dụ:
- Cặp (f_001, f_201): score = 0.92 + 0.91 = 1.83
- Cặp (f_004, f_202): score = 0.85 + 0.87 = 1.72

---

#### Pair B (E₂, E₃)

Làm tương tự:

**Forward**: 300 frame E₂ → tìm E₃ trong [ts, ts+300s]
**Backward**: 300 frame E₃ → tìm E₂ trong [ts-300s, ts]
**Merge**: gộp, dedup, top 300 pairs

Ví dụ merged pairs cho (E₂, E₃):

| E₂ frame | E₂ ts | E₂ score | E₃ frame | E₃ ts | E₃ score | Pair score |
|----------|-------|----------|----------|-------|----------|------------|
| f_201    | 55.0s | 0.91     | f_301    | 130.0s | 0.90    | 1.81 |
| f_202    | 210.0s | 0.87    | f_302    | 400.0s | 0.78    | 1.65 |

---

### Bước 3: Join các cặp thành chain

Chain được tạo bằng cách **nối các cặp có chung frame_id**.

Pair A (E₁, E₂): (f_001, f_201) — E₁ frame f_001, E₂ frame f_201
Pair B (E₂, E₃): (f_201, f_301) — E₂ frame f_201, E₃ frame f_301

→ Join trên frame_id của E₂ = f_201 → chain hợp lệ:
```
Chain: f_001 (E₁, ts=10.5s) → f_201 (E₂, ts=55.0s) → f_301 (E₃, ts=130.0s)
```

Nếu không có cặp nào ở Pair B có E₂ frame_id = f_201, chain đó không tồn tại.

---

### Bước 4: Tính điểm chain

Công thức:

```
Chain Score = mean(sim_i, sim_j, ...)
```

Trong đó:
- `mean(sim)` = trung bình similarity scores của tất cả frame trong chain
- Không temporal penalty — điểm chỉ dựa trên độ khớp semantic

Ví dụ với chain E₁(10.5s, 0.92) → E₂(55.0s, 0.91) → E₃(130.0s, 0.90):

```
Chain Score = (0.92 + 0.91 + 0.90) / 3 = 0.91
```

---

### Bước 5: Gom kết quả và sắp xếp

Tất cả chain hợp lệ từ mọi video:

1. video_A — score 0.872 — E₁(10.5s), E₂(55.0s), E₃(130.0s)
2. video_B — score 0.597 — E₁(200.1s), E₂(210.0s), E₃(400.0s)

Sắp xếp theo Chain Score giảm dần, gửi JSON về UI.

---

## Chi Tiết Từng Bước

Giả sử user nhập 3 events:

```
events = ["a person riding a motorbike", "a person falling off", "person lies on ground"]
```

---

### Bước 1: UI gửi request

File: `src/ui/server.py` (JS frontend)

```javascript
fetch("/api/trake-search", {
  method: "POST",
  body: JSON.stringify({ events: [...], top_k: 50 })
})
```

> `top_k: 50` từ UI bị **ignore**. Server hardcode `top_k = 300`.

---

### Bước 2: Server nhận request

File: `src/ui/server.py`

```python
events_raw = payload.get("events")
events = [str(e).strip() for e in events_raw if str(e).strip()]
top_k = 300
result = trake_search(experiment=experiment, events=events, top_k=top_k)
```

Gọi hàm `trake_search()` trong `src/retrieval/vqa.py`.

---

### Bước 3: Bên trong trake_search()

#### 3a. Build Retriever (1 lần duy nhất)

File: `src/retrieval/vqa.py`

```python
retriever = build_retriever(experiment)
```

`build_retriever` làm 3 việc (`src/retrieval/search.py`):
1. Tạo **SigLIP embedder** — để biến text thành vector
2. Kết nối **Qdrant index** — database chứa tất cả frame vectors
3. Tạo **ResultHydrator** — để gắn metadata (frame_path, video_id, timestamp) vào kết quả

**Chỉ build 1 lần, dùng chung cho tất cả events.**

---

#### 3b-3d. Search từng event — global, top 300

```python
event_results = []
for i, ev in enumerate(events):
    event_results.append(_search_event(retriever, ev_text, i, top_k=300))
```

Hàm `_search_event`:
1. **Embed event text** → vector (768 chiều)
2. **Search Qdrant**: tìm 300 vectors gần nhất → trả về 300 `SearchResult`
3. **Hydrate**: gắn frame_path, video_id, timestamp_sec, shot_id + vector embedding
4. **Convert thành dict**:

```python
# event_results[0] — 300 frames cho E₁
[
  {"event_index": 0, "rank": 1,  "score": 0.92, "frame_id": "f_001", "video_id": "vid_abc", "timestamp_sec": 10.5, "vector": [...], ...},
  {"event_index": 0, "rank": 2,  "score": 0.88, "frame_id": "f_002", "video_id": "vid_abc", "timestamp_sec": 15.2, "vector": [...], ...},
  ...
]
```

Kết quả: **300 frames tốt nhất cho mỗi event trong TOÀN BỘ DATABASE**.

---

#### 3e. Build lookup structures

```python
# frame_map[event_index][frame_id] = frame_data
# video_frames[event_index][video_id] = [frames]
```

Xây dựng để tra cứu nhanh:
- frame theo frame_id (lấy vector, score, timestamp)
- frame theo video_id (lọc nhanh)

---

#### 3f. Xử lý từng cặp liền kề

Với N=3 events, có N-1=2 cặp: (E₁, E₂) và (E₂, E₃).

##### Forward pass cho (E₁, E₂)

```python
pairs_forward = []
for f_i in event_results[0]:  # 300 E₁ frames
    # Lọc E₂ frames cùng video, trong [ts, ts+300s]
    candidates = [f for f in event_results[1]
                  if f["video_id"] == f_i["video_id"]
                  and f_i["timestamp_sec"] <= f["timestamp_sec"] <= f_i["timestamp_sec"] + 300]
    if not candidates:
        continue
    # Tính in-memory cosine similarity giữa mỗi candidate E₂ vector và query vector của E₂
    top_matches = sorted(candidates, key=lambda f: cosine_sim(f["vector"], query_vector_E₂), reverse=True)[:30]
    for best in top_matches:
        pair_score = f_i["score"] + best["score"]
        pairs_forward.append((f_i["frame_id"], best["frame_id"], pair_score))
```

Lưu ý: `cosine_sim` được tính giữa vector của candidate frame và query vector (không dùng Qdrant score cho E₂ match vì Qdrant score là similarity với query E₁ — cần similarity với query E₂ để đánh giá mức độ khớp của E₂).

##### Backward pass cho (E₁, E₂)

```python
pairs_backward = []
for f_j in event_results[1]:  # 300 E₂ frames
    # Lọc E₁ frames cùng video, trong [ts-300s, ts]
    candidates = [f for f in event_results[0]
                  if f["video_id"] == f_j["video_id"]
                  and f_j["timestamp_sec"] - 300 <= f["timestamp_sec"] <= f_j["timestamp_sec"]]
    if not candidates:
        continue
    # Tính in-memory cosine similarity với query vector của E₁
    top_matches = sorted(candidates, key=lambda f: cosine_sim(f["vector"], query_vector_E₁), reverse=True)[:30]
    for best in top_matches:
        pair_score = best["score"] + f_j["score"]
        pairs_backward.append((best["frame_id"], f_j["frame_id"], pair_score))
```

##### Merge cho (E₁, E₂)

```python
all_pairs = {}
for (f1, f2, s) in pairs_forward + pairs_backward:
    key = (f1, f2)
    if key not in all_pairs or s > all_pairs[key][0]:
        all_pairs[key] = (s, f1, f2)
# Sort theo pair_score giảm dần, lấy top 300
merged = sorted(all_pairs.values(), key=lambda x: -x[0])[:300]
```

Kết quả: 300 cặp (E₁, E₂) tốt nhất.

Làm tương tự cho cặp (E₂, E₃).

---

#### 3g. Join pairs thành chains

```python
# pair_results[0] = [(score, f1_id, f2_id), ...] cho (E₁, E₂)
# pair_results[1] = [(score, f2_id, f3_id), ...] cho (E₂, E₃)

chains = []
for (s_ab, f1, f2) in pair_results[0]:
    # Tìm tất cả cặp (E₂, E₃) có E₂ frame_id = f2
    matches = [(s_bc, f2b, f3) for (s_bc, f2b, f3) in pair_results[1] if f2b == f2]
    for (s_bc, _, f3) in matches:
        chain_score = compute_chain_score([f1_frame_data, f2_frame_data, f3_frame_data])
        chains.append((chain_score, [f1, f2, f3], video_id))
```

**Compute chain score**:

```python
def compute_chain_score(frames):
    mean_sim = sum(f["score"] for f in frames) / len(frames)
    return mean_sim
```

---

#### 3h. Format output

```python
out_videos = [
  {
    "video_id": "vid_abc",
    "video_name": "video_abc.mp4",
    "score": 0.872,
    "events": [
      {"event_index": 0, "score": 0.92, "timestamp_sec": 10.5, "frame_path": "data/frames/f_001.jpg", ...},
      {"event_index": 1, "score": 0.91, "timestamp_sec": 55.0, "frame_path": "data/frames/f_201.jpg", ...},
      {"event_index": 2, "score": 0.90, "timestamp_sec": 130.0, "frame_path": "data/frames/f_301.jpg", ...},
    ]
  },
  ...
]

return {
  "videos": out_videos,
  "total_candidates": len(out_videos),
}
```

---

### Bước 4: Server hydrate image URLs

File: `src/ui/server.py`

```python
for video in result.get("videos", []):
    for ev in video.get("events", []):
        if ev.get("frame_path"):
            ev["image_url"] = f"/frame?path={quote(ev['frame_path'])}"
```

---

### Bước 5: Gửi JSON response về UI

```json
{
  "videos": [
    {
      "video_id": "vid_abc",
      "score": 0.872,
      "events": [
        {"event_index": 0, "score": 0.92, "timestamp_sec": 10.5, "image_url": "/frame?path=..."},
        {"event_index": 1, "score": 0.91, "timestamp_sec": 55.0, "image_url": "/frame?path=..."},
        {"event_index": 2, "score": 0.90, "timestamp_sec": 130.0, "image_url": "/frame?path=..."}
      ]
    }
  ],
  "total_candidates": 1
}
```

---

### Bước 6: UI render

```javascript
renderTrake(data);
// Hiển thị:
//   - "N video(s) match all events TRAKE"
//   - Mỗi video card: thumbnails E₁, E₂, ..., Eₙ kèm timestamp
```

---

## Tóm tắt

| Bước | Mô tả | Số frame/entry |
|------|-------|---------------|
| 1 | Search E₁ global → top 300 | 300 |
| 2 | Search E₂ global → top 300 | 300 |
| 3 | Search Eₙ global → top 300 | 300 |
| 4 | Forward pass (E₁→E₂) — top 30 mỗi candidate | ≤9000 pairs |
| 5 | Backward pass (E₂←E₁) — top 30 mỗi candidate | ≤9000 pairs |
| 6 | Merge & top 300 (E₁,E₂) | 300 pairs |
| 7 | Forward pass (E₂→E₃) — top 30 mỗi candidate | ≤9000 pairs |
| 8 | Backward pass (E₃←E₂) — top 30 mỗi candidate | ≤9000 pairs |
| 9 | Merge & top 300 (E₂,E₃) | 300 pairs |
| 10 | Join pairs → chains | (trong bộ nhớ) |
| 11 | Score chains, sort → trả về | (trong bộ nhớ) |

**Tổng cộng: N lần search Qdrant × 300 frames. Pair join hoàn toàn in-memory, không search lại.**

---

## Hyperparameters

| Parameter | Value | File |
|-----------|-------|------|
| top_k per event | 300 | `src/retrieval/trake_search.py` |
| WINDOW_SIZE | 300s (5 phút) | `src/retrieval/trake_search.py` |
| Window search top-K | 30 | `src/retrieval/trake_search.py` |
| Max pairs per adjacent pair | 300 | `src/retrieval/trake_search.py` |
