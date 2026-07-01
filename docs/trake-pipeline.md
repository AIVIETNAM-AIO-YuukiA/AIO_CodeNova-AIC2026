# TRAKE Pipeline — Sequential Windowed Search

## Overview

TRAKE (**TRA**cking **KE**y events) finds videos containing **N sequential events** described in natural language. Ví dụ: events = ["a person riding a motorbike", "a person falling off", "a person lying on ground"].

## Chi Tiết Từng Bước

Giả sử user nhập 3 events:

```
events = ["a person riding a motorbike", "a person falling off", "person lies on ground"]
```

---

### Bước 1: UI gửi request

File: `src/ui/server.py` (JS frontend, ~dòng 508-511)

```javascript
// JS trong trình duyệt gửi:
fetch("/api/trake-search", {
  method: "POST",
  body: JSON.stringify({ events: ["a person riding a motorbike", "a person falling off", "person lies on ground"], top_k: 50 })
})
```

> `top_k: 50` từ UI bị **ignore**. Server hardcode `top_k = 200` (dòng 75).

---

### Bước 2: Server nhận request

File: `src/ui/server.py`, dòng 67-89

```python
events_raw = payload.get("events")   # ["a person riding a motorbike", "a person falling off", "person lies on ground"]
events = [str(e).strip() for e in events_raw if str(e).strip()]
top_k = 200
result = trake_search(experiment=experiment, events=events, top_k=top_k)
```

Gọi hàm `trake_search()` trong `src/retrieval/vqa.py`.

---

### Bước 3: Bên trong trake_search()

#### 3a. Build Retriever (1 lần duy nhất)

File: `src/retrieval/vqa.py`, dòng 270

```python
retriever = build_retriever(experiment)
```

`build_retriever` làm 3 việc (`src/retrieval/search.py:68-76`):
1. Tạo **SigLIP embedder** — để biến text thành vector
2. Kết nối **Qdrant index** — database chứa tất cả frame vectors
3. Tạo **ResultHydrator** — để gắn metadata (frame_path, video_id, timestamp) vào kết quả

**Chỉ build 1 lần, dùng chung cho tất cả events.** (Cũ hơn build N lần — 1 lần mỗi event).

---

#### 3b. Search E1 — global, top 200

File: `src/retrieval/vqa.py`, dòng 271-276

```python
event_results = []
for i, ev in enumerate(events):
    event_results.append(_search_event(retriever, ev_text, i, top_k=200))
```

Hàm `_search_event` (dòng 224-248) làm:

1. **Embed event text**: `"a person riding a motorbike"` → vector (768 chiều)
2. **Search Qdrant**: tìm 200 vectors gần nhất trong toàn bộ database → trả về 200 `SearchResult`
3. **Hydrate**: gắn frame_path, video_id, timestamp_sec, shot_id cho mỗi kết quả
4. **Convert thành dict**:

```python
# event_results[0] — kết quả cho E1, 200 frames
[
  {"event_index": 0, "rank": 1,  "score": 0.92, "frame_id": "f_001", "video_id": "vid_abc", "timestamp_sec": 10.5,  "frame_path": "data/frames/f_001.jpg", ...},
  {"event_index": 0, "rank": 2,  "score": 0.88, "frame_id": "f_002", "video_id": "vid_abc", "timestamp_sec": 15.2,  "frame_path": "data/frames/f_002.jpg", ...},
  {"event_index": 0, "rank": 3,  "score": 0.85, "frame_id": "f_003", "video_id": "vid_xyz", "timestamp_sec": 200.1, "frame_path": "data/frames/f_003.jpg", ...},
  ...
  {"event_index": 0, "rank": 200, "score": 0.12, "frame_id": "f_200", "video_id": "vid_abc", "timestamp_sec": 500.0, ...},
]
```

Kết quả: **200 frames tốt nhất cho "a person riding a motorbike" trong TOÀN BỘ DATABASE** (mọi video).

---

#### 3c. Search E2 — global, top 200

Tương tự: embed `"a person falling off"` → search Qdrant → 200 frames → `event_results[1]`

```python
# event_results[1] — kết quả cho E2
[
  {"event_index": 1, "rank": 1,  "score": 0.91, "frame_id": "f_201", "video_id": "vid_abc", "timestamp_sec": 55.0, ...},
  {"event_index": 1, "rank": 2,  "score": 0.87, "frame_id": "f_202", "video_id": "vid_xyz", "timestamp_sec": 210.0, ...},
  ...
]
```

---

#### 3d. Search E3 — global, top 200

Tương tự: embed `"person lies on ground"` → search Qdrant → 200 frames → `event_results[2]`

```python
# event_results[2] — kết quả cho E3
[
  {"event_index": 2, "rank": 1,  "score": 0.90, "frame_id": "f_301", "video_id": "vid_abc", "timestamp_sec": 130.0, ...},
  ...
]
```

---

#### 3e. Gom kết quả theo video_id

File: `src/retrieval/vqa.py`, dòng 278-287

```python
results_by_video = []
for er in event_results:
    by_vid = {}
    for r in er:
        vid = r["video_id"]
        if vid not in by_vid:
            by_vid[vid] = []
        by_vid[vid].append(r)
    results_by_video.append(by_vid)
```

Kết quả sau khi gom:

```
results_by_video[0] (E1, "a person riding a motorbike"):
{
  "vid_abc": [  ← 5 frames của vid_abc lọt top 200 E1
    {score:0.92, ts:10.5}, {score:0.88, ts:15.2}, {score:0.76, ts:18.0}, {score:0.65, ts:22.1}, {score:0.40, ts:35.0}
  ],
  "vid_xyz": [  ← 12 frames của vid_xyz lọt top 200 E1
    {score:0.85, ts:200.1}, {score:0.72, ts:205.3}, ...
  ],
  "vid_def": [  ← 3 frames của vid_def lọt top 200 E1
    {score:0.60, ts:300.0}, ...
  ]
}

results_by_video[1] (E2, "a person falling off"):
{
  "vid_abc": [  ← 3 frames của vid_abc lọt top 200 E2
    {score:0.91, ts:55.0}, {score:0.70, ts:65.0}, {score:0.45, ts:80.0}
  ],
  "vid_xyz": [  ← 8 frames của vid_xyz lọt top 200 E2
    {score:0.87, ts:210.0}, ...
  ]
  // vid_def KHÔNG có frame nào trong top 200 E2
}

results_by_video[2] (E3, "person lies on ground"):
{
  "vid_abc": [  ← 2 frames của vid_abc lọt top 200 E3
    {score:0.90, ts:130.0}, {score:0.62, ts:150.0}
  ],
  "vid_xyz": [  ← 4 frames của vid_xyz lọt top 200 E3
    {score:0.78, ts:400.0}, ...
  ]
  // vid_def KHÔNG có frame nào trong top 200 E3
}
```

---

#### 3f. Lấy danh sách video từ E1

File: `src/retrieval/vqa.py`, dòng 289-296

```python
e1_by_video = {}
for r in event_results[0]:
    vid = r["video_id"]
    if vid not in e1_by_video:
        e1_by_video[vid] = []
    e1_by_video[vid].append(r)
```

Kết quả:

```python
e1_by_video = {
  "vid_abc": [5 frames E1 sorted by score],
  "vid_xyz": [12 frames E1 sorted by score],
  "vid_def": [3 frames E1 sorted by score],
}
```

Chỉ các video **có ít nhất 1 frame trong top 200 E1** mới được xử lý. Nếu 1 video không có frame nào khớp E1, nó bị loại ngay từ đầu.

---

#### 3g. Xây dựng chain cho từng video

File: `src/retrieval/vqa.py`, dòng 300-337

##### Xử lý "vid_abc":

**Bước i: Sắp xếp E1 frames theo score giảm dần**

```python
e1_list = [
  {score:0.92, ts:10.5, rank:1},   ← candidate 1
  {score:0.88, ts:15.2, rank:2},   ← candidate 2
  {score:0.76, ts:18.0, rank:5},   ← candidate 3
  {score:0.65, ts:22.1, rank:10},  ← candidate 4
  {score:0.40, ts:35.0, rank:50},  ← candidate 5
]
```

**Bước ii: Duyệt tối đa 20 E1 candidates (e1_list[:20])**

**Candidate 1**: frame E1 (score=0.92, ts=10.5)

```
Chain = [E1_frame_a]
t_curr = 10.5s
```

→ **Tìm E2**: lọc `results_by_video[1]["vid_abc"]` — frame nào có timestamp trong [10.5, 10.5+300] = [10.5s, 310.5s]?

```
Candidates E2 cho vid_abc:
  {score:0.91, ts:55.0}   ← 55.0 trong [10.5, 310.5] ✓
  {score:0.70, ts:65.0}   ← 65.0 trong [10.5, 310.5] ✓
  {score:0.45, ts:80.0}   ← 80.0 trong [10.5, 310.5] ✓
```

Cả 3 đều hợp lệ. Chọn **best score**: frame score=0.91, ts=55.0

```
Chain = [E1_frame_a, E2_frame_d]
t_curr = 55.0s
```

→ **Tìm E3**: lọc `results_by_video[2]["vid_abc"]` — frame nào có timestamp trong [55.0, 355.0]?

```
Candidates E3 cho vid_abc:
  {score:0.90, ts:130.0}   ← 130.0 trong [55.0, 355.0] ✓
  {score:0.62, ts:150.0}   ← 150.0 trong [55.0, 355.0] ✓
```

Chọn **best score**: frame score=0.90, ts=130.0

```
Chain = [E1_frame_a, E2_frame_d, E3_frame_f]
t_curr = 130.0s
```

→ **Hết events (N=3)**. Chain hợp lệ!

**Tính Final Score**:

```python
mean_sim = (0.92 + 0.91 + 0.90) / 3 = 0.91
duration = 130.0 - 10.5 = 119.5s
temporal_factor = exp(-0.5 * 119.5 / 300) = exp(-0.199) = 0.819
final_score = 0.91 * 0.819 = 0.745
```

→ `best_score = 0.745`, `best_chain = [E1_frame_a, E2_frame_d, E3_frame_f]`

**Candidate 2**: frame E1 (score=0.88, ts=15.2)

```
Chain = [E1_frame_b]
t_curr = 15.2s
```

→ Tìm E2 trong [15.2, 315.2]: cả 3 frame đều hợp lệ. Chọn best: frame score=0.91, ts=55.0

```
Chain = [E1_frame_b, E2_frame_d]
t_curr = 55.0s
```

→ Tìm E3 trong [55.0, 355.0]: frame score=0.90, ts=130.0

```
Chain = [E1_frame_b, E2_frame_d, E3_frame_f]
```

**Tính Final Score**:

```python
mean_sim = (0.88 + 0.91 + 0.90) / 3 = 0.897
duration = 130.0 - 15.2 = 114.8s
temporal_factor = exp(-0.5 * 114.8 / 300) = exp(-0.191) = 0.826
final_score = 0.897 * 0.826 = 0.741
```

→ `0.741 < 0.745` → giữ nguyên best.

**Candidate 3**: frame E1 (score=0.76, ts=18.0)

→ Tìm E2 trong [18.0, 318.0]: frame score=0.91, ts=55.0
→ Tìm E3 trong [55.0, 355.0]: frame score=0.90, ts=130.0

```python
mean_sim = (0.76 + 0.91 + 0.90) / 3 = 0.857
duration = 130.0 - 18.0 = 112.0s
temporal_factor = exp(-0.5 * 112.0 / 300) = exp(-0.187) = 0.829
final_score = 0.857 * 0.829 = 0.710
```

→ `0.710 < 0.745` → giữ nguyên best.

**Kết quả cho vid_abc**: score=0.745, chain=[E1 ts=10.5, E2 ts=55.0, E3 ts=130.0]

---

##### Xử lý "vid_xyz":

E1 candidates cho vid_xyz:

```python
e1_list = [
  {score:0.85, ts:200.1, rank:3},
  {score:0.72, ts:205.3, rank:7},
  ...
]
```

**Candidate 1**: frame E1 (score=0.85, ts=200.1)

```
Chain = [E1_frame_c]
t_curr = 200.1s
```

→ **Tìm E2**: lọc `results_by_video[1]["vid_xyz"]` — timestamp trong [200.1, 500.1]?

```
Candidates E2 cho vid_xyz:
  {score:0.87, ts:210.0}   ← 210.0 trong [200.1, 500.1] ✓
  ...
```

Chọn best: frame score=0.87, ts=210.0

```
Chain = [E1_frame_c, E2_frame_e]
t_curr = 210.0s
```

→ **Tìm E3**: lọc `results_by_video[2]["vid_xyz"]` — timestamp trong [210.0, 510.0]?

```
Candidates E3 cho vid_xyz:
  {score:0.78, ts:400.0}   ← 400.0 trong [210.0, 510.0] ✓
```

Chọn: frame score=0.78, ts=400.0

```
Chain = [E1_frame_c, E2_frame_e, E3_frame_g]
```

Chain hợp lệ!

**Tính Final Score**:

```python
mean_sim = (0.85 + 0.87 + 0.78) / 3 = 0.833
duration = 400.0 - 200.1 = 199.9s
temporal_factor = exp(-0.5 * 199.9 / 300) = exp(-0.333) = 0.717
final_score = 0.833 * 0.717 = 0.597
```

→ `best_score = 0.597` cho vid_xyz

---

##### Xử lý "vid_def":

E1 candidates cho vid_def:

```python
e1_list = [
  {score:0.60, ts:300.0, ...},
  ...
]
```

**Candidate 1**: frame E1 (score=0.60, ts=300.0)

```
t_curr = 300.0s
```

→ **Tìm E2**: lọc `results_by_video[1]["vid_def"]`?

`results_by_video[1]` (E2) **không có** key "vid_def" → `results_by_video[1].get("vid_def", [])` = `[]`

→ **0 candidates cho E2** → discard chain.

Không có chain nào cho vid_def.

---

#### 3h. Gom tất cả chain hợp lệ

File: `src/retrieval/vqa.py`, dòng 339-340

```python
valid_chains = [
  (0.745, [E1 ts=10.5, E2 ts=55.0, E3 ts=130.0], "vid_abc"),
  (0.597, [E1 ts=200.1, E2 ts=210.0, E3 ts=400.0], "vid_xyz"),
]
valid_chains.sort(key=lambda x: -x[0])  # sort score giảm dần
```

---

#### 3i. Format output

File: `src/retrieval/vqa.py`, dòng 342-371

```python
out_videos = [
  {
    "video_id": "vid_abc",
    "video_name": "video_abc.mp4",
    "score": 0.745,
    "temporal_order_valid": True,
    "events": [
      {"event_index": 0, "score": 0.92, "timestamp_sec": 10.5, "frame_path": "data/frames/f_001.jpg", ...},
      {"event_index": 1, "score": 0.91, "timestamp_sec": 55.0, "frame_path": "data/frames/f_201.jpg", ...},
      {"event_index": 2, "score": 0.90, "timestamp_sec": 130.0, "frame_path": "data/frames/f_301.jpg", ...},
    ]
  },
  {
    "video_id": "vid_xyz",
    ...
  }
]

return {
  "videos": out_videos,
  "total_candidates": 2,  # 2 video có chain hợp lệ
}
```

---

### Bước 4: Server hydrate image URLs

File: `src/ui/server.py`, dòng 81-84

```python
for video in result.get("videos", []):
    for ev in video.get("events", []):
        if ev.get("frame_path"):
            ev["image_url"] = f"/frame?path={quote(ev['frame_path'])}"
```

Thêm đường dẫn ảnh để trình duyệt hiển thị.

---

### Bước 5: Gửi JSON response về UI

```json
{
  "videos": [
    {
      "video_id": "vid_abc",
      "video_name": "video_abc.mp4",
      "score": 0.745,
      "temporal_order_valid": true,
      "events": [
        {"event_index": 0, "score": 0.92, "timestamp_sec": 10.5, "image_url": "/frame?path=data%2Fframes%2Ff_001.jpg", ...},
        {"event_index": 1, "score": 0.91, "timestamp_sec": 55.0, "image_url": "/frame?path=data%2Fframes%2Ff_201.jpg", ...},
        {"event_index": 2, "score": 0.90, "timestamp_sec": 130.0, "image_url": "/frame?path=data%2Fframes%2Ff_301.jpg", ...}
      ]
    },
    {
      "video_id": "vid_xyz",
      "score": 0.597,
      "events": [...]
    }
  ],
  "total_candidates": 2
}
```

---

### Bước 6: UI render

File: `src/ui/server.py`, dòng 523-526 (JS)

```javascript
renderTrake(data);
// Hiển thị:
//   - "2 video(s) match all events TRAKE"
//   - Video card 1: vid_abc (score 0.745)
//     - 3 thumbnails: E1 (10.5s), E2 (55.0s), E3 (130.0s)
//   - Video card 2: vid_xyz (score 0.597)
//     - 3 thumbnails: ...
```

---

## Tóm tắt

| Bước | Mô tả | Số frame |
|------|-------|----------|
| 1 | Search E1 global → top 200 | 200 |
| 2 | Search E2 global → top 200 | 200 |
| 3 | Search E3 global → top 200 | 200 |
| 4 | Gom kết quả theo video_id | (trong bộ nhớ) |
| 5 | Duyệt từng video trong E1, thử tối đa 20 E1 candidates | (trong bộ nhớ) |
| 6 | Với mỗi candidate: lọc E2 theo video_id + window 300s | (trong bộ nhớ) |
| 7 | Nếu có E2: lọc E3 theo video_id + window 300s | (trong bộ nhớ) |
| 8 | Tính score cho chain, giữ best chain/video | (trong bộ nhớ) |
| 9 | Sort chains theo score → trả về | (trong bộ nhớ) |

**Tổng cộng: 3 lần search Qdrant × 200 frames = 600 frames retrieved. Không search lại.**

## Tại sao vid_def bị loại?

vid_def có frame E1 (score 0.60) nhưng **không có bất kỳ frame E2 nào trong top 200 global của event "a person falling off"**. Có thể vì video đó không có cảnh té ngã, hoặc cảnh té ngã trong vid_def có vector quá khác so với query.

## Tại sao vid_xyz có score thấp hơn vid_abc?

Mặc dù E1 của vid_xyz (score 0.85) khá tốt, nhưng:
- Khoảng cách giữa E1 và E3 là 199.9s (gần hết window), bị phạt temporal nhiều hơn
- Điểm E2, E3 cũng thấp hơn

→ **vid_abc** (các event xảy ra sát nhau trong 119.5s, điểm cao hơn) được ưu tiên.

## Hyperparameters

| Parameter | Value | File |
|-----------|-------|------|
| top_k per event | 200 | `src/retrieval/vqa.py:254` |
| WINDOW_SIZE | 300s (5 phút) | `src/retrieval/vqa.py:266` |
| LAMBDA_TEMPORAL (λ) | 0.5 | `src/retrieval/vqa.py:267` |
| Max E1 candidates/video | 20 | `src/retrieval/vqa.py:305` |
