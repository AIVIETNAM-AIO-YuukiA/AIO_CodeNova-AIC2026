# KIS Detail 2-Stage — Weighted Normalized Sum Fusion

Date: 2026-07-29 11:17:10
Branch: cuong-test-kis-later-early-fusion

## Mục tiêu

Cải thiện thuật toán KIS Detail 2-Stage từ sum fusion đơn thuần (cộng dồn cosine scores)
sang weighted normalized sum fusion, giúp mỗi subquery có "tiếng nói" công bằng hơn.

## Thuật toán mới

### Weighted Normalized Sum Fusion

```
Mỗi subquery i:
  - maxScore_i = max(cosine(frame, query_i)) over all frames
  - normalized_score(frame, i) = cosine(frame, query_i) / maxScore_i
  - final_score(frame) = Σ normalized_score(frame, i) × w_i
```

Trong đó:
- w_i là weight của subquery i, Σ w_i = 1
- Mặc định w_i = 1/N (uniform)
- Sau này frontend có thể gửi custom weights qua field `general_weights` / `specific_weights`

### Pipeline

```
User Input (general list + specific list)
  ↓
Stage 1 — General Weighted Normalized Sum Fusion
  ↓ (top 1000 candidates)
Stage 2 — Specific Weighted Normalized Sum Fusion
  ↓ (top 300 results)
Hydrate frame info → Render cards
```

## Files thay đổi

### `src/retrieval/kis_detail_search.py`
- Thêm hàm `_weighted_sum_fusion(scores, weights)` — helper:
  - Input: [N_frames, K_subqueries] raw cosine matrix
  - Tìm maxScore mỗi cột → normalize → nhân weight → sum
  - Output: [N_frames] final scores
- Update `kis_detail_2stage_search()`:
  - Thêm params `general_weights` / `specific_weights` (optional, default None)
  - Stage 1: `final1 = _weighted_sum_fusion(scores1, weights=gen_weights_arr)`
  - Stage 2: `final2 = _weighted_sum_fusion(scores2, weights=spec_weights_arr)`

### `src/ui/server.py`
- Endpoint `/api/kis-detail-2stage`:
  - Truyền `payload.get("general_weights")` và `payload.get("specific_weights")` xuống hàm

### `prompt/preprocess-kis-detail.md`
- Mới: merged prompt từ 2 file cũ
- Output: 2 section `## GENERAL` và `## SPECIFIC`, tiếng Anh
- Action-to-static conversion: verbs → static visual descriptions
- Critical filter: loại bỏ yếu tố không search được (trừu tượng, cảm xúc, mục đích, biểu tượng, văn hóa)

## UI

- Dropdown: "KIS Detail 2-Stage"
- General subqueries: events-list riêng với nút "+ Add general"
- Specific subqueries: events-list riêng với nút "+ Add specific"
- Submit → `/api/kis-detail-2stage` → renderDetailCards()

## Future work

- Frontend gửi custom weights: user có thể kéo slider hoặc nhập số cho từng subquery
- Hiển thị weight UI bên cạnh mỗi subquery input
