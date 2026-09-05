# Luồng model theo từng track (CodeNova Retrieval UI)

Cấu hình hiện tại: `EMBEDDING_MODELS=jina-clip-v2,siglip2-so400m,vietnamese-embedding` (3 model).
Tất cả 5 track **dùng chung một `Retriever` instance** (cache `_get_retriever(experiment)` trong
`src/retrieval/vqa.py`) — đây là điểm mấu chốt vừa sửa: trước đó `server.py` và `vqa.py` mỗi bên tự
build retriever riêng, khiến 6 embedder (3+3) cùng tồn tại trên GPU 4 GB và làm tràn VRAM.

---

## 1. KIS Basic (`textual_kis`) — endpoint `/api/search`

```
query + context
   │
   ▼
build_retrieval_text()          [retrieval/tracks.py] — ghép chuỗi text thuần
   │
   ▼
retriever.search(text, top_k)   [retrieval/search.py]
   │
   ├─ 1 model  → embed_text() 1 lần → Qdrant.search() → hydrate → xong
   │
   └─ >1 model (hiện tại: 3) → với MỖI model:
         embed_text(text) → Qdrant.search(model_name=...) → results_by_model[model]
      rồi:
         srrf_fuse(results_by_model)      [retrieval/fusion.py — port từ AIC_2025]
         reranker.rerank(...)             [BLIP-2 ITM, nếu >1 model VÀ chưa DISABLE_RERANKER]
   │
   ▼
kết quả top_k, đã hydrate metadata
```

**Model chạm vào:** Jina-CLIP-v2 + SigLIP2-so400m + Vietnamese-Embedding (embed text) → SRRF → BLIP-2 (rerank, nếu bật).
**Nhẹ nhất** trong 5 track — không đụng temporal/agent.

---

## 2. TRAKE (`trake`) — endpoint `/api/trake-search`

```
events: [{text, sub_details}, ...]  (UI gửi mảng, ≥2 event)
   │
   ▼
_events_to_query()               [ui/server.py] — convert thành "E1: ...\nE2: ..."
   │
   ▼
trake_search()                   [retrieval/vqa.py]
   │
   ├─ parse multi-event từ chuỗi E1:/E2:/...
   │
   └─ VỚI MỖI event:
         _run_temporal_pipeline(event_text)
            │
            ├─ retriever = _get_retriever(experiment)   ← retriever DÙNG CHUNG
            ├─ retriever.search(event_text, top_k)       [giống luồng KIS Basic ở trên]
            ├─ (optional) reranker.rerank(...)
            ├─ load_temporal_data(experiment)             [numpy, CPU — không đụng GPU]
            ├─ find_segments() + gather_frame_s()         [temporal expansion, CPU]
            └─ ShotValidator().validate(shot, query_embedding, ...)
                  │
                  └─ query_embedding lấy từ embedder CACHED
                     (KHÔNG build_embedder() mới — đây là chỗ vừa sửa)
   │
   ▼
DP sequence search tìm chuỗi video khớp thứ tự các event
   │
   ▼
events[] (shot theo từng event) + results[] (top frame)
```

**Model chạm vào:** giống KIS Basic, nhưng lặp lại **cho mỗi event** trong chuỗi. Trước khi sửa: mỗi
event từng gọi `build_embedder()` riêng cho `ShotValidator` → nạp model mới mỗi lần → leak VRAM.

---

## 3. KIS Multi-Scene (`kis_multi_scene`) — endpoint `/api/kis-multi-scene`

```
events: [{text, sub_details}, ...]  (UI gửi mảng, ≥2 scene)
```

**Dùng chung 100% code với TRAKE** — cùng route xử lý (`trake_search()`), chỉ khác:
- Trước khi sửa: route `/api/kis-multi-scene` **không tồn tại ở backend** → bấm Search sẽ 404.
- Đã vá: `do_POST` giờ khớp cả `"/api/trake-search"` và `"/api/kis-multi-scene"` vào cùng 1 handler.

Về mặt khái niệm, "Multi-Scene" và "TRAKE" là cùng một bài toán (chuỗi sự kiện/scene tuần tự) — UI
tách 2 lựa chọn cho người dùng chọn nhãn phù hợp ngữ cảnh cuộc thi, nhưng luồng xử lý là một.

---

## 4. KIS Detail 2-Stage (`kis_detail_2stage`) — endpoint `/api/kis-detail-2stage`

```
general: [subquery1, subquery2, ...]     (thô, Stage 1)
specific: [subquery1, subquery2, ...]    (tinh, Stage 2)
   │
   ▼
kis_detail_2stage_search()        [retrieval/kis_detail_search.py]
   │
   ├─ load_temporal_data(experiment)      [numpy, toàn bộ frame embeddings 1 model]
   │
   ├─ Stage 1 (general):
   │     embedder = _get_retriever(experiment).embedders[model_đầu_tiên]   ← cached, KHÔNG build mới
   │     gen_embs = embed_text(mỗi subquery general)
   │     scores = frame_embeddings @ gen_embs.T   (numpy, CPU)
   │     weighted_sum_fusion() → giữ top_k_stage1 (mặc định 1000)
   │
   └─ Stage 2 (specific):
         embedder = _get_retriever(experiment).embedders[model_đầu_tiên]   ← CÙNG embedder, cached
         spec_embs = embed_text(mỗi subquery specific)
         scores = frame_embeddings[stage1] @ spec_embs.T
         weighted_sum_fusion() → top_k_stage2 (mặc định 300)
   │
   ▼
results (300 frame, mỗi frame có contribution từ cả general + specific)
```

**Model chạm vào:** chỉ **1 model** (model đầu tiên trong `embedding_models`, tức Jina-CLIP-v2) — track
này KHÔNG dùng SRRF/3-model fusion, nó tự làm "weighted normalized sum fusion" riêng trên 1 không
gian embedding. Trước khi sửa: mỗi request tự `build_embedder()` **2 lần** (1 lần/stage) → model mới
mỗi lần gọi.

---

## 5. VQA (`vqa`) — endpoint `/api/vqa-search`

```
query + context + question
   │
   ▼
vqa_search()                      [retrieval/vqa.py] — 3-stage pipeline
   │
   ├─ Stage 1: _run_temporal_pipeline(query)
   │     (giống hệt nhánh trong TRAKE — retriever cached, SRRF nếu 3 model, rerank optional)
   │
   ├─ Stage 2: Temporal search + Shot gather + Shot validation
   │     (numpy, CPU — giống TRAKE)
   │
   └─ Stage 3: Agent (Qwen)
         create_agent(backend=vqa_backend)     [agent/__init__.py]
            │
            ├─ backend="local"  → VllmChatClient → http://localhost:8884/v1
            │                     (KHÔNG có container nào chạy ở port này trong session này)
            │
            └─ .env: VLM_BACKEND=openrouter     → tự động fallback sang OpenRouter
                                                   model: qwen/qwen3.7-flash
         agent.run(question, context=shots đã gather) → answer text
   │
   ▼
{answer, results[], pipeline: {embed_search, temporal_search, gather_shot, shot_validation, agent}}
```

**Model chạm vào:** 3 embedder (fusion) → BLIP-2 (rerank, optional) → **Qwen qua OpenRouter API**
(không phải GPU cục bộ). Đây là track duy nhất gọi ra LLM sinh câu trả lời tự nhiên; 4 track kia chỉ
trả về danh sách frame xếp hạng.

---

## Bảng tổng hợp

| Track | Endpoint | Model dùng | Số lần embed | Có Agent? | Đã test |
|---|---|---|---|---|---|
| KIS Basic | `/api/search` | 3 (SRRF+rerank) | 1 lần/model | Không | ✅ 5s |
| TRAKE | `/api/trake-search` | 3 (SRRF+rerank) | 1 lần/model **×N event** | Không | ✅ 24s |
| KIS Multi-Scene | `/api/kis-multi-scene` | 3 (SRRF+rerank) | giống TRAKE (code dùng chung) | Không | ✅ 25s |
| KIS Detail 2-Stage | `/api/kis-detail-2stage` | **1** (model đầu tiên) | 1 lần/subquery ×2 stage | Không | ✅ 3s |
| VQA | `/api/vqa-search` | 3 (SRRF+rerank) | 1 lần/model | **Có** (Qwen/OpenRouter) | ✅ 9s |

## Điểm chung mấu chốt (vừa sửa trong phiên này)

Tất cả 5 track giờ **dùng chung đúng 1 retriever** (`_get_retriever(experiment)`, cache theo tên
experiment) thay vì mỗi track/mỗi request tự build model riêng. Với 3 model trên GPU 4 GB, đây là
khác biệt giữa "chạy được" và "CUDA out of memory" — 2 retriever độc lập (6 embedder) không vừa,
1 retriever dùng chung (3 embedder) thì vừa, dư khoảng 150 MiB.
