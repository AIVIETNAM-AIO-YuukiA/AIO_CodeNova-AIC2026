# Paper 1 — LLandMark (UIT + HCMUT)

> **Trạng thái nguồn:** tóm tắt từ `hcmc_ai_challenge_pipeline_analysis.md` (chưa có
> PDF gốc đầy đủ). Nếu paper gốc được cung cấp sau, file này cần viết lại theo đúng
> công thức/số liệu trong PDF thay vì bản tóm tắt hiện tại.
>
> Kết quả: **77.40 / 88** (Top 56 / 410+ đội). Đây là tài liệu **thuật toán tham
> khảo** để implement/nâng cấp — luôn cross-check trạng thái "Implemented / Not
> implemented" trước khi giả định code đã có sẵn. Xem [`../CLAUDE.md`](../CLAUDE.md)
> cho bản đồ mã nguồn thật, và [`paper2_cascaded_system.md`](paper2_cascaded_system.md)
> để so sánh với hệ thống đối chiếu.

---

## 1. Kiến trúc tổng quan

### Offline Pipeline

```
Video → TransNetV2 (shot boundary detection) → Keyframes
  ├── CLIP ConvNeXt-XXLarge          → Milvus (vector search)
  ├── PaddleOCR + Gemini 2.5 Flash correction → Elasticsearch
  ├── WhisperX (word-level alignment) → Elasticsearch
  └── YOLOv9-e (COCO classes)         → JSON file (object metadata)
Keyframe metadata → MongoDB
```

### Online Pipeline

```
Query → Parsing & Planning Agent
      → Landmark Knowledge Agent (kích hoạt nếu phát hiện địa danh trong query)
      → Parallel Search trên 4 modality (visual / landmark / OCR / ASR)
      → Weighted Fusion + Reranking Agent
      → Kết quả cuối
```

Khác biệt cốt lõi với Cascaded System (paper 2): kiến trúc **multi-agent** (Parsing
Agent, Landmark Knowledge Agent, Reranking Agent tách rời) thay vì 1 pipeline
tuyến tính retrieve-then-rerank; và **MongoDB riêng cho metadata** thay vì dùng
payload của vector DB.

---

## 2. LLM-Assisted Landmark Image-to-Image Search — **chưa implement**

Đây là đóng góp nổi bật nhất của paper, nhắm giải quyết điểm yếu chung của mọi
embedding text-image model: tên riêng địa danh (landmark) hiếm gặp trong dữ liệu
pretrain nên CLIP/SigLIP hiểu rất kém.

```
User query: "Nhà thờ Đức Bà"
  1. Landmark Knowledge Agent dùng Gemini để nhận diện đây là tên địa danh
  2. Gemini sinh mô tả visual chi tiết bằng tiếng Anh, ví dụ:
       "Twin square bell towers, dark gray stone facade, Gothic architecture,
        red-brick color, colonial French style"
  3. Google Custom Search API dùng mô tả đó (hoặc trực tiếp tên landmark)
     để lấy ảnh reference thật của địa danh
  4. CLIP/SigLIP encode ảnh reference → vector ảnh
  5. Vector ảnh reference được dùng để search trong vector DB
     (image-to-image search, thay vì / kèm với text-to-image search thông thường)
```

**Lý do hiệu quả:** thay vì hy vọng model hiểu được cụm từ "Nhà thờ Đức Bà" (named
entity, hiếm trong pretrain corpus), hệ thống vòng qua bằng cách chuyển landmark →
mô tả hình ảnh (LLM làm tốt việc này) → ảnh thật (search engine) → vector ảnh
(embedding model làm tốt việc so khớp ảnh-ảnh hơn text-ảnh cho chi tiết kiến trúc).

**Rủi ro/hạn chế cần lưu ý khi implement:**
- Phụ thuộc Google Custom Search API — cần quota/API key riêng, và độ trễ mạng có
  thể vi phạm ngân sách 5 phút/query nếu không cache.
- Chỉ có lợi cho query có landmark rõ ràng — cần bước phát hiện named-entity
  landmark trước khi kích hoạt nhánh này (không chạy cho mọi query).
- Nên cache kết quả ảnh reference theo tên landmark đã từng tra (landmark Việt Nam
  là tập hữu hạn, không cần gọi API mỗi lần).

**Điểm tích hợp gợi ý vào code hiện tại:** `src/retrieval/query_processor.py::LlmQueryProcessor`
đã có sẵn cơ chế gọi Gemini để dịch/enrich query (đã implement) — nhánh landmark
reasoning ở trên là một mở rộng tự nhiên: thêm bước phát hiện landmark → nếu có,
sinh mô tả + tra ảnh reference + encode → truyền vector ảnh này song song với vector
text vào `retrieval/search.py`, rồi fusion 2 kết quả (có thể tái dùng SRRF ở
[paper2_cascaded_system.md §3](paper2_cascaded_system.md#3-srrf--score-reflected-reciprocal-rank-fusion)).

---

## 3. Temporal Search — Min-Score Intersection

📍 Đối chiếu code: `src/retrieval/temporal_search.py` (frame-to-frame walk hiện tại
khác cách tiếp cận này — xem [CLAUDE.md](../CLAUDE.md)). **Chưa implement min-score
intersection.**

Dùng khi TRAKE cần chuỗi N sự kiện theo thứ tự, mỗi sự kiện có một câu mô tả riêng
trong query:

```
Query phân rã thành N bước (event) tuần tự: step_1, step_2, ..., step_N
Với mỗi video ứng viên v:
    S_i(v) = similarity của v (tại vị trí phù hợp) với step_i

Score(v) = min_{i=1..N} S_i(v)
```

Video `v` chỉ được xếp hạng cao nếu **mọi bước** đều có similarity tốt — logic
"weakest link" (điểm yếu nhất quyết định điểm chung). Đơn giản, dễ cài đặt, nhưng:

- **Ưu điểm:** không cần tune hyperparameter (không có α, tolerance, beam width...).
- **Nhược điểm:** phạt nặng nếu 1 bước bị miss do nhiễu embedding (VD OCR/ASR sai,
  hoặc keyframe không rơi đúng vào khoảnh khắc mô tả) — không có cơ chế "khoan
  dung" theo khoảng cách thời gian như beam search + λ-decay của paper 2. Đây chính
  là hạn chế mà Cascaded System cải tiến (xem
  [paper2_cascaded_system.md §2](paper2_cascaded_system.md#2-temporal-search-with-adaptive-decay-and-multi-stage-refinement)).

---

## 4. Reranking — Weighted Fusion Agent

**Chưa implement dạng multi-agent riêng** (BLIP-2 ITM đơn lẻ đã có trong code, xem
CLAUDE.md, nhưng không phải kiến trúc "Reranking Agent" tổng hợp 4 nguồn của paper
này).

```
Reranking Agent nhận đầu vào: kết quả từ 4 nhánh song song
  (visual search, landmark search, OCR search, ASR search)

score_final(candidate) = Σ_modality  w_modality * score_modality(candidate)

  w_modality: trọng số cố định hoặc do agent quyết định theo loại query
```

Khác với **Adaptive Score Fusion** của paper 2 (trọng số do LLM dự đoán động theo
từng query, có min-max normalization tường minh — xem
[paper2_cascaded_system.md §4](paper2_cascaded_system.md#4-adaptive-score-fusion)),
paper 1 mô tả cơ chế fusion tổng quát hơn nhưng ít chi tiết công thức cụ thể (tài
liệu nguồn hiện có không nêu rõ w_modality được học/set như thế nào — cần paper gốc
để biết chính xác).

**Chỉ cần thiết khi có ≥3 modality hoạt động song song.** Hiện tại hệ thống mới có
visual search hoạt động đầy đủ (OCR/ASR/text store chưa wired — xem CLAUDE.md phần
"stores/text") nên kiến trúc multi-agent fusion 4 nhánh là **over-engineering ở giai
đoạn hiện tại** — ưu tiên implement OCR/ASR producer trước.

---

## 5. Object Detection — YOLOv9-e

📍 Đối chiếu code: `src/agent/tools.py` hiện dùng **YOLOv8n** (không phải v9-e) làm
placeholder cho `DetectTool`, degrade về text placeholder nếu `ultralytics` chưa cài.

Paper dùng YOLOv9-e (mAP COCO 55.6%, Feb 2024) chạy offline trên mọi keyframe, ghi
kết quả detection ra file JSON riêng (không đẩy vào vector DB) — dùng làm metadata
filter phụ trợ (VD: "tìm cảnh có xe máy" → lọc trước bằng object detection, sau đó
mới rank bằng visual embedding).

**Lý do chọn YOLOv9-e thay vì bản NMS-free (YOLOv10):** NMS truyền thống chạy tốt
hơn khi có nhiều VRAM + batch size lớn (giai đoạn offline không bị ràng buộc
real-time), trong khi NMS-free tối ưu cho edge/TensorRT deployment (không phải nhu
cầu ở đây). NMS trên GPU khó parallelize (sequential nature, warp divergence) nhưng
batch lớn amortize được chi phí này.

**Nếu nâng cấp:** YOLO11 (Sep 2024, mAP 54.7%, tốt hơn v9 overall theo tài liệu
phân tích) là lựa chọn hợp lý hơn YOLOv9-e cho pipeline mới, và cũng hợp lý hơn
YOLOv8n hiện đang dùng trong `agent/tools.py`.

---

## 6. Vector DB — Milvus (đối lập với Qdrant của paper 2 và code hiện tại)

Paper 1 dùng Milvus, không phải Qdrant. So sánh nhanh (từ tài liệu phân tích):

| Tiêu chí | Milvus | Qdrant (đang dùng trong code) |
|---|---|---|
| Setup | Nặng (cần etcd + MinIO) | Single binary |
| Latency p50 | ~10ms | ~4ms |
| GPU index (CAGRA) | ✅ độc quyền | ❌ |
| Billion-scale | Tốt hơn | ⚠️ hạn chế hơn |
| Phù hợp scale ~10-20M vector (AIC) | Overkill | Phù hợp hơn |

Không có lý do để chuyển từ Qdrant sang Milvus ở scale dataset của cuộc thi này —
mục này chỉ để tham khảo khi so sánh kiến trúc, không phải đề xuất thay đổi.

---

## 7. Metadata — MongoDB (đánh giá: over-engineering)

Paper 1 dùng MongoDB riêng để lưu metadata keyframe (group_id, video_id, frame_id,
timestamp, object detection JSON...). Chính tài liệu phân tích tổng hợp
(`hcmc_ai_challenge_pipeline_analysis.md`) đánh giá đây là **redundant** — Qdrant
payload (đã dùng trong code hiện tại, xem `src/repository/`) đủ để thay thế hoàn
toàn:

```
Qdrant payload:
{
  "group_id": "K13",
  "video_id": "V028",
  "frame_id": 2607,
  "timestamp": 86.9
}
→ Đủ để replace MongoDB, không cần thêm 1 DB nữa
```

**Kết luận:** không nên theo hướng MongoDB của paper 1; giữ nguyên kiến trúc
Qdrant-payload-as-metadata-store hiện tại.

---

## So sánh nhanh với Paper 2 (Cascaded System)

Xem bảng so sánh đầy đủ trong
[paper2_cascaded_system.md §8](paper2_cascaded_system.md#8-so-sánh-với-paper-1-llandmark).
Tóm tắt: Paper 1 mạnh hơn về xử lý landmark Việt Nam và điểm thi cao hơn (77.40 vs
76.4), nhưng kiến trúc multi-agent + MongoDB có phần over-engineering so với pipeline
tuyến tính sạch hơn của paper 2. Paper 2 lại có temporal search lý thuyết chặt chẽ
hơn (beam search + λ-decay có công thức tường minh, có ablation) và đạt điểm tuyệt
đối ở Round 3.

---

## Cách dùng tài liệu này

Khi được yêu cầu cải thiện xử lý query có địa danh Việt Nam, hoặc đánh giá kiến trúc
multi-agent vs pipeline tuyến tính, tham khảo mục tương ứng ở trên. Luôn kiểm tra
trạng thái Implemented/Not trước khi đề xuất — phần lớn nội dung file này (landmark
reasoning, min-score intersection, weighted fusion agent) **chưa có trong code**,
chỉ là hướng tham khảo.
