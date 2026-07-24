# Paper 3 — Vortex (VNU-HCM University of Science)

> **Nguồn:** "Vortex: Multi-Modal Fusion System for Intelligent Video Retrieval"
> (arXiv:2606.19682v1 [cs.CV], 18 Jun 2026), Nguyen, Tran-Minh, Lam, Ly, Huynh,
> Tran, Le — đội **FocusOnFun**, PDF gốc đầy đủ (`aic.pdf`).
>
> Kết quả: **79.6 / 88 (90.5%)** ở Preliminary Round, đạt hạng **Excellent** ở
> Final Round (Excellent cho TKIS, Very Good cho VKIS, Very Good cho TRAKE,
> **Outstanding** cho Q&A) — điểm sơ khảo cao nhất trong 3 hệ thống đối chiếu
> (so với 77.40 của LLandMark và 76.4 của Cascaded System). Đây là tài liệu
> **thuật toán tham khảo** để implement/nâng cấp — luôn cross-check trạng thái
> "Implemented / Not implemented" trước khi giả định code đã có sẵn. Xem
> [`../CLAUDE.md`](../CLAUDE.md) cho bản đồ mã nguồn thật, và
> [`paper1_llandmark.md`](paper1_llandmark.md) /
> [`paper2_cascaded_system.md`](paper2_cascaded_system.md) để so sánh với 2 hệ
> thống đối chiếu khác.

---

## 0. Ba đóng góp chính của paper

1. **Two-stage adaptive keyframe extraction**: AutoShot cho shot boundary
   detection, sau đó lọc keyframe trong mỗi shot bằng ngưỡng L2-norm động trên
   embedding CLIP (không phải số keyframe cố định/shot như 2 paper còn lại).
2. **Hybrid dual-embedding retrieval qua RRF**: CLIP (semantic toàn cục) +
   SigLIP2 (chi tiết tinh vi) fusion bằng Reciprocal Rank Fusion **thuần rank**
   (khác SRRF của paper2 — không giữ lại similarity score gốc).
3. **Vòng lặp tương tác thủ công mạnh**: Rocchio relevance feedback (like/dislike
   do người dùng gắn nhãn) + multi-stage Temporal Search (3 ô input Before/Now/
   After) + LLM chỉ đóng vai trò gợi ý làm rõ query (không tự động rewrite).

**Khác biệt triết lý cốt lõi so với paper1/paper2:** Vortex đặt cược vào
**human-in-the-loop tốc độ cao** (OCR filter thủ công qua lệnh `/filterall`,
"nearby frame" để dò tay, Query-by-Example) thay vì tự động hóa bằng agent LLM
— đây có thể là lý do điểm thi cao hơn dù kiến trúc "kém thông minh" hơn về mặt
suy luận tự động (xem §8 để phân tích sâu hơn).

---

## 1. Two-Stage Adaptive Keyframe Extraction

📍 Đối chiếu code: cần kiểm tra `src/preprocessing/keyframe_extraction.py` (hoặc
tương đương) — **chưa xác nhận trạng thái implement**, vì repo hiện tại (theo
2 file paper1/paper2) đang dùng logic khác (TransNetV2 + percentile cố định).
Đây là **kỹ thuật thứ 3, khác cả 2**, cần đối chiếu riêng.

```
Video → AutoShot (shot boundary detection) → Shots
  Trong mỗi shot:
    → CLIP (ViT-L-14-quickgelu, pretrained DFN2B) trích embedding
      MỖI 8 FRAME một lần (không phải mọi frame — tối ưu tốc độ)
    → So sánh frame hiện tại (e_current) với keyframe giữ gần nhất (e_prev)
      bằng L2-norm tương đối:

      rel_diff = ||e_current - e_prev|| / ||e_prev||        (Eq. 1)

    → Nếu rel_diff > 0.4 (ngưỡng thực nghiệm) → giữ làm keyframe mới
```

**Khác biệt quan trọng so với paper1 (percentile cố định [0.15, 0.5, 0.85]) và
paper2 (3 keyframe/shot cố định):**

| Tiêu chí | Vortex (paper3) | LLandMark (paper1) | Cascaded (paper2) |
|---|---|---|---|
| Shot detection | AutoShot | TransNetV2 | TransNetV2 |
| Số keyframe/shot | **Động** (0, 1, nhiều tùy nội dung) | Cố định 3 (percentile) | Cố định 3 (representative) |
| Cơ chế chọn | L2-norm threshold trên CLIP embedding | Percentile vị trí thời gian | Không nêu công thức cụ thể |
| Chi phí tính toán | Cần chạy CLIP trên mọi frame lấy mẫu (mỗi 8 frame) trong bước preprocessing | Không cần embedding, chỉ cần vị trí thời gian | Không nêu rõ |

**Trade-off paper tự nêu rõ (đáng giữ lại khi justify thiết kế):** sampling mỗi
8 frame là tối ưu hiệu năng cho dữ liệu quy mô lớn; lọc L2-norm là phần "adaptive"
đảm bảo chỉ giữ frame có thay đổi hình ảnh đáng kể — đánh đổi là **có thể bỏ lỡ
sự kiện thoáng qua xảy ra giữa khoảng 8-frame**, nhưng paper chấp nhận đánh đổi
này để ưu tiên hiệu năng và bắt tốt các thay đổi cảnh lớn (hiệu quả cho tác vụ
event-level retrieval theo thực nghiệm của nhóm).

**Điểm tích hợp gợi ý:** nếu muốn thử nghiệm phương án "adaptive" thay cho
"percentile cố định" hiện tại, đây là ứng viên thay thế trực tiếp cho bước chọn
keyframe trong mỗi shot — cần thêm 1 lần gọi CLIP nhỏ (không phải model chính
dùng để index) chỉ để tính rel_diff, tách biệt khỏi embedding chính thức lưu
vào vector DB.

---

## 2. Hybrid Retrieval — Reciprocal Rank Fusion (RRF thuần, KHÔNG phải SRRF)

📍 Đối chiếu code: nếu `src/retrieval/fusion.py` đã có SRRF (theo mô tả trong
paper2_cascaded_system.md §3), thì RRF thuần của Vortex là **phiên bản đơn giản
hơn** — có thể coi là baseline/ablation để so sánh, không cần implement thêm
nếu SRRF đã hoạt động tốt.

```
RRF_Score(d) = Σ_{i=1}^{N} 1 / (k + rank_i(d))          (Eq. 2)

  N = 2 (CLIP và SigLIP2)
  k = 60 (hằng số thực nghiệm, giảm độ nhạy với kết quả xếp hạng thấp)
```

**Điểm khác biệt cốt lõi so với SRRF (paper2):** RRF thuần của Vortex **chỉ dùng
rank position**, bỏ hoàn toàn giá trị similarity gốc. Điều này nghĩa là 2
candidate cùng đứng hạng 1 nhưng có similarity 0.99 và 0.55 sẽ được đối xử
**y hệt nhau** trong công thức — đây chính là nhược điểm mà paper2 (SRRF) khắc
phục bằng cách nhân thêm hệ số `s_m(d)` vào tử số.

**Model dùng cho 2 nhánh embedding:**
- **CLIP** (DFN5B pretrained) — cho ngữ cảnh semantic toàn cục (1024-dim)
- **SigLIP2** — cho nhận diện chi tiết tinh vi (1152-dim)

Cả 2 được lưu **riêng trong Milvus** (không rõ có dùng named-vector 1-point-nhiều-
field như Qdrant của paper2 hay không — paper không nêu chi tiết schema).

**Không có bước rerank bằng VLM/cross-encoder** (khác hẳn paper2 dùng BLIP-2 ITM)
— đây là lựa chọn kiến trúc có chủ đích để giữ pipeline nhẹ, tốc độ nhanh, đánh
đổi lấy độ chính xác fine-grained mà cross-encoder mang lại.

---

## 3. Multi-Modal Metadata Generation — OCR + Captioning (dùng CHUNG 1 model)

📍 Đối chiếu code: `modules/ocr/base.py` (nếu là stub như paper2 mô tả) —
**chưa implement**. Nếu implement theo hướng Vortex, cần chú ý đây là 1 model
làm **2 nhiệm vụ tách biệt** (không phải 1 nhiệm vụ như paper1/paper2 chỉ OCR).

```
Qwen2.5-VL-3B-Instruct chạy trên mỗi keyframe, sinh 2 trường riêng biệt:
  1. OCR field       → đọc chữ viết thật xuất hiện trong khung hình
                        (biển hiệu, banner, tỷ số, logo kênh...)
  2. Description field → caption tự nhiên mô tả nội dung ảnh
                        (hành động, vật thể, bối cảnh — KHÔNG phải đọc chữ)

Cả 2 field → index vào Elasticsearch
```

**Điểm khác biệt quan trọng cần lưu ý khi implement — dễ nhầm lẫn:**
- Paper1 (LLandMark): chỉ có OCR (PaddleOCR + Gemini sửa lỗi), **không có
  captioning riêng**.
- Paper2 (Cascaded): chỉ có OCR (Gemini 2.0 Flash JSON prompt), **không có
  captioning riêng**.
- **Vortex là paper DUY NHẤT trong 3 paper có captioning tường minh**, sinh mô
  tả ngôn ngữ tự nhiên cho mỗi keyframe — đây là input quan trọng cho Video KIS
  task (dùng mô tả sinh ra để so khớp semantic khi query là 1 đoạn video ngắn,
  không có text hint).

**Automatic Speech Recognition:** dùng Whisper (không nêu rõ version, khác
Whisper Large-v3 của paper2 và WhisperX của paper1). Cơ chế alignment: mỗi
frame timestamp match vào khoảng transcription `(a, b)` chứa nó; nếu rơi vào
khoảng lặng thì **propagate câu nói gần nhất về phía trước** để đảm bảo mọi
keyframe đều có trường transcription (không để trống).

**Điểm tích hợp gợi ý vào code hiện tại:** nếu module OCR hiện tại chỉ trích
xuất text, cân nhắc mở rộng prompt để sinh **thêm 1 field caption riêng** trong
cùng 1 lần gọi model (tiết kiệm hơn gọi 2 lần) — miễn model đủ mạnh làm cả 2 tác
vụ trong 1 prompt (Qwen2.5-VL-3B-Instruct được paper mô tả là "trade-off tốt
giữa độ chính xác và hiệu năng" cho việc này).

---

## 4. Temporal Search — Additive Scoring (khác cả paper1 và paper2)

📍 Đối chiếu code: `src/retrieval/temporal_search.py` — nếu hiện tại đang có
frame-to-frame walk (tolerance tĩnh, theo mô tả paper2_cascaded_system.md §2)
hoặc min-score intersection (theo paper1_llandmark.md §3), thì đây là **cơ chế
thứ 3, khác cả 2**, cần so sánh riêng trước khi chọn hướng implement.

```
Input: 3 query riêng biệt do NGƯỜI DÙNG tự nhập (không phải LLM tự decompose):
  Q_previous, Q_current, Q_next  (3 ô input trên UI — xem Fig. 5 trong paper)

Bước 1 — Independent Retrieval (dùng Milvus HNSW ANN):
  R_previous ← Search(Q_previous)
  R_current  ← Search(Q_current)
  R_next     ← Search(Q_next)

Bước 2 — Compute Best Scores per Video (dùng hash map, O(K)):
  bestPrev[video_id] ← max(bestPrev[video_id], score)  ∀ (video_id, score) ∈ R_previous
  bestNext[video_id] ← max(bestNext[video_id], score)  ∀ (video_id, score) ∈ R_next

Bước 3 — Temporal Re-scoring (O(K)):
  S_final(r_c) = S(r_c) + bestPrev[video_id] + bestNext[video_id]     (Eq. 4)
  (nếu không tìm thấy match cùng video_id ở previous/next → coi Smax = 0)

Bước 4 — Re-ranking (O(K log K)):
  Sort R*_current giảm dần theo S_final
```

**Độ phức tạp tổng: O(K log K)** — paper có phân tích chi tiết từng bước (Step
1 dùng ANN nên là baseline cost, Step 2-3 là O(K), Step 4 là O(K log K) do sort
— đây là bước dominant).

### So sánh 3 cách tiếp cận temporal search — bảng tổng hợp quan trọng

| Tiêu chí | Vortex (paper3, file này) | LLandMark (paper1) | Cascaded (paper2) |
|---|---|---|---|
| Cách sinh sub-query | **Người dùng tự nhập** 3 ô input | LLM tự decompose N sự kiện | GPT-4o tự decompose N sự kiện + beam search |
| Công thức scoring | **Additive, cộng điểm max của video cùng ID** (Eq. 4) | Min-score intersection (weakest-link) | Additive (exploration) + Multiplicative (validation, 2 giai đoạn) |
| Có phạt theo khoảng cách thời gian không | **Không** — chỉ cần cùng video_ID, không quan tâm frame cách nhau bao xa | Không | **Có** — λ-decay theo Δt (Eq. 3) |
| Độ phức tạp | O(K log K) | Không nêu rõ, nhưng set intersection ~O(N×K) | O(B×K×M) nhờ beam search (thay vì O(M^K) brute-force) |
| Rủi ro chính | Không phạt gap thời gian lớn → có thể ghép nhầm 2 sự kiện xa nhau trong cùng video thành 1 chuỗi hợp lệ | Phạt nặng nếu 1 bước bị miss (không khoan dung) | Phức tạp nhất, cần tune α và beam width |
| Cần hyperparameter tune không | Không (không có α, tolerance...) — điểm mạnh về đơn giản | Không | Có (α=0.01, beam width=8 theo paper2) |

**Nhận xét quan trọng khi implement (rủi ro cần lưu ý):** công thức (4) của
Vortex **không có cơ chế phạt theo khoảng cách thời gian** — chỉ cần frame
"previous" và "next" thuộc cùng video_ID là được cộng điểm, bất kể chúng cách
nhau 2 giây hay 2 phút. Đây là điểm yếu tiềm ẩn nếu video dài và có nhiều sự
kiện lặp lại tương tự — khác hẳn cơ chế λ-decay của paper2 (Eq. 3) được thiết
kế chính để giải quyết vấn đề này. Paper Vortex biện minh bằng lý do tốc độ
(tránh DP tốn kém) chứ không thảo luận rủi ro "ghép nhầm" này.

**Lý do chọn heuristic thay vì DP (paper tự nêu, đáng giữ khi so sánh):**
Dynamic Programming truyền thống cho alignment 2 chuỗi hữu hạn tốn chi phí tính
toán đáng kể khi áp dụng lên toàn bộ database ở chế độ tương tác thời gian thực
— đây là lý do chính đáng dùng heuristic re-ranking thay vì DP, tương tự tinh
thần "tránh combinatorial explosion" mà paper2 giải quyết bằng beam search.

**Điểm tích hợp gợi ý:** đây là phương án **đơn giản nhất, ít hyperparameter
nhất** trong 3 cách — phù hợp làm **baseline nhanh** để so sánh trước khi đầu
tư vào beam search + λ-decay phức tạp hơn của paper2. Có thể implement trước
làm mốc so sánh (ablation), vì không cần tune α/beam width.

---

## 5. Query Refinement — Rocchio-based Relevance Feedback (ĐỘC QUYỀN của Vortex)

📍 Đối chiếu code: **chưa thấy đề cập trong paper1/paper2** — đây là kỹ thuật
**chỉ có ở Vortex**, cần đánh giá độ ưu tiên implement riêng.

```
q_m = α·q_0 + β · (1/|C_r|) · Σ_{d_j ∈ C_r} d_j − γ · (1/|C_nr|) · Σ_{d_j ∈ C_nr} d_j    (Eq. 3)

  q_0   = query vector gốc
  C_r   = tập keyframe được người dùng gắn nhãn "Prefer this answer" (like)
  C_nr  = tập keyframe được người dùng gắn nhãn "Not prefer" (dislike)
  α, β, γ = hệ số điều khiển ảnh hưởng của query gốc / tập relevant / tập non-relevant
```

**Cơ chế UI:** sau lần search đầu, người dùng có thể gắn nhãn like/dislike cho
bất kỳ số lượng keyframe nào trong kết quả trả về. Hệ thống dùng nhãn đó tính
lại `q_m`, rồi **re-query Milvus bằng vector mới** để trả kết quả tinh chỉnh hơn.

**Đánh giá vai trò trong kiến trúc:** đây là kỹ thuật **cổ điển** (Vector Space
Model, không phải deep learning) nhưng bù lại **cực nhẹ về tính toán** (chỉ là
phép cộng/trừ vector có trọng số) và **tận dụng trực tiếp phản hồi con người**
— khớp với triết lý "human-in-the-loop tốc độ cao" của toàn bộ hệ thống Vortex
(xem thêm ở §8).

**So sánh với 2 paper còn lại:** paper1 (LLandMark) và paper2 (Cascaded) **không
có cơ chế relevance feedback nào** — cả 2 đều dựa hoàn toàn vào 1 lượt search +
rerank tự động bằng agent/BLIP-2, không có vòng lặp tương tác lấy nhãn người
dùng ngay trong phiên truy vấn.

**Điểm tích hợp gợi ý:** nếu hệ thống hiện tại đã có UI cho phép người dùng
chọn/loại kết quả (like/dislike button), đây là kỹ thuật **rẻ và nhanh để thêm
vào** — chỉ cần lưu 2 tập C_r/C_nr theo phiên, tính lại vector Eq. 3, và gọi
lại search — không cần thêm model mới, không cần GPU, độ trễ gần như tức thời.

---

## 6. LLM-Assisted Query Interpretation — CHỦ ĐÍCH KHÔNG tự động rewrite

📍 Đối chiếu code: nếu `query_processor.py::LlmQueryProcessor` (theo mô tả trong
paper1/paper2) đang tự động enrich/dịch query mà không hỏi lại người dùng, đây
là **điểm khác biệt triết lý quan trọng cần cân nhắc**, không chỉ là chi tiết
kỹ thuật.

```
Query mơ hồ (VD: "building")
  → LLM sinh NHIỀU gợi ý cụ thể (VD: "a tall office building",
    "a building under construction", "a university campus building")
  → Hệ thống HIỂN THỊ gợi ý cho người dùng
  → Người dùng TỰ CHỌN gợi ý phù hợp (không tự động áp dụng)
```

**Lý do paper nêu rõ (rationale quan trọng, khác hẳn triết lý agent của paper1/
paper2):** nhiều hệ thống hiện đại dùng LLM để tự động rewrite query, nhưng cách
này có rủi ro **intent drift** (câu rewrite lệch khỏi ý định gốc) và
**hallucination**. Trong bối cảnh thi đấu tương tác thời gian thực, việc giữ
quyền kiểm soát và độ trung thực của query (query fidelity) được đánh giá là
quan trọng hơn tốc độ tự động hóa.

**So sánh trực diện với paper1/paper2 — đây là khác biệt triết lý cốt lõi:**

| | Vortex (paper3) | LLandMark (paper1) | Cascaded (paper2) |
|---|---|---|---|
| LLM có tự sửa query không | **Không — chỉ gợi ý, người dùng chọn** | Có — tự dịch + enrich, không hỏi lại | Có — tự expand thành N=4 biến thể, tự chạy song song |
| Rủi ro chính | Chậm hơn vì cần thao tác người dùng | Agent hiểu sai → cả pipeline lệch hướng | GPT-4o hiểu sai → toàn bộ N biến thể có thể cùng lệch hướng |
| Độ tin cậy | Cao (con người luôn duyệt cuối cùng) | Phụ thuộc độ chính xác của agent | Phụ thuộc độ chính xác của agent |
| Độ trễ (latency) | Có "human think time" xen giữa | Nhanh hơn — không cần chờ người chọn | Chậm nhất về mặt tính toán (N=4 lần search song song) |

**Đánh giá tổng thể (giả thuyết, cần thực nghiệm để xác nhận):** đây có thể là
**một trong các lý do Vortex đạt điểm sơ khảo cao hơn** dù kiến trúc "kém tự
động hóa" hơn — giảm rủi ro sai lệch do AI tự quyết định, đổi lại bằng việc yêu
cầu con người tham gia nhiều hơn. Đây chỉ là suy luận hợp lý từ so sánh kiến
trúc và điểm số, paper không có ablation trực tiếp so sánh "tự động hoàn toàn"
với "gợi ý rồi người dùng chọn" để chứng minh nhân quả.

---

## 7. Hạ tầng — Milvus + Elasticsearch + Redis (khác cả paper1 và paper2)

📍 Đối chiếu code: hiện tại dùng Qdrant (theo paper2_cascaded_system.md), khác
Milvus của cả Vortex (paper3) và LLandMark (paper1).

```
Milvus       → vector search cho CLIP + SigLIP2 embeddings
Elasticsearch → text indexing (OCR + Description + ASR transcription) + metadata filtering
Redis        → cache kết quả cuối, giảm độ trễ cho query lặp lại
```

**Điểm khác biệt so với 2 paper còn lại:** Vortex là paper **duy nhất có tầng
cache riêng (Redis)** — cả paper1 (dùng MongoDB cho metadata) và paper2 (dùng
Qdrant payload) đều không nhắc đến cơ chế caching kết quả tìm kiếm.

```
Bảng so sánh hạ tầng nhanh (bổ sung từ paper1_llandmark.md §6):

| Tiêu chí          | Milvus (Vortex, LLandMark) | Qdrant (Cascaded, code hiện tại) |
|--------------------|------------------------------|-----------------------------------|
| Setup             | Nặng (cần etcd + MinIO)      | Single binary                      |
| Latency p50       | ~10ms                         | ~4ms                               |
| GPU index (CAGRA) | ✅ độc quyền                  | ❌                                  |
| Cache riêng        | Redis (chỉ Vortex có)        | Không nêu rõ trong paper2          |
```

**Đánh giá:** không có lý do rõ ràng để chuyển hạ tầng vector DB hiện tại
(Qdrant) sang Milvus chỉ vì Vortex dùng Milvus — như đã kết luận ở
paper1_llandmark.md §6, ở quy mô dataset AIC (~10-20M vector), Milvus có phần
overkill. Tuy nhiên, **cơ chế Redis cache là ý tưởng độc lập, đáng cân nhắc bổ
sung riêng** (không phụ thuộc việc dùng Milvus hay Qdrant) nếu hệ thống hiện tại
chưa có tầng cache cho query lặp lại trong phiên tương tác.

---

## 8. Kết quả thực nghiệm

### 8.1 Điểm số Preliminary Round theo module tích hợp dần

| Round | Điểm | Module tích hợp |
|---|---|---|
| Round 1 (24 câu) | 20.6 | Baseline (chỉ CLIP-only search) |
| Round 2 (30 câu) | 27.8 | Hybrid RRF (CLIP + SigLIP2) |
| Round 3 (35 câu) | 31.2 | Temporal Search + Relevance Feedback |
| **Tổng** | **79.6/88 (90.5%)** | Toàn bộ hệ thống |

**Quan sát quan trọng:** điểm tăng dần rõ rệt theo từng module — đây là bằng
chứng thực nghiệm gián tiếp cho việc **cả RRF hybrid lẫn Temporal+Feedback đều
đóng góp cải thiện đo lường được**, không phải chỉ là thêm độ phức tạp không
cần thiết.

### 8.2 Kết quả Final Round (đánh giá định tính bởi Jury Board)

| Task | Đánh giá |
|---|---|
| TKIS (Textual Known-Item Search) | Excellent |
| VKIS (Video Known-Item Search) | Very Good |
| TRAKE (Temporal Retrieval and Alignment) | Very Good |
| Q&A (Question Answering) | **Outstanding** |

**Điểm mạnh nổi bật nhất theo tự đánh giá của nhóm:** hiệu năng Q&A "Outstanding"
được cho là nhờ pipeline metadata phong phú (OCR + Description + ASR) hoạt động
đặc biệt tốt cho việc trả lời câu hỏi cần đọc hiểu nội dung cụ thể trong khung
hình (VD: đếm số miếng bánh sandwich bị cắt — dựa vào caption + nearby frame,
xem ví dụ qa-query-02 ở §8.3).

### 8.3 Ví dụ định tính đáng tham khảo khi viết test case

- **tkis-query-02:** hint rất cụ thể ("giải phóng khí hidro") → **bỏ qua semantic
  search, dùng thẳng OCR filter** (`/filterall ocr{hidro}`) → tìm đúng ngay lập
  tức. Minh chứng: khi hint đủ đặc trưng, OCR filter trực tiếp nhanh hơn nhiều
  so với semantic embedding search.
- **qa-query-02:** dùng caption sinh sẵn (liệt kê nguyên liệu bánh sandwich) để
  tìm điểm bắt đầu clip, sau đó dùng tính năng **"nearby frame"** để dò tay đến
  đúng bước cắt bánh, xác nhận đáp án bằng mắt thường. Minh chứng: caption +
  thao tác thủ công bổ trợ tốt cho tác vụ QA cần đếm/quan sát chi tiết.
- **vkis-07 (Video KIS, không có text hint):** người dùng tự quan sát video,
  phát hiện chữ trên hiện vật ("DI TICH KIM LONG"), rồi tự gõ OCR filter để tìm
  → tìm đúng ngay. Minh chứng: khả năng OCR tốt trên chi tiết nhỏ (chữ khắc trên
  vật thể) là lợi thế thực chiến rõ rệt.
- **trake-03 (TRAKE, 3 sự kiện: bàn thắng 1, cứu phạt đền, bàn thắng 2):** chiến
  thuật coarse-to-fine 3 bước — (1) OCR filter bảng tỷ số ("PHI 1 0 BRU") để tìm
  frame ngay sau bàn thắng, (2) dùng "nearby frame" dò ngược lại để tìm chuỗi sự
  kiện dẫn đến bàn thắng, (3) Query-by-Example (dùng chính 1 frame vừa tìm làm
  query ảnh) để định vị chính xác khoảnh khắc bóng qua vạch vôi. Minh chứng: kết
  hợp OCR + temporal navigation thủ công + Query-by-Example là chiến thuật hiệu
  quả cho TRAKE, dùng lặp lại cho cả 3 sự kiện E1/E2/E3.

**Điểm chung của các ví dụ:** tất cả đều thể hiện rõ **con người can thiệp trực
tiếp, nhiều bước, tận dụng tối đa các công cụ filter/navigate thủ công** — khác
hẳn triết lý "để agent tự quyết định modality và trọng số" của paper1/paper2.

---

## 9. So sánh tổng hợp 3 hệ thống

| Hạng mục | Vortex (paper3, file này) | LLandMark (paper1) | Cascaded (paper2) |
|---|---|---|---|
| Shot Detection | AutoShot | TransNetV2 | TransNetV2 |
| Keyframe/shot | Động (L2-norm threshold) | Cố định 3 (percentile) | Cố định 3 (không rõ công thức) |
| Visual Embedding | CLIP (DFN5B) + SigLIP2 | CLIP ConvNeXt-XXL | BEiT-3 + SigLIP |
| Fusion trong-modality | RRF thuần (chỉ rank) | Không rõ | SRRF (giữ similarity gốc) |
| Reranking | Không có (không dùng cross-encoder) | Weighted Fusion Agent | BLIP-2 ITM (cross-encoder) |
| OCR | Qwen2.5-VL-3B-Instruct | PaddleOCR + Gemini 2.5 Flash | Gemini 2.0 Flash |
| Captioning riêng | **Có** (cùng model với OCR) | Không | Không |
| ASR | Whisper | WhisperX | Whisper Large-v3 |
| Vector DB | Milvus | Milvus | Qdrant |
| Text Search | Elasticsearch | Elasticsearch | Elasticsearch |
| Cache | **Redis (độc quyền)** | Không có | Không có |
| Metadata store | (không rõ, có thể tích hợp Elasticsearch) | MongoDB (đánh giá: redundant) | Qdrant payload |
| Object Detection | Không có | YOLOv9-e | Không có |
| Query Expansion/Interpretation | LLM gợi ý, **người dùng tự chọn** | Không nêu rõ | GPT-4o tự sinh N=4 biến thể tự động |
| Modality Routing | Không có agent — người dùng tự chọn mode (CLIP-only/SigLIP2-only/RRF) | Landmark Knowledge Agent (có điều kiện) | GPT-4o phân rã + trọng số động mọi query |
| Temporal Search | Additive, không phạt theo Δt (Eq. 4) | Min-score intersection (weakest-link) | Beam search + λ-decay 2 giai đoạn |
| Relevance Feedback | **Rocchio (độc quyền, có UI like/dislike)** | Không có | Không có |
| Landmark reasoning (địa danh VN) | Không có | **Có (điểm mạnh riêng)** | Không có |
| Điểm sơ khảo | **79.6 / 88 (90.5%)** — cao nhất | 77.40 / 88 | 76.4 / 88 |
| Đánh giá tổng thể | Pipeline đơn giản, không dùng cross-encoder rerank hay agent phức tạp, nhưng bù lại bằng vòng lặp tương tác con người mạnh (Rocchio + temporal thủ công + query suggestion không ép buộc) | Điểm cao, kiến trúc multi-agent + MongoDB hơi over-engineering | Pipeline tuyến tính sạch, công thức toán tường minh, agent tự động hoàn toàn nhưng độ trễ và rủi ro sai lệch agent cao hơn |

### Giả thuyết về nguyên nhân điểm số (cần thực nghiệm để xác nhận, không phải kết luận chắc chắn)

Vortex đạt điểm sơ khảo cao nhất dù có **ít tự động hóa bằng AI nhất** trong 3
hệ thống (không cross-encoder rerank, không agent tự quyết định trọng số
modality, LLM chỉ gợi ý chứ không tự sửa query). Các giả thuyết hợp lý (rút ra
từ so sánh kiến trúc + ví dụ định tính, KHÔNG phải kết luận đã được paper chứng
minh bằng ablation trực tiếp):

1. **Giảm rủi ro lỗi dây chuyền từ agent:** paper1/paper2 phụ thuộc LLM bên
   ngoài (Gemini/GPT-4o) để quyết định trọng số/modality — nếu LLM hiểu sai,
   cả pipeline lệch hướng. Vortex tránh rủi ro này bằng cách để con người quyết
   định cuối cùng (chọn gợi ý, gắn nhãn like/dislike, tự nhập 3 ô temporal).
2. **Độ trễ thấp hơn ở phần tính toán tự động:** không cần gọi API LLM lớn để
   decompose/expand/route mỗi query — có thể bù lại bằng thời gian con người
   thao tác, nhưng loại bỏ được biến số "LLM response time" khỏi pipeline lõi.
3. **Tối ưu hóa cho tương tác thời gian thực có con người kiểm soát**, đúng như
   paper tự mô tả triết lý thiết kế (§3.6 trong paper gốc) — đây là lựa chọn
   thiết kế có chủ đích, không phải thiếu sót kỹ thuật.

**Lưu ý khi dùng giả thuyết này để ra quyết định implement:** đây là suy luận
từ so sánh gián tiếp (điểm số + kiến trúc), không phải kết quả ablation study
trực tiếp so sánh "có agent" vs "không agent" trên cùng 1 hệ thống. Trước khi
quyết định bỏ agent/rerank để chạy theo hướng Vortex, nên cân nhắc thử nghiệm
A/B trên chính dataset và use case hiện tại.

---

## Cách dùng tài liệu này

Khi cần **baseline nhanh, ít hyperparameter** cho temporal search, tham khảo §4
(công thức Eq. 4, đơn giản hơn beam search + λ-decay của paper2 nhưng thiếu cơ
chế phạt khoảng cách thời gian — cân nhắc rủi ro ghép nhầm sự kiện xa nhau).
Khi cần thêm **relevance feedback dựa trên tương tác người dùng** (like/dislike),
tham khảo §5 — đây là kỹ thuật độc quyền của Vortex, rẻ và nhanh để tích hợp.
Khi cần **captioning riêng biệt với OCR** (không chỉ trích xuất text mà còn mô
tả ngữ nghĩa ảnh), tham khảo §3. Khi thiết kế UI cho phép người dùng **kiểm
soát/xác nhận trước khi query được sửa** (thay vì để LLM tự động rewrite hoàn
toàn), tham khảo §6 để hiểu rationale.

Luôn kiểm tra trạng thái Implemented/Not ở đầu mỗi mục — phần lớn nội dung file
này (adaptive keyframe L2-norm, RRF thuần, captioning kết hợp OCR, temporal
additive Eq. 4, Rocchio relevance feedback, Redis cache) **chưa được xác nhận
có trong code hiện tại**; cần đối chiếu trực tiếp với `CLAUDE.md` và codebase
thật trước khi giả định bất kỳ phần nào đã hoạt động.
