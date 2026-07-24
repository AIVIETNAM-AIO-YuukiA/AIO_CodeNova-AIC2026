# Paper 2 — Cascaded System (UIT + IU + HCMUT)

> **Nguồn:** "Unified Interactive Multimodal Moment Retrieval via Cascaded
> Embedding-Reranking and Temporal-Aware Score Fusion" (AAAI 2026 submission),
> Ngo, Ha, Nguyen Dang, Nguyen Le, Nguyen Nhu — PDF gốc đầy đủ.
>
> Kết quả: **76.4 / 88** (86.8%), vào vòng chung kết, đạt **30/35 (85.7%)** ở
> Round 3. Đánh giá trên AI Challenge 2025 dataset: ~1,500 video, >200GB dữ liệu
> đa phương thức. Đây là tài liệu **thuật toán tham khảo** để implement/nâng cấp —
> luôn cross-check trạng thái "Implemented / Not implemented" trước khi giả định
> code đã có sẵn. Xem [`../CLAUDE.md`](../CLAUDE.md) cho bản đồ mã nguồn thật, và
> [`paper1_llandmark.md`](paper1_llandmark.md) để so sánh với hệ thống đối chiếu.

---

## 0. Ba đóng góp chính của paper

1. **Cascaded dual-embedding retrieval**: BEiT-3 + SigLIP cho broad retrieval, tinh
   chỉnh bằng BLIP-2 reranking — cân bằng recall/precision.
2. **Temporal-aware scoring với exponential decay**: beam search + λ-decay để dựng
   chuỗi sự kiện mạch lạc thay vì frame rời rạc.
3. **Agent-guided query decomposition**: GPT-4o diễn giải query mơ hồ, tách thành
   sub-query theo modality (visual/OCR/ASR), tự động fusion không cần user chọn
   modality thủ công.

---

## 1. Cascaded Dual-Embedding Retrieval Pipeline

### 1.1 Lý do kiến trúc 2 tầng (retrieval-then-rerank)

```
Cross-encoder (BLIP-2):  score(query, frame) jointly — chính xác cao,
                          nhưng chi phí tính toán tăng TUYẾN TÍNH theo số cặp
                          → không khả thi chạy trên toàn bộ index

Dual-encoder (SigLIP, BEiT-3): embed query và frame ĐỘC LẬP → search bằng
                          precomputed index (Qdrant) — nhanh, scale tốt,
                          nhưng thiếu cross-attention token-level nên kém
                          chính xác ở chi tiết tinh vi
```

**Giải pháp cascade:** dùng dual-encoder cho first-pass (tối ưu recall, quét toàn
bộ index), dành cross-encoder (BLIP-2 ITM) cho second-pass rerank trên tập nhỏ đã
lọc (tối ưu precision). Đây chính là pattern `stores/vector` (Qdrant, dual-encoder)
→ `modules/reranker/blip2_itm.py` (cross-encoder) đã có trong code hiện tại — khớp
với kiến trúc paper.

### 1.2 Offline Indexing Pipeline

```
Video
  ↓ tách audio track
  ↓ TransNetV2 (shot boundary detection) trên visual stream
  ↓ Với mỗi shot: extract 3 keyframe đại diện
  ↓
  ├── Visual embedding: BEiT-3 + SigLIP (dual, named vectors) → Qdrant
  ├── OCR: Gemini 2.0 Flash, JSON-based prompt → Elasticsearch
  └── ASR: Whisper Large-v3 → Elasticsearch (segment có timestamp,
        align với keyframe gần nhất theo timeline)
```

**Visual Embedding Generation — lý do dùng CẢ HAI BEiT-3 và SigLIP:** BEiT-3 có độ
chính xác semantic cao (high semantic precision), SigLIP có khả năng tổng quát hóa
rộng (broad generalization). Kết hợp cả hai → retrieval mạnh hơn dùng 1 model đơn.
Cả 2 embedding được normalize và lưu **cùng nhau trong Qdrant bằng named vectors**
(1 point, nhiều vector field) — đây chính xác là cách `stores/vector/qdrant.py`
trong code hiện tại đã làm (xem CLAUDE.md).

**OCR — lý do chọn Gemini 2.0 Flash thay vì OCR truyền thống:** chữ trên màn hình
(banner, biển hiệu) có thể ở nhiều kiểu dáng/hướng/mờ/bị che một phần — MLLM (Gemini)
xử lý context thị giác tốt hơn Tesseract/PaddleOCR, hỗ trợ đa ngôn ngữ (Anh + Việt),
và trả JSON có cấu trúc sẵn sàng để index. **Trạng thái trong code:** chưa có OCR
producer nào (xem CLAUDE.md — `modules/ocr/base.py` là stub thuần).

**ASR — Whisper Large-v3:** chọn vì khả năng đa ngôn ngữ mạnh. Output chia thành
segment có timestamp chính xác, mỗi segment là 1 đơn vị semantic để index, align
với keyframe gần nhất trên timeline để map ngược về visual segment tương ứng.
**Trạng thái trong code:** `agent/tools.py` có Whisper qua optional dependency
nhưng dùng cho agent tool (trả lời câu hỏi), chưa wired thành ASR producer cho
index/text store.

### 1.3 Online Retrieval Pipeline — Visual branch chi tiết

```
Query → [Query Decomposition Agent] → visual sub-query (nếu có)
  → embed bằng SigLIP + BEiT-3 (2 vector riêng)
  → mỗi vector search độc lập trong Qdrant (cosine similarity)
  → 2 ranked list → SRRF fusion (xem §3)
  → top-100 candidates sau SRRF
  → BLIP-2 ITM rerank (cross-attention, fine-grained) → top-K cuối
```

---

## 2. Temporal Search with Adaptive Decay and Multi-Stage Refinement

📍 Đối chiếu code: `src/retrieval/temporal_search.py` hiện có
`temporal_search_forward/backward` + `find_segments` (frame-to-frame walk, dùng
`tolerance` tĩnh) — đây là cơ chế **khác** với beam search + λ-decay mô tả dưới đây.
**Beam search + λ-decay CHƯA implement** trong code hiện tại.

### 2.1 Vấn đề cốt lõi mà kỹ thuật này giải quyết

1. **Combinatorial explosion**: query có K sự kiện, mỗi sự kiện có M frame ứng viên
   → số tổ hợp chuỗi khả dĩ là M^K, bùng nổ tổ hợp khi video dài.
2. **Temporal misalignment**: frame khớp semantic tốt có thể nằm rải rác ở khoảng
   thời gian phi thực tế → chuỗi kết quả rời rạc, không mạch lạc.

### 2.2 Temporal Sequence Construction via Beam Search

Thay vì duyệt toàn bộ M^K tổ hợp, chỉ giữ lại **top-B chuỗi con (beam) tốt nhất** ở
mỗi bước:

```
Input:  K sự kiện (từ query decomposition), M candidate frame/sự kiện, beam width B
Độ phức tạp: O(B × K × M)   thay vì O(M^K)   (exponential → tuyến tính theo K)
```

Đây là **greedy approximation có kiểm soát** — giữ đa dạng trong không gian tìm
kiếm (không chỉ 1 đường đi tham lam duy nhất) nên tránh hội tụ sớm vào local optimum
sai, trong khi vẫn giữ được tính khả thi tính toán.

**Hyperparameter thực nghiệm của paper:** beam width B = 8.

### 2.3 Temporal Decay Weighting — công thức (3)

```
λ_i = e^{-α · Δt_i}          Δt_i = t_i - t_{i-1}
```

- `α`: hyperparameter điều khiển độ nhạy thời gian (temporal sensitivity).
  **Giá trị thực nghiệm của paper: α = 0.01.**
- `Δt_i`: khoảng cách thời gian giữa sự kiện i và sự kiện i-1 trong chuỗi.
- Tính chất: `λ_i → 0` khi `Δt_i → ∞` (phạt mềm gap lớn), `λ_i → 1` khi `Δt_i → 0`
  (khoan dung với độ trễ nhỏ, thực tế).

**Tại sao dùng exponential decay thay vì các lựa chọn khác (paper có rationale rõ
ràng, đáng note lại khi implement):**

| So với | Vấn đề của cách kia | Ưu điểm của exponential decay |
|---|---|---|
| Hard threshold (cắt cứng) | Cutoff nhị phân đột ngột làm invalid cả chuỗi khi vượt ngưỡng | Suy giảm mượt (smooth degradation), không cắt đột ngột |
| Linear decay | Không phản ánh đúng cảm nhận con người về liên kết thời gian | Khớp trực giác: sự kiện gần cảm thấy liên kết mạnh, xa dần thì rời rạc dần theo cấp số nhân |
| ABTS (Nguyen-Nhu et al. 2025) — đo local stability bằng variance trong neighborhood cố định | Chỉ đo ổn định cục bộ | λ-decay là cơ chế **global**, áp đặt ràng buộc trên toàn chuỗi, bổ sung cho tính nhất quán frame cục bộ chứ không thay thế |

Ngưỡng cảm quan minh họa từ paper: `Δt < 2s` → weight gần 1.0 (đóng góp đầy đủ);
`Δt > 10s` → phạt theo cấp số nhân, tự nhiên loại bỏ cấu hình thời gian phi thực tế.

### 2.4 Sequence Scoring — Additive Aggregation — công thức (4), (5)

```
SS_j = Σ_{i=1}^{K}  s_i · e^{-α(t_i - t_{i-1})}                    (4)

S* = argmax_{SS_j}  Σ_{i=1}^{K}  s_i · e^{-α(t_i - t_{i-1})}        (5)
```

Beam search chọn chuỗi `S*` tối đa hóa tổng tích lũy này.

**Tại sao cộng (additive) thay vì nhân (multiplicative) ở giai đoạn này — rationale
quan trọng, dễ làm sai nếu implement cẩu thả:**

```
Multiplicative (SAI ở bước này):  Π_i (s_i · λ_i)
  → quá nhạy cảm với 1 transition điểm thấp — 1 bước tệ phá hỏng cả chuỗi

Additive (ĐÚNG ở bước exploration):  Σ_i (s_i · λ_i)
  → cho phép vài liên kết yếu mà không làm sập toàn chuỗi
  → phản ánh thực tế: hầu hết sự kiện khớp tốt, thỉnh thoảng 1 transition mơ hồ
    không nên loại toàn bộ kết quả
```

### 2.5 Final Reranking with BLIP-2-Based Validation — công thức (6), (7)

Sau khi beam search chọn được `S*`, chuỗi này đi qua **giai đoạn validation thứ 2**,
lần này DÙNG PHÉP NHÂN (multiplicative) — khác hẳn bước exploration ở §2.4:

```
S_i^(final) = s_i · λ_i · b_i                                       (6)

SS^(final) = Σ_{i=1}^{K}  s_i · λ_i · b_i                            (7)

  s_i  = semantic relevance (similarity gốc)
  λ_i  = temporal coherence (decay theo công thức (3))
  b_i  = fine-grained alignment — BLIP-2 ITM validation score
```

**Tại sao đổi sang nhân ở bước validation (đây là điểm dễ nhầm — 2 giai đoạn dùng
2 phép toán khác nhau CÓ CHỦ ĐÍCH, không phải nghịch lý):**

Phép nhân đóng vai trò **gating mechanism** — buộc điểm cuối cao chỉ khi CẢ BA tiêu
chí đồng thời thỏa mãn: khớp ngữ nghĩa (s_i), hợp lý về thời gian (λ_i), và khớp
chi tiết hình ảnh-văn bản đã qua xác thực BLIP-2 (b_i). Nếu bất kỳ thành phần nào
thấp, điểm tổng bị dìm xuống — đây là ràng buộc chất lượng nghiêm ngặt (strict
multi-faceted quality constraint), **giảm false positive** — trường hợp frame khớp
semantic "coi hợp lý" nhưng thực chất lệch hình ảnh sẽ bị lọc ở bước này.

**Tóm tắt kiến trúc 2-giai đoạn (quan trọng khi implement, đừng gộp làm 1):**

```
Giai đoạn 1 (exploration, trong beam search):  ADDITIVE  — công thức (4)/(5)
Giai đoạn 2 (validation, sau khi chọn S*):      MULTIPLICATIVE — công thức (6)/(7)
```

### 2.6 Điểm tích hợp gợi ý vào code hiện tại

`src/retrieval/vqa.py::trake_search()` hiện gọi `find_segments()` (frame-walk,
`tolerance` tĩnh) rồi build event list tuyến tính. Để nâng cấp theo paper này:

1. Giữ nguyên `find_segments()` làm bước sinh candidate frame/sự kiện (vai trò
   tương đương "M candidate frame per event" trong paper).
2. Thay bước gộp/score cuối bằng: beam search giữ top-B chuỗi (công thức 4/5, additive)
   → validation cuối bằng công thức 6/7 (multiplicative), tái dùng `b_i` từ
   `modules/reranker/blip2_itm.py` đã có sẵn (chỉ chưa gắn vào TRAKE ranking).
3. Thêm 2 hyperparameter mới vào `PipelineConfig`: `temporal_decay_alpha` (gợi ý
   khởi điểm 0.01 theo paper) và `beam_width` (gợi ý khởi điểm 8) — **không** hard-code,
   vì cần tune theo tốc độ diễn biến sự kiện của dataset AIC thực tế, và phải log
   vào `runs/<exp>/config.json` để tái lập được.

---

## 3. SRRF — Score-Reflected Reciprocal Rank Fusion

📍 Đối chiếu code: `src/retrieval/fusion.py` — **đã implement**, khớp với mô tả
paper (kích hoạt khi có ≥2 embedding model).

```
RRF chuẩn (chỉ dùng rank, bỏ qua độ lớn similarity):
  RRF(d) = Σ_m  1 / (k + rank_m(d))

SRRF (giữ nguyên similarity score gốc):
  SRRF(d) = Σ_m  s_m(d) / (k + rank_m(d))
```

Paper dùng SRRF để fusion kết quả từ SigLIP và BEiT-3 (2 dual-encoder branch) trước
khi đưa vào BLIP-2 rerank. Lý do quan trọng hơn RRF thuần: 2 candidate cùng rank 1
nhưng similarity 0.95 vs 0.55 không nên coi là ngang nhau về độ tin cậy — SRRF phản
ánh đúng chênh lệch đó thay vì chỉ nhìn vị trí xếp hạng.

---

## 4. Adaptive Score Fusion

📍 Đối chiếu code: **chưa implement** ở dạng multi-modal (visual/OCR/ASR) — code
hiện tại chỉ có single-modality (visual) nên chưa cần bước fusion liên-modal này.
Đây là kỹ thuật khác SRRF: SRRF fusion **trong cùng 1 modality** (2 visual embedder),
còn Adaptive Score Fusion dưới đây fusion **giữa các modality khác nhau** (visual,
OCR, ASR — mỗi modality thang điểm khác hẳn nhau).

### 4.1 Hai vấn đề cần giải quyết

1. Mỗi modality có thang điểm khác nhau — similarity score của visual (Qdrant
   cosine) không so sánh trực tiếp được với relevance score của Elasticsearch
   (BM25-based).
2. Tầm quan trọng của mỗi modality phụ thuộc vào chính nội dung query — trọng số cố
   định (fixed weighting) không hiệu quả cho mọi trường hợp.

### 4.2 Min-max normalization — công thức (1)

```
s_norm_m(f) = (s_m(f) - min(s_m)) / (max(s_m) - min(s_m) + ε)
```

Đưa điểm số về cùng khoảng chung trong khi giữ nguyên thứ hạng nội-modality
(intra-modality ranking). `ε` tránh chia cho 0 khi mọi score trong modality bằng
nhau.

### 4.3 Fusion theo trọng số agent dự đoán — công thức (2)

```
S(f) = Σ_{m ∈ {vis, ocr, asr}}  w_m · s_norm_m(f)
```

`w_m` là trọng số của modality `m`, **do agent (GPT-4o) dự đoán động theo ngữ nghĩa
của từng query cụ thể** — không phải hằng số cấu hình tĩnh.

### 4.4 Query Decomposition Agent — cách sinh w_m

```
Query → GPT-4o phân tích, với mỗi modality:
  1. Phát hiện cue đặc trưng cho modality đó trong câu hỏi
  2. Đánh giá mức độ phân biệt (distinctiveness) của cue đó
     trong việc xác định đúng frame
  3. Gán trọng số phù hợp
  4. Giải thích ngắn gọn lý do (để dễ debug/audit)
```

Heuristic phân loại modality (prompt-driven, không cần model routing riêng huấn
luyện):

| Modality | Bắt cụm từ nào trong query | Ví dụ |
|---|---|---|
| **KIS (Visual Concepts)** | Yếu tố **nhìn thấy được** | hành động, cảnh, vật thể, màu sắc |
| **OCR (On-Screen Text)** | Yếu tố **đọc được** trên màn hình | banner, số, tên áo cầu thủ |
| **ASR (Spoken Keywords)** | Yếu tố **nghe được** | từ khóa lời nói, lời bài hát (KHÔNG tính động từ hành động chung chung như "hát", "nói") |

**Ví dụ từ paper:** query "Cristiano Ronaldo scoring a goal" → agent gán trọng số
cao cho KIS (hành động ghi bàn là yếu tố thị giác), trọng số vừa cho OCR (tên cầu
thủ trên áo), trọng số thấp cho ASR (bình luận thường generic, không đặc trưng).

**3 lợi ích so với cách tiếp cận truyền thống (paper nêu rõ, đáng giữ lại khi
justify thiết kế):**

- **No exhaustive search**: không mù quáng query mọi modality như late-fusion
  truyền thống → giảm chi phí tính toán.
- **No routing model training**: không cần dữ liệu gán nhãn hay pipeline huấn luyện
  riêng cho việc routing.
- **No complex joint fusion**: tránh chi phí align embedding đa phương thức phức
  tạp như kiến trúc joint-fusion (VD ViLBERT-style).

### 4.5 Điểm tích hợp gợi ý vào code hiện tại

Không nên implement ngay — cần đợi ít nhất OCR/ASR producer hoạt động (index được
text vào Elasticsearch) trước, vì Adaptive Score Fusion vô nghĩa nếu chỉ có 1
modality. Khi cả 3 modality sẵn sàng, điểm tích hợp tự nhiên là mở rộng
`src/retrieval/query_processor.py::LlmQueryProcessor` — nó đã có sẵn cơ chế gọi
LLM để enrich query — thêm bước phân rã modality + gán trọng số ở đây, rồi
implement công thức (1)/(2) như 1 hàm fusion mới song song với `fusion.py` (SRRF)
hiện có (2 tầng fusion khác mục đích: SRRF cho intra-modality, adaptive fusion cho
inter-modality).

---

## 5. Agent-Guided Query Expansion and Decomposition

📍 Đối chiếu code: `src/retrieval/query_processor.py::LlmQueryProcessor` **đã có**
dịch + enrich 1 câu (dùng Gemini, không phải GPT-4o) — nhưng **chưa sinh nhiều biến
thể song song** như mô tả dưới đây.

### 5.1 Query Expansion — sinh N=4 biến thể

```
Query gốc → GPT-4o sinh N=4 biến thể (mặc định), MỌI biến thể đều bằng tiếng Anh

2 quy tắc bắt buộc:
  1. Direct translation bắt buộc: biến thể ĐẦU TIÊN luôn là bản dịch tiếng Anh
     trực tiếp, giữ nguyên ý nghĩa gốc (tiếng Anh cho hiệu suất embedding tốt hơn)
  2. Giữ nguyên ý nghĩa gốc: các biến thể khác có thể đổi góc nhìn, bối cảnh,
     hoặc phong cách mô tả — nhưng KHÔNG được thêm object/action không có trong
     query gốc
```

**Ví dụ minh họa từ paper (Figure 3):** query tiếng Việt "cô gái nấu ăn" →

| Biến thể | Nội dung |
|---|---|
| Direct translation | "girl cooking" |
| Synonym variation | "young woman preparing food" |
| Location emphasis | "woman cooking at stove indoors" |
| Role-based variation | "female chef working in home kitchen" |

Sau khi sinh 4 biến thể, **tất cả đều được embed bằng SigLIP và BEiT-3, rerank và
search song song** — rồi fusion lại (không nêu rõ cơ chế fusion 4 biến thể trong
đoạn text, có thể dùng SRRF hoặc max-pooling theo candidate).

**Cơ sở lý thuyết (paper trích dẫn):** sinh nhiều biến thể câu truy vấn bằng LLM
giúp giảm nhiễu từ caption không hoàn hảo và khớp tốt hơn với đánh giá của con
người (theo nghiên cứu GQE — Generalized Query Expansion, và MQVR — Multi-Query
Video Retrieval).

### 5.2 Điểm tích hợp gợi ý vào code hiện tại

`query_processor.py::LlmQueryProcessor` hiện xử lý 1 câu duy nhất (dịch + enrich).
Nâng cấp theo paper: đổi từ "1 câu → 1 câu enriched" thành "1 câu → N=4 câu enriched
song song", rồi mỗi câu chạy qua `retrieval/search.py` độc lập, fusion N kết quả
bằng SRRF (`fusion.py` đã có sẵn, chỉ cần coi N biến thể như N "model" thay vì N
embedder). Cần đánh đổi latency (N lần search thay vì 1) trong ngân sách 5 phút/
query — cân nhắc chạy song song (async/thread pool), không tuần tự.

---

## 6. Modality Routing — chi tiết bổ sung

Đã tóm tắt công thức ở §4.4; điểm bổ sung quan trọng: routing **hoàn toàn
prompt-driven** (fully prompt-driven), heuristic theo modality nhúng thẳng vào
prompt — không cần training riêng, không cần dữ liệu gán nhãn. Đây là lựa chọn kiến
trúc có chủ đích để né 3 nhược điểm nêu ở §4.4 cuối cùng (exhaustive search, routing
model training, complex joint fusion).

---

## 7. Kết quả thực nghiệm (tham khảo khi thiết lập hyperparameter mặc định)

### 7.1 Hyperparameter cố định dùng trong mọi thí nghiệm của paper

| Hyperparameter | Giá trị |
|---|---|
| Số biến thể Query Expansion (N) | 4 |
| Top-K candidate sau first-stage retrieval (BEiT-3 + SigLIP) | 100 |
| Beam width (temporal search) | 8 |
| Temporal decay coefficient (α) | 0.01 |

### 7.2 Điểm số theo vòng thi

| Round | Score | Max | % |
|---|---|---|---|
| Round 1 | 19.8 | 23 | 86.1% |
| Round 2 | 26.6 | 30 | 88.6% |
| Round 3 | 30 | 35 | **85.7%** (điểm tuyệt đối trong hạng mục) |
| **Tổng** | **76.4** | **88** | **86.8%** |

### 7.3 Phân bố câu hỏi theo loại tác vụ

| Round | KIS | VQA | TRAKE |
|---|---|---|---|
| 1 | 17 | 3 | 3 |
| 2 | 26 | 2 | 2 |
| 3 | 29 | 4 | 2 |
| **Tổng** | **72** | **9** | **7** |

KIS chiếm áp đảo số lượng câu hỏi (72/88 ≈ 82%) — nhấn mạnh tầm quan trọng của việc
tối ưu pipeline visual retrieval cơ bản (dual-embedding + SRRF + BLIP-2 rerank) hơn
là các nhánh phức tạp (TRAKE temporal, VQA agent) vốn chiếm tỷ trọng nhỏ hơn nhiều
trong điểm số thực tế.

### 7.4 Kết quả định tính (qualitative) — đáng tham khảo khi viết test case

- **Hiệu quả của cascaded rerank:** hệ baseline "Without Rerank" (chỉ dual-encoder)
  thất bại — Ground Truth (ảnh 1 con chim xanh) bị vùi trong candidate không liên
  quan (người mặc áo đỏ, hoa hồng). Sau khi thêm BLIP-2 rerank ("With Rerank"),
  Ground Truth được đẩy lên vị trí top-1. Minh chứng: rerank là bước bắt buộc,
  không phải tùy chọn, để đạt precision chấp nhận được.
- **Hiệu quả của multimodal fusion:** hệ visual-only thất bại trước các cảnh giống
  nhau về hình ảnh (nền đỏ, trẻ em, cầm biển hiệu) nhưng khác nội dung chữ trên
  biển. Hệ đầy đủ (có Agent fusion) gán `w_ocr ≈ 0.7` cao hơn `w_vis ≈ 0.4`, ưu tiên
  đọc đúng nội dung chữ trên biển ("Program: Financial Support...") → tìm đúng
  cảnh. Minh chứng: OCR có thể quan trọng hơn visual similarity khi câu hỏi phụ
  thuộc vào text hiển thị.
- **Hiệu quả của temporal coherence:** ví dụ chuỗi 3 sự kiện dây chuyền lắp ráp xe
  hơi ("khung xe được lắp bởi cánh tay robot" → "công nhân lắp đặt, xoay tay cầm" →
  "công nhân lắp cửa xe") — chuỗi đúng có thời lượng ngắn 5.1s, đạt điểm
  0.9234; chuỗi sai (dài 34.1s, rời rạc) bị điểm thấp hơn nhờ λ-decay phạt gap lớn.
  Minh chứng trực tiếp cho tác dụng của công thức (3)/(6)/(7).

---

## 8. So sánh với Paper 1 (LLandMark)

| Hạng mục | Paper 1 (LLandMark) | Paper 2 (Cascaded System, file này) |
|---|---|---|
| Shot Detection | TransNetV2 | TransNetV2 |
| Visual Embedding | CLIP ConvNeXt-XXL | BEiT-3 + SigLIP (dual, named vectors) |
| Reranking | Weighted Fusion Agent (nhiều modality) | BLIP-2 ITM (cross-encoder, single-stage rerank) |
| OCR | PaddleOCR + Gemini 2.5 Flash correction | Gemini 2.0 Flash trực tiếp |
| ASR | WhisperX (word-level alignment) | Whisper Large-v3 |
| Vector DB | Milvus | Qdrant |
| Text Search | Elasticsearch | Elasticsearch |
| Metadata store | MongoDB (đánh giá: redundant) | Qdrant payload (gọn hơn) |
| Object Detection | YOLOv9-e | Không có |
| Query Expansion | Không nêu rõ | GPT-4o, N=4 biến thể, 2 quy tắc chặt (§5.1) |
| Modality Routing | Landmark Knowledge Agent (kích hoạt có điều kiện) | GPT-4o phân rã + trọng số động mọi query (§4.4) |
| Temporal Search | Min-score intersection (weakest-link) | Beam search + λ-decay, additive/multiplicative 2 giai đoạn (§2) |
| Điểm mạnh | Landmark reasoning (image-to-image cho địa danh VN) | Temporal coherence có lý thuyết chặt chẽ + ablation rõ ràng |
| Điểm số | **77.40 / 88** | 76.4 / 88 (Round 3 tuyệt đối 30/35) |
| Đánh giá tổng thể | Điểm cao hơn nhưng kiến trúc multi-agent + MongoDB hơi over-engineering | Pipeline tuyến tính sạch hơn, công thức toán học tường minh, dễ tái lập/kiểm chứng hơn |

---

## Cách dùng tài liệu này

Khi được yêu cầu cải thiện temporal search / TRAKE, tra §2 (đặc biệt chú ý 2 công
thức additive (4)/(5) và multiplicative (6)/(7) dùng ở 2 giai đoạn khác nhau — dễ
nhầm nếu đọc lướt). Khi cần fusion đa modality (sau khi OCR/ASR đã có), tra §4.
Khi cần mở rộng query xử lý tiếng Việt, tra §5. Luôn kiểm tra trạng thái
Implemented/Not ở đầu mỗi mục — phần lớn nội dung file này (beam search, λ-decay,
adaptive score fusion, multi-variant query expansion) **chưa có trong code hiện
tại**; chỉ SRRF (§3) và cascade dual-embedding + BLIP-2 rerank (§1) là đã hoạt
động.
