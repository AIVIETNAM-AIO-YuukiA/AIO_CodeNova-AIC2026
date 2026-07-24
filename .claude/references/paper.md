# Research Sources — 3-Model Embedding + OCR/ASR Implementation

> Nguồn tham khảo dùng khi implement nhánh Vietnamese caption embedding
> (SigLIP2 + BEiT-3 + `Vietnamese_Embedding_v2`), OCR/ASR → Elasticsearch, và
> `Vietnamese_Reranker`. Khác với [`paper1_llandmark.md`](paper1_llandmark.md) /
> [`paper2_cascaded_system.md`](paper2_cascaded_system.md) /
> [`paper3_vortex.md`](paper3_vortex.md) (phân tích kiến trúc tổng thể 3 hệ
> thống đối chiếu), file này tập hợp các nguồn **research có mục tiêu hẹp**
> dùng trực tiếp để quyết định thiết kế prompt/chunking/generation-params khi
> code — mỗi mục ghi rõ quyết định nào trong code bắt nguồn từ nguồn nào.

---

## 1. Structured captioning cho embedding (không phải cho người đọc)

**Quyết định trong code:** [`src/prompts/captioning.py`](../../src/prompts/captioning.py) —
prompt caption ép cấu trúc 5 khối cố định (loại cảnh → bối cảnh → đối tượng →
hành động → chữ trên màn hình), văn phong đời thường, từ vựng nhất quán, kèm
2 few-shot example.

**Vấn đề gốc:** caption tốt cho người đọc (đa dạng từ vựng, tự nhiên) khác
hẳn caption tối ưu cho dense retrieval (nhất quán, ít nhiễu). Với hàng trăm
nghìn keyframe cùng gọi 1 VLM, nếu để model tự do văn phong, 2 frame có nội
dung giống hệt nhau (VD "áo đỏ" vs "áo màu đỏ" vs "đỏ tươi") sẽ cho ra vector
embedding hơi lệch nhau — pha loãng cụm ngữ nghĩa lẽ ra phải sát nhau, làm
giảm precision khi retrieval.

**Nguồn:**
- [PixelProse — From Pixels to Prose (arXiv 2406.10328)](https://arxiv.org/abs/2406.10328) —
  prompt ép format cố định, mở đầu bằng cụm cố định, tách riêng OCR trong ngoặc kép.
- [Structured Captions Improve Prompt Adherence — Re-LAION-Caption 19M (arXiv 2507.05300)](https://arxiv.org/html/2507.05300v1) —
  template 4 khối cố định theo đúng thứ tự, báo cáo cải thiện "prompt adherence".
- [DOCCI (Google Research)](https://google.github.io/docci/) — nhắm trực
  tiếp vào 3 điểm CLIP-style embedding yếu: spatial relation, counting, text rendering.
- [Foundations of LLM Knowledge Materialization (arXiv 2510.06780)](https://arxiv.org/pdf/2510.06780) —
  xác nhận lexical variance là rủi ro thật: nội dung ngữ nghĩa ổn định qua
  nhiều lần chạy nhưng lexical similarity chỉ ở mức trung bình.
- [Benchmarking LLM Volatility (arXiv 2311.15180)](https://arxiv.org/pdf/2311.15180) —
  structured/CoT prompt hoạt động như regularizer giữ output ổn định, quan
  trọng hơn chỉ hạ temperature.
- [Exploring Diverse In-Context Configurations for Image Captioning (arXiv 2305.14800)](https://arxiv.org/pdf/2305.14800) —
  cơ sở cho việc dùng few-shot example để neo văn phong/độ dài thay vì chỉ mô
  tả bằng lời trong system prompt.

---

## 2. AIC 2025/2026 — kỹ thuật hiểu ảnh và rerank của các đội khác

**Quyết định trong code:** xác nhận hướng đi (VLM caption + Vietnamese
sentence-embedding song song visual embedding) là pattern đã có tiền lệ ở
AIC 2025 nhưng **chưa ai công bố xử lý đặc thù tin tức** (ticker/banner/logo/
phân loại cảnh) — đây là lý do khối "LOẠI CẢNH" và "CHỮ TRÊN MÀN HÌNH" trong
prompt caption được thiết kế chi tiết hơn mức trung bình các đội khác.

**Phát hiện quan trọng nhất:** tính đến 7/2026, **AIC 2026 chưa có kết quả**
(đăng ký 15/5–15/6/2026, sơ khảo dự kiến tháng 8/2026, chung kết 9/2026) —
theo [trang chính thức](https://aichallenge.hochiminhcity.gov.vn/en/home).
Mọi so sánh kỹ thuật dưới đây dùng **AIC 2025** (mùa gần nhất, cùng ban tổ
chức, cùng dạng dataset tin tức truyền hình Việt Nam) làm nguồn thay thế gần
nhất, không phải dữ liệu AIC 2026 trực tiếp.

**Các hệ thống AIC 2025 khác đã khảo sát** (ngoài paper1/2/3 đã có file riêng):

| Đội/Paper | Nguồn | Điểm | Kỹ thuật đáng chú ý |
|---|---|---|---|
| U-CESE (Nomial) | [arXiv 2605.23274](https://arxiv.org/pdf/2605.23274) | — | Unified instruction prompt cho LVLM vừa reasoning vừa caption |
| AIO_Owlgorithms (QUEST+DANTE) | [arXiv 2512.13169](https://arxiv.org/html/2512.13169) | — | Gemini OCR trích on-screen text thành key-value pairs |
| AIO_Trinh (MADTempo) | [arXiv 2512.12929](https://arxiv.org/html/2512.12929) | 75.4 | **Vintern-1B-v3_5** (OCR tiếng Việt chuyên biệt) chạy song song Qwen2.5-VL |
| MERVIN | [arXiv 2605.16120](https://arxiv.org/html/2605.16120) | 79/88 | `dangvantuan/vietnamese-embedding` cho nhánh text tiếng Việt |

**Khoảng trống xác nhận:** không đội nào trong số các paper khảo sát mô tả
xử lý riêng cho ticker chạy chữ, tên chương trình, logo kênh, hay schema
JSON phân loại cảnh tin tức (trường quay/hiện trường/đồ họa số liệu) — OCR
luôn được xử lý như 1 khối chung, không tách theo loại banner. Prompt 5-khối
hiện tại (mục 1) là hướng khác biệt thực sự so với các hệ thống đã công bố.

Xem thêm bảng công thức multiplicative gating rerank (`S_final = s_i · λ_i · b_i`)
đã phân tích chi tiết trong [`paper2_cascaded_system.md`](paper2_cascaded_system.md)
(cùng 1 paper, arXiv 2512.12935 = 2605.23274's sibling analysis).

**Nguồn đầy đủ:**
- [Home - AI Challenge TP.HCM (lịch trình 2026 chính thức)](https://aichallenge.hochiminhcity.gov.vn/en/home)
- [U-CESE: Unified Clip-based Event Search Engine for AI Challenge HCMC 2025](https://arxiv.org/pdf/2605.23274)
- [Integrated Semantic and Temporal Alignment for Interactive Video Retrieval](https://arxiv.org/html/2512.13169)
- [MADTempo: Multi-Event Temporal Video Retrieval with Query Augmentation](https://arxiv.org/html/2512.12929)
- [MERVIN: Multimodal Event Retrieval in Vietnamese News Videos](https://arxiv.org/html/2605.16120)
- [VERGE in VBS 2026 (Zenodo, embargo tới 1/2027 — không lấy được chi tiết)](https://zenodo.org/records/18268841)
- [Video Browser Showdown — Teams & Papers](https://videobrowsershowdown.org/teams/)
- [TRECVID 2026 Guidelines (chưa chốt track)](https://www-nlpir.nist.gov/projects/tv2026/contacts.html)

---

## 3. ASR chunking + khử trùng lặp overlap

**Quyết định trong code:** [`src/modules/asr/gipformer.py`](../../src/modules/asr/gipformer.py) —
chunk audio dài thành cửa sổ cố định 30s, overlap 1s, khử trùng lặp bằng
token-level LCS (longest common subsequence) ở ranh giới giữa 2 chunk liền kề.

**Vấn đề gốc:** checkpoint ONNX công bố của `g-group-ai-lab/gipformer-65M-rnnt`
là offline (non-causal) transducer — xác nhận trực tiếp bằng cách thử load
vào `sherpa_onnx.OnlineRecognizer` (thất bại, thiếu metadata streaming-only
`encoder_dims`) — nên **không có streaming thật**. Audio dài (nguyên audio
track 1 video, có thể 20+ phút) làm treo `OfflineRecognizer` nếu đẩy nguyên
khối vào 1 lần (`accept_waveform` timeout > 60s trên input 21 phút, trong khi
input 15s chạy tức thời) — cần chunk trước khi transcribe.

**Nguồn:** [ASR in 2025-2026: A Deep Dive into Speech Recognition Technology Selection](https://ruoqijin.com/blog/asr-deep-dive-2025-2026) —
xác nhận đúng con số đã dùng trong code: chunk 30-40s với 1 giây overlap là
thực hành phổ biến cho "pseudo-streaming" (chunked offline ASR, không phải
true streaming), và khử trùng lặp overlap bằng LCS ở cấp token là kỹ thuật
chuẩn để tránh đếm/index trùng phần audio chung giữa 2 chunk liền kề.

---

## Cách dùng tài liệu này

Mỗi mục ở trên gắn trực tiếp với 1 quyết định code cụ thể — khi cần tune lại
prompt caption, chunk size ASR, hay đánh giá liệu 1 kỹ thuật mới có đáng thử
không, quay lại đúng mục tương ứng để xem lý do gốc trước khi thay đổi. Nếu
research thêm nguồn mới cho cùng chủ đề, thêm vào đúng mục hiện có thay vì
tạo file mới, trừ khi đó là 1 mảng hoàn toàn khác (VD: nếu sau này research
riêng cho `Vietnamese_Reranker` fusion strategy, có thể tách mục 4 riêng).
