"""Vietnamese keyframe captioning prompts, for embedding — not for human reading.

Captions from this template are encoded by a Vietnamese sentence embedder
(``AITeamVN/Vietnamese_Embedding_v2``) and stored as a third named vector
alongside SigLIP2/BEiT-3, specifically to cover what CLIP-style visual
embedders miss: on-screen text, named entities, and relative position.

The structure is deliberately rigid (fixed section order, controlled
vocabulary, "everyday language" ban on flowery wording) rather than free-form,
because these captions feed a similarity search over hundreds of thousands of
keyframes, not a human reader — lexical drift between near-duplicate scenes
(e.g. "áo đỏ" vs "áo màu đỏ" vs "đỏ tươi" for the same shirt) spreads out what
should be a tight embedding cluster and degrades ranking. See
.claude/references/ for the research behind this design.

The dataset is Vietnamese TV news footage (bản tin thời sự), so the template
adds a news-specific scene-type classification and instructs the model to
transcribe on-screen ticker/banner text verbatim in quotes — that text is
usually the single most query-able detail in a news keyframe (station name,
program title, statistics, place names) and is exactly what visual embedders
are worst at.
"""

from __future__ import annotations

CAPTION_SYSTEM_PROMPT = """Bạn là hệ thống tự động mô tả ảnh cho một cơ sở dữ liệu tìm kiếm video.
Ảnh đầu vào là một khung hình (keyframe) trích từ bản tin thời sự truyền hình Việt Nam.

Nhiệm vụ: viết MỘT đoạn mô tả tiếng Việt, ngắn gọn, theo đúng cấu trúc
cố định bên dưới. Đoạn mô tả này sẽ được máy tính so khớp với câu hỏi
của người dùng, KHÔNG phải để người đọc thưởng thức văn chương.

QUY TẮC BẮT BUỘC:
1. Chỉ dùng tiếng Việt. Không chèn tiếng Anh trừ khi đó là chữ/số
   xuất hiện trực tiếp trên màn hình (biển hiệu, phụ đề, số liệu, tên
   riêng nước ngoài).
2. Văn phong đời thường, viết như tường thuật lại những gì nhìn thấy.
   Cấm dùng từ hoa mỹ, ẩn dụ, ví von, đánh giá cảm xúc (KHÔNG viết
   "khung cảnh nên thơ", "ánh sáng lung linh huyền ảo", "tuyệt đẹp",
   "ấn tượng"...). Thay bằng mô tả trực tiếp: "trời có nắng",
   "đèn trường quay sáng", "phòng có ánh sáng vàng".
3. Dùng từ vựng phổ biến nhất, nhất quán. Với cùng 1 khái niệm luôn
   dùng đúng 1 từ, không đổi qua từ đồng nghĩa. Ví dụ cố định:
   - vị trí: "bên trái" / "bên phải" / "ở giữa" / "phía trên" / "phía dưới"
     (không dùng "góc trái", "mé phải", "khu vực trung tâm"...)
   - người: "người đàn ông" / "người phụ nữ" / "bé trai" / "bé gái" /
     "nhóm người" (không dùng "quý ông", "chàng trai", "cô gái" trừ khi
     rõ ràng là người trẻ)
   - màu sắc: dùng đúng 1 trong các màu cơ bản (đỏ, cam, vàng, xanh lá,
     xanh dương, tím, hồng, nâu, đen, trắng, xám), không dùng màu văn hoa
     (đỏ tươi, xanh ngọc, be, pastel...)
4. Không suy đoán, không thêm thông tin không thấy trong ảnh. Nếu
   không chắc, mô tả chung chung thay vì đoán chi tiết.
5. Không lặp lại chủ ngữ "trong ảnh", "hình ảnh cho thấy", "có thể thấy"
   nhiều lần. Vào thẳng nội dung mô tả.
6. Độ dài: 3-5 câu, khoảng 50-100 từ. Không viết dài hơn."""

CAPTION_USER_TEMPLATE = """Mô tả khung hình này theo đúng 5 phần sau, viết liền thành đoạn văn
(không cần đánh số/gạch đầu dòng), theo ĐÚNG thứ tự:

1. LOẠI CẢNH: xác định đây là loại cảnh nào trong bản tin thời sự —
   "cảnh trường quay" (có người dẫn chương trình ngồi/đứng trong studio),
   "cảnh phóng sự hiện trường" (quay ngoài trời hoặc tại địa điểm cụ thể,
   không phải studio), "cảnh đồ họa số liệu" (biểu đồ, bảng số liệu, bản đồ,
   hình ảnh minh họa dựng bằng đồ họa máy tính), hoặc "cảnh khác" nếu không
   rõ thuộc loại nào ở trên.

2. BỐI CẢNH: nơi chốn cụ thể hơn (trong nhà/ngoài trời, loại không gian:
   đường phố, văn phòng, cánh đồng, chợ, trường quay...), thời điểm nếu
   nhận biết được (ban ngày/ban đêm), thời tiết nếu ngoài trời.

3. ĐỐI TƯỢNG CHÍNH: người/vật/phương tiện chính trong khung hình, kèm đặc
   điểm nhận dạng trực quan (trang phục và màu sắc, loại phương tiện,
   giới tính/độ tuổi ước lượng nếu là người, có phải người dẫn chương
   trình/phóng viên nếu nhận ra qua bối cảnh trường quay hay không).

4. HÀNH ĐỘNG: đối tượng chính đang làm gì, tư thế gì (đứng, ngồi, đi,
   cầm vật gì, tương tác với ai/cái gì, đang nói/phỏng vấn).

5. CHỮ TRÊN MÀN HÌNH: mọi chữ, số, thanh chạy chữ (ticker) dưới màn hình,
   tên chương trình, logo kênh, bảng số liệu, phụ đề tên người đang phỏng
   vấn xuất hiện trong ảnh — CHÉP LẠI CHÍNH XÁC nội dung trong dấu ngoặc
   kép, kể cả khi không chắc chắn 100% (chép đúng gần nhất theo những gì
   đọc được, không được bỏ qua vì không chắc). Nếu ảnh có nhiều dòng chữ
   khác nhau, liệt kê từng đoạn trong ngoặc kép riêng. Nếu hoàn toàn không
   có chữ/số nào trong ảnh, bỏ qua phần này, không viết "không có chữ"
   hay câu tương đương.

Viết thành 3-5 câu liền mạch, không dùng gạch đầu dòng, không nhắc lại
tên 5 phần trên trong câu trả lời."""

_FEW_SHOT_EXAMPLES = [
    {
        "caption": (
            "Cảnh trường quay, trong studio, đèn trường quay sáng. Một người phụ nữ mặc áo "
            "vest xanh dương đang ngồi ở bàn dẫn chương trình, có vẻ là người dẫn chương "
            "trình. Người phụ nữ tay cầm giấy, nhìn thẳng vào máy quay và đang nói. Phía "
            'dưới màn hình có logo kênh ở góc trên bên trái và dòng chữ "THỜI SỰ 18H30", '
            'thanh chạy chữ phía dưới ghi "Miền Trung khắc phục hậu quả sau bão số 6".'
        )
    },
    {
        "caption": (
            "Cảnh phóng sự hiện trường, ngoài trời, trên một cánh đồng, trời có nắng. Một "
            "nhóm người mặc áo mưa màu xanh lá đang đứng xem xét ruộng lúa bị ngập nước. "
            "Một người đàn ông cúi người chỉ tay xuống đám lúa. Phía dưới màn hình có dòng "
            'chữ phụ đề tên "Ông Nguyễn Văn A - Nông dân xã Hòa Bình", góc dưới bên phải '
            'hiển thị số liệu "Thiệt hại ước tính 40 ha lúa".'
        )
    },
]


def build_caption_prompt() -> tuple[str, str]:
    """Return ``(system_prompt, user_prompt)`` for VLM keyframe captioning.

    The pair is designed for a chat-completions call: send ``system_prompt``
    as the system message, then a user message containing ``user_prompt``
    plus the keyframe image. Callers that support few-shot messages should
    also inject :data:`_FEW_SHOT_EXAMPLES` as prior assistant turns — see
    ``modules/captioning/vllm.py`` for the concrete wiring — since few-shot
    anchoring of style/vocabulary is more reliable than prose instructions
    alone across large batch runs.
    """
    return CAPTION_SYSTEM_PROMPT, CAPTION_USER_TEMPLATE


def few_shot_messages() -> list[dict[str, str]]:
    """Return few-shot assistant example captions to anchor style and vocabulary.

    These are real (fabricated but representative) news-broadcast captions
    matching the exact structure/vocabulary rules in the system prompt.
    Prepending them as prior turns keeps the VLM's output consistent across
    hundreds of thousands of calls better than describing style in prose
    alone (in-context examples anchor lexical choices more reliably than
    abstract instructions).
    """
    return [{"role": "assistant", "content": example["caption"]} for example in _FEW_SHOT_EXAMPLES]
