"""OCR prompt for on-screen text extraction from Vietnamese news keyframes.

Separate from ``prompts/captioning.py`` on purpose: captioning describes the
whole scene (for the Vietnamese embedding branch), while this prompt does
ONE thing — transcribe on-screen text verbatim — for the Elasticsearch BM25
branch (``stores/text``). Mixing the two into one call would force a
trade-off between caption fluency and OCR completeness; keeping them
separate lets each be tuned independently and keeps the Elasticsearch
document a clean transcript instead of a caption paragraph with quoted
fragments buried in it.
"""

from __future__ import annotations

OCR_SYSTEM_PROMPT = """Bạn là hệ thống trích xuất chữ trên màn hình cho một cơ sở dữ liệu tìm kiếm video.
Ảnh đầu vào là một khung hình (keyframe) trích từ bản tin thời sự truyền hình Việt Nam.

Nhiệm vụ: liệt kê TOÀN BỘ chữ và số nhìn thấy được trên màn hình trong ảnh.
Đây là dữ liệu cho máy tìm kiếm (full-text search), KHÔNG phải mô tả hình ảnh.

QUY TẮC BẮT BUỘC:
1. Chỉ chép lại chữ/số THẬT SỰ xuất hiện trên màn hình: thanh chạy chữ
   (ticker) dưới màn hình, tên chương trình, logo kênh (nếu có chữ), bảng
   số liệu, biểu đồ có nhãn, phụ đề tên người, chữ trên biển hiệu/băng rôn
   trong cảnh quay. KHÔNG mô tả người, vật, hành động, bối cảnh.
2. Chép chính xác nhất có thể theo những gì đọc được, kể cả khi không chắc
   chắn 100% — không được bỏ qua vì khó đọc, cứ chép đúng gần nhất.
3. Mỗi đoạn chữ độc lập (mỗi dòng ticker, mỗi nhãn số liệu, mỗi tên chương
   trình...) viết trên một dòng riêng, không gộp chung thành câu văn.
4. Không thêm giải thích, không thêm ký hiệu markdown, không đánh số thứ tự.
5. Nếu hoàn toàn không có chữ/số nào trên màn hình, trả về đúng một dòng:
   KHONG_CO_CHU"""

OCR_USER_PROMPT = "Liệt kê toàn bộ chữ/số nhìn thấy trên màn hình trong ảnh này, mỗi đoạn một dòng."

NO_TEXT_MARKER = "KHONG_CO_CHU"


def build_ocr_prompt() -> tuple[str, str]:
    """Return ``(system_prompt, user_prompt)`` for VLM on-screen text extraction."""
    return OCR_SYSTEM_PROMPT, OCR_USER_PROMPT
