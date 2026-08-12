"""Frame modal HTML template component."""

from __future__ import annotations

MODAL_HTML = r"""      <!-- Modal -->
      <div id="frame-modal">
        <div class="modal-box" id="modal-box">
          <div class="modal-top">
            <span id="modal-title">Loading...</span>
            <span id="modal-time" class="time-badge"></span>
            <button class="close-x" id="modal-close-x">&times;</button>
          </div>
          <div class="modal-mid">
            <button class="modal-nav" id="modal-prev" title="Shot trước (←)">&#9664;</button>
            <div class="img-area">
              <img id="modal-img" src="" alt="Frame preview">
            </div>
            <button class="modal-nav" id="modal-next" title="Shot tiếp (→)">&#9654;</button>
          </div>
          <div class="modal-strip" id="modal-strip"></div>
          <div class="modal-bot">
            <span id="modal-footer"></span>
            <div class="actions">
              <button class="btn-setthumb" id="btn-setthumb">Làm thumbnail</button>
              <button class="btn-revert-modal" id="btn-revert-modal" style="display:none">Revert</button>
              <button class="btn-close-modal" id="btn-close-modal">Đóng (Esc)</button>
            </div>
          </div>
        </div>
      </div>

      <!-- Track Docs Modal -->
      <div id="track-docs-modal" style="display:none; position:fixed; inset:0; z-index:1000; background:rgba(0,0,0,0.65); align-items:center; justify-content:center;">
        <div style="background:var(--panel); color:var(--text); width:90vw; max-width:680px; max-height:85vh; border-radius:12px; overflow-y:auto; padding:24px; box-shadow:0 10px 30px rgba(0,0,0,0.3); position:relative;">
          <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid var(--line); padding-bottom:12px; margin-bottom:16px;">
            <h3 style="margin:0; font-size:18px; color:var(--accent); display:flex; align-items:center; gap:8px;">📖 Hướng dẫn Retrieval Tracks</h3>
            <button type="button" id="close-track-docs" style="background:none; border:none; font-size:22px; cursor:pointer; color:var(--muted); width:auto; padding:0; margin:0;">&times;</button>
          </div>
          <div style="display:flex; flex-direction:column; gap:12px; font-size:13px; line-height:1.5;">
            <div style="padding:10px 12px; background:#f8fafc; border-left:4px solid var(--accent); border-radius:6px;">
              <strong style="color:var(--accent-strong);">textual_kis (KIS Basic)</strong><br>
              Tìm kiếm video/frame bằng mô tả văn bản trực tiếp qua mô hình embedding (BEiT-3, SigLIP2, Vietnamese-Embedding) + Qdrant.
            </div>
            <div style="padding:10px 12px; background:#f8fafc; border-left:4px solid var(--accent); border-radius:6px;">
              <strong style="color:var(--accent-strong);">kis_detail_2stage (KIS Detail 2-Stage)</strong><br>
              Tìm kiếm 2 giai đoạn: Giai đoạn 1 lọc khung cảnh chung (coarse), Giai đoạn 2 lọc chi tiết đặc thù (fine).
            </div>
            <div style="padding:10px 12px; background:#f8fafc; border-left:4px solid var(--accent); border-radius:6px;">
              <strong style="color:var(--accent-strong);">asr_search (ASR Search)</strong><br>
              Tìm kiếm văn bản giọng nói (Speech-to-Text) bằng Elasticsearch BM25.
            </div>
            <div style="padding:10px 12px; background:#f8fafc; border-left:4px solid var(--accent); border-radius:6px;">
              <strong style="color:var(--accent-strong);">ocr_search (OCR Search)</strong><br>
              Tìm kiếm chữ xuất hiện trên màn hình (On-screen text) bằng Elasticsearch BM25.
            </div>
            <div style="padding:10px 12px; background:#f8fafc; border-left:4px solid var(--accent); border-radius:6px;">
              <strong style="color:var(--accent-strong);">intelligent (Intelligent KIS+OCR+ASR)</strong><br>
              Dùng LLM đọc query duy nhất, tự động phân tách thành (Prompt hình ảnh + Keywords OCR + Keywords ASR + Trọng số từng loại), sau đó tìm kiếm độc lập và dung hợp bằng Weighted SRRF.
            </div>
            <div style="padding:10px 12px; background:#f8fafc; border-left:4px solid var(--accent); border-radius:6px;">
              <strong style="color:var(--accent-strong);">vqa (VQA)</strong><br>
              Pipeline 3 bước: Tìm kiếm khung hình &rarr; Gom nhóm Temporal shot &rarr; ReAct Agent (LLM/VLM) trả lời câu hỏi chi tiết.
            </div>
            <div style="padding:10px 12px; background:#f8fafc; border-left:4px solid var(--accent); border-radius:6px;">
              <strong style="color:var(--accent-strong);">trake (TRAKE)</strong><br>
              Tìm kiếm chuỗi nhiều sự kiện diễn ra theo thứ tự thời gian (E1 &rarr; E2 &rarr; E3).
            </div>
            <div style="padding:10px 12px; background:#f8fafc; border-left:4px solid var(--accent); border-radius:6px;">
              <strong style="color:var(--accent-strong);">temporal_enhanced (Temporal Enhanced)</strong><br>
              Nhận 1 câu mô tả tự nhiên duy nhất, dùng LLM tự động tách thành chuỗi các event (E1, E2,...) rồi chạy qua pipeline TRAKE.
            </div>
          </div>
        </div>
      </div>"""
