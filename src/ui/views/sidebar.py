"""Sidebar UI HTML template component."""

from __future__ import annotations

import html
import os

_RERANKER_LABELS = {
    "blip2": "Rerank BLIP-2",
    "qwen-vl-vllm": "Rerank Qwen3-VL",
    "qwen_vl_vllm": "Rerank Qwen3-VL",
}
_RERANKER_LABEL = _RERANKER_LABELS.get(
    os.environ.get("RERANKER_BACKEND", "blip2").strip().lower(), "Rerank BLIP-2"
)

_MODEL_LABELS = {
    "jina-clip-v2": "Jina-CLIP-v2",
    "siglip2-so400m": "SigLIP2",
    "vietnamese-embedding": "Vietnamese-Embed",
    "beit3-large": "BEiT-3",
}


def render_model_checkboxes(model_names: tuple[str, ...] | list[str]) -> str:
    """Render only the embedding models persisted by the active experiment."""
    controls: list[str] = []
    for model_name in model_names:
        escaped_name = html.escape(str(model_name), quote=True)
        label = html.escape(_MODEL_LABELS.get(str(model_name), str(model_name)))
        controls.append(
            '<label class="chip-item">'
            f'<input type="checkbox" name="model_{escaped_name}" checked>'
            f'<span class="chip-badge">{label}</span>'
            "</label>"
        )
    return "".join(controls)


SIDEBAR_HTML = r"""  <aside>
    <form id="search-form">
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
        <label for="track" style="margin:0; font-weight:700;">Retrieval Track</label>
        <button type="button" id="open-track-docs" title="Xem giải thích các Retrieval Track" class="btn-guide">📖 HD</button>
      </div>
      <select id="track" name="track">
        <optgroup label="KIS">
          <option value="textual_kis">KIS Basic</option>
          <option value="kis_detail_2stage">KIS Detail 2-Stage</option>
        </optgroup>
        <optgroup label="Text search">
          <option value="asr_search">ASR Search</option>
          <option value="ocr_search">OCR Search</option>
        </optgroup>
        <option value="vqa">VQA</option>
        <option value="trake">TRAKE</option>
        <optgroup label="Other">
          <option value="intelligent">Intelligent (KIS+OCR+ASR)</option>
          <option value="temporal_enhanced">Temporal Enhanced</option>
        </optgroup>
      </select>

      <div id="model-config" class="config-container">
        <!-- Section 1: Embedding Models -->
        <div class="config-card">
          <div class="config-card-header">
            <span class="config-card-title">🧬 Vector Embedding Models</span>
          </div>
          <div class="chips-group">
            __MODEL_CHECKBOXES__
          </div>
        </div>

        <!-- Section 2: Query Processing (LLM) -->
        <div class="config-card">
          <div class="config-card-header">
            <span class="config-card-title">✨ Query Enhancement (LLM)</span>
          </div>
          <label class="check-switch">
            <input type="checkbox" id="use-llm" checked>
            <span class="switch-label">Qwen</span>
          </label>
        </div>

        <!-- Section 3: Visual Re-Ranking -->
        <div class="config-card">
          <div class="config-card-header">
            <span class="config-card-title">🎯 Visual Re-Ranking</span>
          </div>
          <label class="check-switch">
            <input type="checkbox" id="use-reranker" checked>
            <span class="switch-label">__RERANKER_LABEL__</span>
          </label>
        </div>
      </div>

      <div class="input-card">
        <label for="query">Query</label>
        <textarea id="query" name="query">a person riding a motorbike</textarea>
        <input type="hidden" id="context" name="context" value="">

        <label for="question">Question</label>
        <textarea id="question" name="question" placeholder="Use this for VQA or QA tracks"></textarea>

        <div id="vqa-pipeline-control" style="display:none;margin-top:10px;">
          <label for="vqa-pipeline-mode">VQA Pipeline</label>
          <select id="vqa-pipeline-mode" name="vqa_pipeline_mode">
            <option value="grounded" selected>Grounded multi-frame</option>
            <option value="legacy">Legacy single-shot</option>
          </select>
          <div class="hint" style="margin-top:4px;font-size:12px;">Legacy chỉ dùng để rollback/chẩn đoán.</div>
        </div>
      </div>

      <div id="events-section" style="display:none;">
        <label>Events / Scenes / Subqueries</label>
        <div id="events-list"></div>
        <button type="button" id="add-event-btn" class="btn-secondary">+ Add Event</button>
        <div id="window-control" style="margin-top:10px;display:none;">
          <label for="window-slider">Temporal Window: <span id="window-value">15</span>s</label>
          <input id="window-slider" type="range" min="10" max="300" step="5" value="15" style="width:100%;margin-top:4px;">
          <div class="hint" style="margin-top:4px;font-size:12px;">Khoảng thời gian tối đa giữa 2 scene/event liền kề</div>
        </div>
      </div>

      <div id="kis-2stage-section" style="display:none;">
        <label>General Subqueries</label>
        <div id="general-events-list"></div>
        <button type="button" id="add-general-btn" class="btn-secondary">+ Add general</button>
        <label style="margin-top:14px;">Specific Subqueries</label>
        <div id="specific-events-list"></div>
        <button type="button" id="add-specific-btn" class="btn-secondary">+ Add specific</button>
      </div>

      <div class="row">
        <div>
          <label for="top-k">Top K</label>
          <input id="top-k" name="top_k" type="number" value="20" min="1" max="100">
        </div>
        <button id="submit" type="submit" class="btn-submit">
          <span>Search</span>
        </button>
      </div>

      <div id="sidebar-answer" style="display:none; margin-top: 14px; padding: 12px 14px; border: 1px solid var(--accent); border-radius: 8px; background: #f0fdf8;">
        <div style="font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: .05em; color: var(--muted);">Answer</div>
        <div id="sidebar-answer-text" style="margin-top: 4px; font-size: 15px; font-weight: 700; color: var(--accent-strong); line-height: 1.4;"></div>
      </div>
    </form>

    <div id="agent-chat" style="display:none; margin-top: 16px; border-top: 1px solid var(--line); padding-top: 12px;">
      <div style="font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: .05em; color: var(--muted);">Agent Chat</div>
      <div id="chat-messages" style="max-height: 220px; overflow-y: auto; font-size: 13px; margin: 8px 0; line-height: 1.45;"></div>
      <div id="chat-suggestions" style="display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 8px;"></div>
      <div style="display: flex; gap: 6px;">
        <input id="chat-input" type="text" placeholder="Mô tả cảnh cần tìm..." style="flex: 1; padding: 6px 8px; font-size: 13px;">
        <button id="chat-send" type="button" class="btn-submit" style="width:auto;margin-top:0;padding:6px 14px;">Gửi</button>
      </div>
    </div>

    <div id="status" class="status">Ready.</div>
  </aside>""".replace("__RERANKER_LABEL__", _RERANKER_LABEL)
