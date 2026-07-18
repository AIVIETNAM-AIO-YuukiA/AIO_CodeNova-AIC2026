"""Agent tab UI for AI Co-Pilot (Retrieval Agent) - 2 Column Layout."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote
from uuid import uuid4

from config.settings import Experiment
from core.types import SearchResult
from retrieval.vqa import trake_search, vqa_search


@dataclass
class AgentSessionState:
    """Conversation state kept in memory for one browser session."""
    turns: list[dict[str, str]] = field(default_factory=list)
    pending_route: str = ""


# AI Co-Pilot Styles (2 Column Dashboard)
AGENT_TAB_STYLE = r"""
    /* AI Co-Pilot 2-Column Dashboard Styles */
    .copilot-main {
      display: none; 
      grid-template-columns: 480px 1fr; 
      height: calc(100vh - 61px);
      background: #f1f5f9;
    }
    .copilot-col {
      display: flex; 
      flex-direction: column; 
      border-right: 1px solid var(--line); 
      background: var(--panel); 
      overflow: hidden;
    }
    .copilot-col.last { 
      border-right: none; 
    }
    .copilot-header {
      padding: 14px 18px; 
      border-bottom: 1px solid var(--line); 
      background: #f8fafc;
      font-weight: 750; 
      font-size: 15px; 
      display: flex; 
      justify-content: space-between; 
      align-items: center;
    }
    
    .copilot-chat-history {
      flex: 1; 
      padding: 18px; 
      overflow-y: auto; 
      display: flex; 
      flex-direction: column; 
      gap: 14px; 
      background: #fafafa;
    }
    .copilot-msg { 
      max-width: 90%; 
      padding: 12px 14px; 
      border-radius: 8px; 
      font-size: 13px; 
      line-height: 1.5; 
      border: 1px solid var(--line); 
      position: relative;
    }
    .copilot-msg.user { 
      align-self: flex-end; 
      background: #ecfdf5; 
      border-color: #a7f3d0; 
      color: var(--accent-strong); 
    }
    .copilot-msg.model { 
      align-self: flex-start; 
      background: #fff; 
      border-color: var(--line); 
      box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    }
    .copilot-msg .meta { 
      font-size: 10px; 
      color: var(--muted); 
      margin-bottom: 5px; 
      text-transform: uppercase; 
      font-weight: 700; 
      letter-spacing: .05em; 
    }
    .copilot-msg pre { 
      white-space: pre-wrap; 
      font-family: inherit; 
      margin: 0; 
    }
    .copilot-msg .agent-steps-summary {
      margin-top: 10px;
      padding-top: 8px;
      border-top: 1px dashed var(--line);
      font-size: 11px;
      color: var(--muted);
      line-height: 1.4;
    }
    .copilot-msg .agent-step-item {
      margin-bottom: 4px;
      display: flex;
      align-items: flex-start;
      gap: 6px;
    }
    
    .copilot-chat-input-area { 
      padding: 14px 18px; 
      border-top: 1px solid var(--line); 
      background: var(--panel); 
    }
    .copilot-chat-input-area textarea { 
      min-height: 72px; 
      margin-bottom: 8px; 
      resize: none; 
      font-size: 13px; 
    }
    
    .copilot-plan-box { 
      padding: 14px 18px; 
      background: #f8fafc; 
      border-top: 1px solid var(--line); 
      max-height: 180px; 
      overflow-y: auto; 
    }
    .copilot-plan-title { 
      font-size: 11px; 
      font-weight: 750; 
      color: var(--muted); 
      margin-bottom: 8px; 
      text-transform: uppercase; 
      letter-spacing: .05em; 
    }
    .copilot-plan-item { 
      display: flex; 
      align-items: start; 
      gap: 8px; 
      font-size: 12px; 
      margin-bottom: 6px; 
      line-height: 1.4; 
    }
    .copilot-plan-item .status-icon { 
      font-weight: bold; 
    }
    .copilot-plan-item.completed { 
      color: var(--accent-strong); 
    }
    .copilot-plan-item.running { 
      color: #0284c7; 
      font-weight: bold; 
    }
    
    .copilot-results-grid { 
      flex: 1; 
      padding: 18px; 
      overflow-y: auto; 
      display: flex; 
      flex-direction: column; 
      gap: 14px; 
      background: #f8fafc; 
    }
    .copilot-results-grid .card { 
      background: #fff; 
    }
    
    /* Loading typing spinner effect */
    .copilot-typing {
      display: inline-flex;
      align-items: center;
      gap: 3px;
      margin-top: 4px;
    }
    .copilot-typing span {
      width: 6px;
      height: 6px;
      background: var(--muted);
      border-radius: 50%;
      animation: copilot-bounce 1.4s infinite ease-in-out both;
    }
    .copilot-typing span:nth-child(1) { animation-delay: -0.32s; }
    .copilot-typing span:nth-child(2) { animation-delay: -0.16s; }
    @keyframes copilot-bounce {
      0%, 80%, 100% { transform: scale(0); }
      40% { transform: scale(1.0); }
    }
"""

# Giao diện HTML của AI Co-Pilot (2 Cột)
AGENT_TAB_HTML = r"""
<main id="copilot-main" class="copilot-main">
  <!-- Cột 1: Chat & Plan -->
  <div class="copilot-col" style="grid-column: 1;">
    <div class="copilot-header">
      <span>AI Assistant</span>
      <span class="pill" style="background:#e0f2fe; color:#0369a1;">Gemini 2.5</span>
    </div>
    <div class="copilot-chat-history" id="copilot-chat-history">
      <div class="copilot-msg model">
        <div class="meta">AI Co-Pilot</div>
        <pre>Chào bạn! Tôi là AI Co-Pilot định tuyến và tối ưu hóa tìm kiếm. Tôi có thể giúp bạn tự động chọn lựa pipeline tìm kiếm (KIS, TRAKE, VQA), phân tích chuỗi sự kiện và tự động cải thiện truy vấn để mang lại kết quả tốt nhất.
Hãy nhập yêu cầu của bạn.".</pre>
      </div>
    </div>
    <div class="copilot-plan-box" id="copilot-plan-box" style="display:none;">
      <div class="copilot-plan-title">Kế hoạch thực hiện</div>
      <div id="copilot-plan-list"></div>
    </div>
    <div class="copilot-chat-input-area">
      <textarea id="copilot-chat-input" placeholder="Nhập yêu cầu... (Ctrl+Enter để gửi)"></textarea>
      <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 4px;">
        <button id="copilot-btn-clear" class="agent-secondary" style="margin:0; padding:6px 12px; background: transparent; border: 1px solid #cbd5e1; color: #64748b; font-size: 12px; cursor: pointer; border-radius: 6px;">🗑️ Xóa chat</button>
        <button id="copilot-btn-send" style="margin:0; padding:6px 20px; background: #0284c7; color: #fff; border: none; font-weight: 600; cursor: pointer; border-radius: 6px;">Gửi 🚀</button>
      </div>
    </div>
  </div>

  <!-- Cột 2: Preview Results -->
  <div class="copilot-col last" style="grid-column: 2;">
    <div class="copilot-header">
      <span>Preview Results</span>
    </div>
    <div class="copilot-results-grid" id="copilot-preview-results">
      <div style="color:var(--muted); font-size:13px; text-align:center; padding-top:40px;">
        Kết quả search test hoặc previews sẽ hiện ở đây.
      </div>
    </div>
  </div>
</main>
"""

# Logic JS cho AI Co-Pilot (Bọc trong script tag)
AGENT_TAB_SCRIPT = r"""
<script>
(function() {
  const eid = id => document.getElementById(id);
  function esc(v) {
    return String(v).replaceAll("&","&amp;").replaceAll("<","&lt;")
      .replaceAll(">","&gt;").replaceAll('"',"&quot;").replaceAll("'","&#039;");
  }

  // ─── AI Co-Pilot Tab Switching ────────────────────────────────────────────────
  const btnTabManual = eid("btn-tab-manual");
  const btnTabCopilot = eid("btn-tab-copilot");
  const manualMain = eid("manual-main");
  const copilotMain = eid("copilot-main");

  if (btnTabManual && btnTabCopilot && manualMain && copilotMain) {
    btnTabManual.addEventListener("click", () => {
      btnTabManual.classList.add("active");
      btnTabCopilot.classList.remove("active");
      manualMain.style.display = "grid";
      copilotMain.style.display = "none";
    });

    btnTabCopilot.addEventListener("click", () => {
      btnTabCopilot.classList.add("active");
      btnTabManual.classList.remove("active");
      manualMain.style.display = "none";
      copilotMain.style.display = "grid";
    });
  }

  // ─── AI Co-Pilot Chat Logic ──────────────────────────────────────────────────
  const copilotChatHistory = eid("copilot-chat-history");
  const copilotChatInput = eid("copilot-chat-input");
  const copilotBtnSend = eid("copilot-btn-send");
  const copilotBtnClear = eid("copilot-btn-clear");
  const copilotPlanBox = eid("copilot-plan-box");
  const copilotPlanList = eid("copilot-plan-list");
  const copilotPreviewResults = eid("copilot-preview-results");

  let copilotSessionId = "copilot-" + Date.now();
  let copilotBusy = false;

  function appendCopilotMsg(role, text) {
    if (!copilotChatHistory) return null;
    const div = document.createElement("div");
    div.className = `copilot-msg ${role}`;
    div.innerHTML = `
      <div class="meta">${role === 'user' ? 'Bạn' : 'AI Co-Pilot'}</div>
      <pre>${esc(text)}</pre>
    `;
    copilotChatHistory.appendChild(div);
    copilotChatHistory.scrollTop = copilotChatHistory.scrollHeight;
    return div;
  }

  function appendCopilotWaiting() {
    if (!copilotChatHistory) return null;
    const div = document.createElement("div");
    div.className = "copilot-msg model waiting";
    div.innerHTML = `
      <div class="meta">AI Co-Pilot</div>
      <div class="copilot-typing">
        <span></span><span></span><span></span>
        <span style="font-size:12px; color:var(--muted); margin-left:6px;">Agent đang suy nghĩ...</span>
      </div>
    `;
    copilotChatHistory.appendChild(div);
    copilotChatHistory.scrollTop = copilotChatHistory.scrollHeight;
    return div;
  }

  function formatStepsSummary(trace) {
    if (!trace || !trace.length) return "";
    const items = trace.map(t => {
      let icon = "⏳";
      let label = t.step;
      if (t.step.startsWith("thought_")) {
        icon = "💡";
        label = "Suy nghĩ: " + t.detail;
      } else if (t.step.startsWith("call_tool_")) {
        icon = "🛠️";
        label = "Gọi Tool: " + t.step.replace("call_tool_", "") + " (" + JSON.stringify(t.detail) + ")";
      } else if (t.step.startsWith("observation_")) {
        icon = "✅";
        label = "Hoàn thành: " + t.step.replace("observation_", "");
      }
      return `<div class="agent-step-item">
        <span>${icon}</span>
        <span>${esc(label)}</span>
      </div>`;
    }).join("");
    return `<div class="agent-steps-summary">
      <div style="font-weight:700; margin-bottom:4px; text-transform:uppercase; font-size:9px; letter-spacing:.05em;">Các bước đã thực hiện:</div>
      ${items}
    </div>`;
  }

  async function submitCopilotMsg() {
    const text = copilotChatInput ? copilotChatInput.value.trim() : "";
    if (!text || copilotBusy) return;

    copilotBusy = true;
    if (copilotChatInput) copilotChatInput.value = "";
    if (copilotBtnSend) copilotBtnSend.disabled = true;
    if (copilotBtnClear) copilotBtnClear.disabled = true;

    // Append user input
    appendCopilotMsg("user", text);

    // Append waiting placeholder
    const waitingDiv = appendCopilotWaiting();

    if (copilotPlanBox) copilotPlanBox.style.display = "block";
    if (copilotPlanList) copilotPlanList.innerHTML = '<div style="color:var(--muted); font-size:12px;">Agent đang lập kế hoạch...</div>';

    try {
      const res = await fetch("/api/task-agent", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: copilotSessionId, message: text })
      });
      const data = await res.json();
      if (!res.ok || data.error) throw new Error(data.error || "Request failed");

      // Render plan checklist on the bottom panel
      renderCopilotPlan(data.trace || []);

      // Replace waiting placeholder with final agent answer + inline steps summary
      if (waitingDiv) {
        const stepsHtml = formatStepsSummary(data.trace || []);
        waitingDiv.className = "copilot-msg model";
        waitingDiv.innerHTML = `
          <div class="meta">AI Co-Pilot</div>
          <pre>${esc(data.answer || "Đã hoàn thành.")}</pre>
          ${stepsHtml}
        `;
      }

      // Check if Agent returned search results to preview
      const lastTraceWithSearch = data.trace ? [...data.trace].reverse().find(t => t.step.includes("observation_search_retrieval") || t.step.includes("observation_search_vector_kis") || t.step.includes("observation_search_trake") || t.step.includes("observation_search_vqa")) : null;
      if (lastTraceWithSearch) {
        try {
          const jsonMatch = lastTraceWithSearch.detail.match(/\{.*\}/s);
          if (jsonMatch) {
            const searchData = JSON.parse(jsonMatch[0]);
            renderCopilotPreviews(searchData.top_results || []);
          }
        } catch (e) {
          console.error("Failed to parse search results from trace", e);
        }
      }

    } catch (err) {
      if (waitingDiv) {
        waitingDiv.className = "copilot-msg model";
        waitingDiv.innerHTML = `
          <div class="meta">AI Co-Pilot</div>
          <pre style="color:#ef4444;">Lỗi: ${esc(err.message)}</pre>
        `;
      }
    } finally {
      copilotBusy = false;
      if (copilotBtnSend) copilotBtnSend.disabled = false;
      if (copilotBtnClear) copilotBtnClear.disabled = false;
      if (copilotChatHistory) copilotChatHistory.scrollTop = copilotChatHistory.scrollHeight;
    }
  }

  function renderCopilotPlan(trace) {
    if (!copilotPlanBox || !copilotPlanList) return;
    if (!trace || !trace.length) {
      copilotPlanBox.style.display = "none";
      return;
    }
    copilotPlanBox.style.display = "block";

    copilotPlanList.innerHTML = trace.map(t => {
      let icon = "⏳";
      let label = t.step;
      let cls = "";
      if (t.step.startsWith("thought_")) {
        icon = "💡";
        label = "Suy nghĩ: " + t.detail;
      } else if (t.step.startsWith("call_tool_")) {
        icon = "🛠️";
        label = "Gọi Tool " + t.step.replace("call_tool_", "") + ": " + JSON.stringify(t.detail);
        cls = "running";
      } else if (t.step.startsWith("observation_")) {
        icon = "✅";
        label = "Hoàn thành: " + t.step.replace("observation_", "");
        cls = "completed";
      }
      return `<div class="copilot-plan-item ${cls}">
        <span class="status-icon">${icon}</span>
        <span style="overflow-wrap: anywhere;">${esc(label)}</span>
      </div>`;
    }).join("");
  }

  function renderCopilotPreviews(results) {
    if (!copilotPreviewResults) return;
    if (!results || !results.length) {
      copilotPreviewResults.innerHTML = '<div style="color:var(--muted); font-size:13px; text-align:center; padding-top:40px;">Không có kết quả search preview.</div>';
      return;
    }
    
    copilotPreviewResults.innerHTML = results.map(r => {
      const imgUrl = r.frame_path ? `/frame?path=${encodeURIComponent(r.frame_path)}` : "";
      return `
        <div class="card">
          ${imgUrl ? `<img src="${esc(imgUrl)}" onclick="showImageModal('${esc(r.video_id)}', '${esc(r.frame_path)}', ${r.frame_index || 0})" title="Click to zoom">` : ""}
          <div class="meta">
            <strong>${esc(r.video_name || r.video_id)}</strong>
            <code>Score: ${r.score.toFixed(4)} · Frame: ${r.frame_index} · Time: ${r.timestamp_sec ? r.timestamp_sec.toFixed(2)+'s' : ''}</code>
          </div>
        </div>
      `;
    }).join("");
  }

  // Helper to open modal from preview click
  window.showImageModal = function(videoId, framePath, frameIndex) {
    fetch(`/api/video-shots?video_id=${encodeURIComponent(videoId)}`)
      .then(res => res.json())
      .then(data => {
        if (typeof modal !== 'undefined') {
          modal.open = true;
          modal.videoId = videoId;
          modal.shots = data.shots || [];
          modal.chainIdx = 0;
          modal.eventIdx = null;

          let foundSi = 0, foundFi = 0, found = false;
          for (let si = 0; si < modal.shots.length && !found; si++) {
            const frames = modal.shots[si].frames || [];
            for (let fi = 0; fi < frames.length; fi++) {
              if (frames[fi].frame_index === frameIndex) {
                foundSi = si; foundFi = fi; found = true; break;
              }
            }
          }
          modal.shotIdx = foundSi;
          modal.frameIdx = foundFi;
          
          const frameModal = eid("frame-modal");
          if (frameModal) frameModal.style.display = "flex";
          if (typeof renderModal === 'function') renderModal();
        }
      });
  };

  if (copilotBtnSend) {
    copilotBtnSend.addEventListener("click", submitCopilotMsg);
  }
  if (copilotChatInput) {
    copilotChatInput.addEventListener("keydown", e => {
      if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        submitCopilotMsg();
      }
    });
  }

  if (copilotBtnClear && copilotChatHistory) {
    copilotBtnClear.addEventListener("click", () => {
      copilotChatHistory.innerHTML = `
        <div class="copilot-msg model">
          <div class="meta">AI Co-Pilot</div>
          <pre>Chào bạn! Tôi là AI Co-Pilot định tuyến và tối ưu hóa tìm kiếm. Tôi có thể giúp bạn tự động chọn lựa pipeline tìm kiếm (KIS, TRAKE, VQA), phân tích chuỗi sự kiện và tự động cải thiện truy vấn để mang lại kết quả tốt nhất.
Hãy nhập yêu cầu của bạn, ví dụ: "Tìm xe máy đỏ vượt lên ở giây thứ 10" hoặc "Có ai đang dắt xe đạp trong video không?".</pre>
        </div>
      `;
      copilotSessionId = "copilot-" + Date.now();
      if (copilotPlanBox) copilotPlanBox.style.display = "none";
    });
  }
})();
</script>
"""


def build_agent_payload(
    experiment: Experiment,
    retriever,
    payload: dict[str, object],
    sessions: dict[str, AgentSessionState],
    default_top_k: int,
) -> dict[str, object]:
    """Legacy routing placeholder for backward compatibility."""
    return {}