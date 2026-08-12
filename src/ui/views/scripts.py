"""Client-side JavaScript logic for CodeNova UI."""

from __future__ import annotations

APP_JS = r"""
    function eid(id) { return document.getElementById(id); }
    function esc(s) { return escapeHtml(s == null ? "" : String(s)); }
    function escapeHtml(s) {
      return String(s == null ? "" : s).replace(/[&<>"']/g, c => (
        { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
      ));
    }
    function formatTime(seconds) {
      const s = Math.max(0, Number(seconds) || 0);
      const mm = Math.floor(s / 60);
      const ss = Math.floor(s % 60);
      return `${mm}:${String(ss).padStart(2, "0")}`;
    }
    function fmtTime(seconds) { return formatTime(seconds); }
    function formatNumber(n) { return Number(n).toLocaleString(); }

    const form = document.getElementById("search-form");
    const statusEl = document.getElementById("status");
    const resultsEl = document.getElementById("results");
    const answerBox = document.getElementById("answer-box");
    const eventsEl = document.getElementById("events-box");
    const pipelineBox = document.getElementById("pipeline-box");
    const submitEl = document.getElementById("submit");
    const sidebarAnswer = document.getElementById("sidebar-answer");
    const sidebarAnswerText = document.getElementById("sidebar-answer-text");
    const EVENTS_LIST = eid("events-list");
    const EVENTS_SEC = eid("events-section");

    // ─── TRAKE event inputs ───────────────────────────────────────────────────────
    function eventCount() { return EVENTS_LIST.children.length; }
    function addSubInput(btn) {
      const wrapper = btn.closest(".event-wrapper");
      if (!wrapper) return;
      const container = wrapper.querySelector(".sub-inputs");
      const row = document.createElement("div");
      row.style.cssText = "display:flex;gap:6px;margin:0 0 4px 28px;align-items:start;";
      row.innerHTML = `
      <textarea class="sub-detail-input" style="flex:1;min-height:36px;font-size:12px;" placeholder="Sub-detail..."></textarea>
      <button type="button" style="width:auto;padding:2px 8px;margin-top:0;font-size:12px;background:transparent;color:var(--muted);border-color:var(--line);" onclick="this.parentElement.remove()" title="Remove">✕</button>`;
      container.appendChild(row);
    }
    function addEvent(value) {
      const idx = eventCount() + 1;
      const wrapper = document.createElement("div");
      wrapper.className = "event-wrapper";
      wrapper.style.cssText = "margin-bottom:8px;";
      wrapper.innerHTML = `
      <div style="display:flex;gap:6px;align-items:start;">
        <textarea class="event-input" style="flex:1;min-height:56px;" placeholder="Event ${idx} description">${esc(value||"")}</textarea>
        <button type="button" class="add-sub-btn" style="width:auto;padding:6px 10px;margin-top:0;font-size:16px;background:transparent;color:var(--accent);border-color:var(--accent);" onclick="addSubInput(this)" title="Add sub-detail">+</button>
        <button type="button" style="width:auto;padding:6px 10px;margin-top:0;font-size:14px;background:transparent;color:var(--muted);border-color:var(--line);" onclick="this.closest('.event-wrapper').remove()" title="Remove">✕</button>
      </div>
      <div class="sub-inputs"></div>`;
    EVENTS_LIST.appendChild(wrapper);
  }
  eid("add-event-btn").addEventListener("click", () => addEvent(""));
  eid("window-slider").addEventListener("input", () => {
    eid("window-value").textContent = eid("window-slider").value;
  });
  function getEvents() {
    return Array.from(EVENTS_LIST.querySelectorAll(".event-wrapper"))
      .map(w => {
        const textarea = w.querySelector(".event-input");
        const text = textarea ? textarea.value.trim() : "";
        if (!text) return null;
        const subInputs = w.querySelectorAll(".sub-detail-input");
        const sub_details = Array.from(subInputs).map(s => s.value.trim()).filter(Boolean);
        return { text, sub_details };
      }).filter(Boolean);
  }

  // ─── KIS Detail 2-Stage: general / specific event lists ──────────────────────
  const GEN_LIST = eid("general-events-list");
  const SPEC_LIST = eid("specific-events-list");

  function addGeneralEvent(value) {
    const wrapper = document.createElement("div");
    wrapper.className = "event-wrapper";
    wrapper.style.cssText = "margin-bottom:6px;";
    wrapper.innerHTML = `
      <div style="display:flex;gap:6px;align-items:start;">
        <textarea class="event-input" style="flex:1;min-height:48px;" placeholder="General subquery...">${esc(value||"")}</textarea>
        <button type="button" style="width:auto;padding:6px 10px;margin-top:0;font-size:14px;background:transparent;color:var(--muted);border-color:var(--line);" onclick="this.closest('.event-wrapper').remove()" title="Remove">✕</button>
      </div>`;
    GEN_LIST.appendChild(wrapper);
  }

  function addSpecificEvent(value) {
    const wrapper = document.createElement("div");
    wrapper.className = "event-wrapper";
    wrapper.style.cssText = "margin-bottom:6px;";
    wrapper.innerHTML = `
      <div style="display:flex;gap:6px;align-items:start;">
        <textarea class="event-input" style="flex:1;min-height:48px;" placeholder="Specific subquery...">${esc(value||"")}</textarea>
        <button type="button" style="width:auto;padding:6px 10px;margin-top:0;font-size:14px;background:transparent;color:var(--muted);border-color:var(--line);" onclick="this.closest('.event-wrapper').remove()" title="Remove">✕</button>
      </div>`;
    SPEC_LIST.appendChild(wrapper);
  }

  eid("add-general-btn").addEventListener("click", () => addGeneralEvent(""));
  eid("add-specific-btn").addEventListener("click", () => addSpecificEvent(""));

  function get2StageEvents() {
    const general = Array.from(GEN_LIST.querySelectorAll(".event-wrapper"))
      .map(w => w.querySelector(".event-input")?.value.trim()).filter(Boolean);
    const specific = Array.from(SPEC_LIST.querySelectorAll(".event-wrapper"))
      .map(w => w.querySelector(".event-input")?.value.trim()).filter(Boolean);
    return { general, specific };
  }

  // ─── Model picker (checkboxes named "model_<embedding-model-name>") ───────────
  function getEnabledModels() {
    return Array.from(document.querySelectorAll('#model-config input[name^="model_"]'))
      .filter(cb => cb.checked)
      .map(cb => cb.name.slice("model_".length));
  }

  // ─── Track selector ───────────────────────────────────────────────────────────
  form.track.addEventListener("change", () => {
    const t = form.track.value;
    const isBasic = t === "textual_kis";
    const is2Stage = t === "kis_detail_2stage";
    const isT = t === "trake";
    const isV = t === "vqa";
    const isTextSearch = t === "asr_search" || t === "ocr_search";
    const isIntelligent = t === "intelligent";
    const isEnhanced = t === "temporal_enhanced";
    const usesEvents = isT;
    // Enhanced temporal derives its own events from one sentence, so it keeps
    // the temporal window control but not the manual event list.
    const showWindow = isT || isEnhanced;
    // Only tracks that actually route through Retriever.search() (i.e. embed
    // the query with one or more configured models) show the model picker.
    const usesEmbedders = isBasic || isT || isV || isIntelligent || isEnhanced;

    eid("query").style.display = usesEvents || is2Stage ? "none" : "";
    form.querySelector("label[for=query]").style.display = usesEvents || is2Stage ? "none" : "";
    eid("question").style.display = isV ? "" : "none";
    form.querySelector("label[for=question]").style.display = isV ? "" : "none";
    eid("top-k").style.display = isT || is2Stage ? "none" : "";
    form.querySelector("label[for=top-k]").style.display = isT || is2Stage ? "none" : "";
    EVENTS_SEC.style.display = usesEvents ? "" : "none";
    eid("window-control").style.display = showWindow ? "" : "none";
    eid("kis-2stage-section").style.display = is2Stage ? "" : "none";
    eid("model-config").style.display = usesEmbedders ? "" : "none";
    if (isT && eventCount()===0) { addEvent("a person riding a motorbike"); addEvent("a person falling off"); }
  });
  form.track.dispatchEvent(new Event("change"));

  // Last submitted events for TRAKE (carries sub-details)
  let lastTrakeInput = null;

  // ─── Search submit ────────────────────────────────────────────────────────────
  form.addEventListener("submit", async e => {
    e.preventDefault();
    submitEl.disabled = true;
    statusEl.className = "status"; statusEl.textContent = "Searching...";
    resultsEl.innerHTML = ""; answerBox.innerHTML = ""; pipelineBox.innerHTML = ""; eventsEl.innerHTML = "";
    sidebarAnswer.style.display = "none";
    const track = form.track.value;
    let endpoint, payload;
    if (track === "trake") {
      const events = getEvents();
      if (events.length < 2) { statusEl.className="status warn"; statusEl.textContent="Need at least 2 events."; submitEl.disabled=false; return; }
      lastTrakeInput = events;
      endpoint = "/api/trake-search";
      payload = {
        events,
        top_k: 300,
        window: Number(eid("window-slider").value),
        enabled_models: getEnabledModels(),
        use_reranker: eid("use-reranker").checked,
        use_llm: eid("use-llm").checked,
      };
    } else if (track === "kis_detail_2stage") {
      const { general, specific } = get2StageEvents();
      if (general.length < 1) { statusEl.className="status warn"; statusEl.textContent="Need at least 1 general subquery."; submitEl.disabled=false; return; }
      if (specific.length < 1) { statusEl.className="status warn"; statusEl.textContent="Need at least 1 specific subquery."; submitEl.disabled=false; return; }
      endpoint = "/api/kis-detail-2stage"; payload = { general, specific };
    } else if (track === "vqa") {
      endpoint = "/api/vqa-search";
      payload = { query: form.query.value, context: form.context.value, question: form.question.value, top_k: Number(form["top_k"].value||20) };
    } else if (track === "asr_search" || track === "ocr_search") {
      endpoint = track === "asr_search" ? "/api/asr-search" : "/api/ocr-search";
      payload = { query: form.query.value, top_k: 300 };
    } else if (track === "intelligent") {
      endpoint = "/api/intelligent-search";
      payload = { query: form.query.value, top_k: Number(form["top_k"].value||20) };
    } else if (track === "temporal_enhanced") {
      endpoint = "/api/enhanced-temporal-search";
      payload = {
        query: form.query.value, context: form.context.value,
        top_k: Number(form["top_k"].value||20), max_events: 5,
      };
    } else {
      endpoint = "/api/search";
      payload = {
        track, query: form.query.value, context: form.context.value, question: form.question.value,
        top_k: 300, enabled_models: getEnabledModels(), use_reranker: eid("use-reranker").checked,
        use_llm: eid("use-llm").checked,
      };
    }

    try {
      const response = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const data = await response.json();
      if (!response.ok || data.error) {
        throw new Error(data.error || "Search failed");
      }

      if (track === "vqa" && data.agent_error) {
        statusEl.className = "status warn";
        statusEl.textContent = "Agent failed (see error below)";
        answerBox.innerHTML = `<div class="answer-box" style="border-color:#e55;background:#fff5f5">
          <div class="label" style="color:#c00">Agent Error</div>
          <div class="answer-text" style="color:#c00;font-size:13px;font-family:monospace">${escapeHtml(data.agent_error)}</div>
        </div>`;
        renderPipeline(data);
        renderResults(data.results || []);
        return;
      }

      if (track === "vqa" && data.answer) {
        statusEl.innerHTML = `<strong>Answer received</strong> via 3-stage pipeline <span class="pill">VQA</span>`;
        renderAnswer(data.answer);
        renderPipeline(data);
        renderResults(data.results || []);
        sidebarAnswer.style.display = "block";
        sidebarAnswerText.textContent = data.answer;
      } else if (track === "trake" || track === "temporal_enhanced") {
        if (data.videos) {
          const chains = data.videos || [];
          const uniqueVideos = new Set(chains.map(v => v.video_id)).size;
          statusEl.innerHTML = `<strong>${chains.length}</strong> chain(s) from <strong>${uniqueVideos}</strong> video(s) match all events <span class="pill">TRAKE</span>`;
          renderTrake(data);
        } else {
          const eventCount = (data.events || []).length;
          const extracted = data.extracted_events || [];
          const label = track === "trake" ? "TRAKE" : "Temporal Enhanced";
          const extractedNote = extracted.length
            ? ` · LLM tách: ${extracted.map(escapeHtml).join(" → ")}`
            : "";
          statusEl.innerHTML = `<strong>${eventCount}</strong> event(s) found <span class="pill">${label}</span>${extractedNote}`;
          renderTrakeEvents(data.events || []);
          renderPipeline(data);
          renderResults(data.results || []);
        }
      } else if (track === "kis_detail_2stage") {
        const results = data.results || [];
        statusEl.innerHTML = `<strong>${results.length}</strong> frames match all details <span class="pill">KIS Detail 2-Stage</span>`;
        renderResults(results);
      } else if (track === "intelligent") {
        const results = data.results || [];
        const a = data.analysis || {};
        const w = a.weights || {};
        const counts = data.component_counts || {};
        const parts = ["kis", "ocr", "asr"]
          .filter(k => (w[k] || 0) > 0)
          .map(k => `${k.toUpperCase()} ${(w[k]).toFixed(2)} (${counts[k] || 0} hit)`);
        statusEl.innerHTML = `<strong>${results.length}</strong> results <span class="pill">Intelligent</span>`
          + (parts.length ? ` · ${escapeHtml(parts.join(" + "))}` : "");
        renderResults(results);
      } else {
        const trackLabels = {
          textual_kis: "Textual KIS", asr_search: "ASR Search", ocr_search: "OCR Search",
        };
        const trackLabel = trackLabels[track] || track;
        statusEl.innerHTML = `<strong>${data.results.length}</strong> results for <span class="pill">${trackLabel}</span>`;
        renderResults(data.results);
      }
    } catch (error) {
      statusEl.className = "status warn";
      statusEl.textContent = error.message;
    } finally {
      submitEl.disabled = false;
    }
  });

    function renderAnswer(answer) {
      answerBox.innerHTML = `
        <div class="answer-box">
          <div class="label">Answer</div>
          <div class="answer-text">${escapeHtml(answer)}</div>
        </div>`;
    }

    function renderPipeline(data) {
      const pipeline = data.pipeline || {};
      const hasAgent = pipeline.agent;
      const stages = hasAgent ? [
        { key: "embed_search", label: "Embedding Search", desc: `Top-${pipeline.embed_search?.top_k} frames retrieved` },
        { key: "temporal_search", label: "Temporal Search", desc: `${pipeline.temporal_search?.segments_found || 0} segments found` },
        { key: "gather_shot", label: "Shot Gather", desc: `${pipeline.gather_shot?.shots_count || 0} valid shots gathered` },
        { key: "shot_validation", label: "Shot Validation", desc: `Score: ${(pipeline.shot_validation?.validation_score || 0).toFixed(4)}` },
        { key: "agent", label: "Agent (Qwen3.5-4B)", desc: `Answer: ${(pipeline.agent?.answer || "N/A").substring(0, 100)}` },
      ] : [
        { key: "embed_search", label: "Embedding Search", desc: `Top-${pipeline.embed_search?.top_k} frames retrieved` },
        { key: "temporal_search", label: "Temporal Search", desc: `${pipeline.temporal_search?.segments_found || 0} segments found` },
        { key: "gather_shot", label: "Shot Gather", desc: `${pipeline.gather_shot?.shots_count || 0} valid shots gathered` },
      ];
      pipelineBox.innerHTML = `
        <button class="pipeline-toggle" onclick="togglePipeline()">Show Pipeline Details</button>
        <div id="pipeline-detail" class="pipeline-detail">
          ${stages.map((s, i) => `
            <div style="margin-bottom: 8px;">
              <strong>Stage ${i + 1}: ${escapeHtml(s.label)}</strong><br>
              ${escapeHtml(s.desc)}
            </div>
          `).join("")}
          ${hasAgent ? `<hr style="margin: 10px 0; border-color: var(--line);"><div><strong>Reasoning:</strong></div><code>${escapeHtml(data.reasoning || "N/A")}</code>` : ""}
        </div>
      `;
    }

  const thumbState = {};
  function trakeKey(videoId, chainIdx, eventIdx) { return videoId + "::" + chainIdx + "::" + eventIdx; }

  function renderTrake(data) {
    const videos = data.videos || [];
    if (!videos.length) {
      resultsEl.innerHTML = `<div class="hint" style="padding:20px;text-align:center;">No video found matching all events.</div>`;
      return;
    }
    resultsEl.innerHTML = videos.map((video, vi) => {
      const cols = Math.min(video.events.length, 5);
      const evHtml = (video.events||[]).map((ev, ei) => {
        const key = trakeKey(video.video_id, vi, ei);
        if (!thumbState[key]) {
          thumbState[key] = {
            originalUrl: ev.image_url||"", originalFrameId: ev.frame_id||"",
            originalTimestamp: ev.timestamp_sec,
            currentUrl: ev.image_url||"", currentTimestamp: ev.timestamp_sec,
            currentFrameId: ev.frame_id||"",
            rank: ev.rank,
            videoId: video.video_id,
            chainIdx: vi,
            eventIdx: ei,
          };
        }
        const st = thumbState[key];
        const url = st.currentUrl;
        const isCustom = url !== st.originalUrl;
        const safeKey = key.replaceAll("::","__");

        return `
          <div class="ev-card${isCustom?" has-custom":""}" id="evcard-${safeKey}">
            <img src="${escapeHtml(url)}" alt="Event ${ei+1}" loading="lazy"
              id="evimg-${safeKey}"
              onclick="openModalFromCard('${safeKey}')">
            <button class="revert-badge" onclick="revertCard('${escapeHtml(video.video_id)}',${vi},${ei})">&#x21a9; Revert</button>
            <div class="ev-info">
              <span id="evinfo-text-${safeKey}"><strong>Event ${ei+1}</strong> &middot; rank ${st.rank} &middot; ${formatTime(st.currentTimestamp)}</span>
              ${isCustom?`<span class="pill" style="font-size:10px">CUSTOM</span>`:""}
            </div>
          </div>`;
      }).join("");
      return `
        <div class="video-block">
          <div class="video-block-header">
            <strong style="font-size:15px">#${vi+1} ${escapeHtml(video.video_name||video.video_id)}</strong>
            <span class="pill">Score: ${video.score} ${video.temporal_order_valid?"✓ temporal":"✗ temporal"}</span>
          </div>
          <div class="event-grid" style="grid-template-columns:repeat(${cols},1fr)">${evHtml}</div>
        </div>`;
    }).join("");
  }

    function renderTrakeEvents(events) {
      if (!events.length) return;
      const html = events.map((ev, i) => `
        <div style="margin-bottom: 16px; padding: 16px; border: 1px solid var(--line); border-radius: 8px; background: var(--panel);">
          <div style="margin-bottom: 8px;"><strong>Event #${i + 1}</strong></div>
          <div style="font-size: 14px; margin-bottom: 4px;">Video: <strong>${escapeHtml(ev.video_name || ev.video_id || "")}</strong></div>
          <div style="font-size: 13px; color: var(--muted); margin-bottom: 4px;">
            Frames: ${ev.frame_count} · Time: ${formatTime(ev.start_timestamp)} - ${formatTime(ev.end_timestamp)}
          </div>
          <div style="font-size: 13px; color: var(--muted); margin-bottom: 12px;">Score: ${(ev.score || 0).toFixed(4)}</div>
          <div style="display: flex; gap: 8px; flex-wrap: wrap;">
            ${(ev.image_urls || []).map(url => `
              <img src="${escapeHtml(url)}" style="width: 100px; height: 56px; object-fit: cover; border-radius: 4px; border: 1px solid var(--line);" loading="lazy">
            `).join("")}
          </div>
        </div>
      `).join("");
      eventsEl.innerHTML = html;
    }

    function togglePipeline() {
      document.getElementById("pipeline-detail").classList.toggle("open");
    }

    function renderResults(results) {
      if (!results || !results.length) {
        resultsEl.innerHTML = `<div class="hint" style="padding:20px;text-align:center;">No matching results found.</div>`;
        return;
      }
      resultsEl.innerHTML = results.map((result, index) => {
        // Sub-scores for 2-stage or multi-query
        const subs = result.sub_scores || result.subquery_scores || {};
        const subKeys = Object.keys(subs);
        const subBadgesHtml = subKeys.length > 0
          ? `<div style="display:flex;gap:4px;flex-wrap:wrap;margin-top:6px;">${subKeys.map(k => `<span class="pill" style="font-size:10px;background:#eef2ff;color:#4338ca;">${escapeHtml(k.replace('sub_',''))}: ${Number(subs[k]).toFixed(3)}</span>`).join('')}</div>`
          : "";

        // Component scores for Intelligent fusion
        const compHtml = (result.kis_score || result.ocr_score || result.asr_score)
          ? `<div style="display:flex;gap:4px;flex-wrap:wrap;margin-top:6px;">
              ${result.kis_score ? `<span class="pill" style="font-size:10px;background:#e0f2fe;color:#0369a1;">KIS ${Number(result.kis_score).toFixed(3)}</span>` : ""}
              ${result.ocr_score ? `<span class="pill" style="font-size:10px;background:#fef3c7;color:#b45309;">OCR ${Number(result.ocr_score).toFixed(3)}</span>` : ""}
              ${result.asr_score ? `<span class="pill" style="font-size:10px;background:#f3e8ff;color:#6b21a8;">ASR ${Number(result.asr_score).toFixed(3)}</span>` : ""}
            </div>`
          : "";

        // Source badge
        const srcBadge = result.source
          ? `<span class="pill" style="font-size:10px;margin-left:6px;">${escapeHtml(result.source.toUpperCase())}</span>`
          : "";

        // Text snippet
        const textContent = result.text || result.matched_text || result.ocr_text || result.asr_text || "";
        const textHtml = textContent
          ? `<div style="margin-top:6px;padding:4px 8px;font-size:12px;background:#f8fafc;border-left:3px solid var(--accent);border-radius:4px;color:var(--text);">${escapeHtml(textContent.slice(0, 160))}</div>`
          : "";

        return `
        <article class="card">
          <img src="${escapeHtml(result.image_url || "")}" alt="Result frame ${index + 1}" loading="lazy"
            onclick="openModal('${escapeHtml(result.image_url || "")}','${escapeHtml(result.video_id || "")}','${escapeHtml(result.frame_id || "")}',null,null)">
          <div class="meta">
            <div><strong>#${index + 1}</strong> score ${Number(result.score).toFixed(4)} ${srcBadge}</div>
            <div>${formatTime(result.timestamp_sec)} · frame ${formatNumber(result.frame_index)}</div>
            <div><strong>${escapeHtml(result.video_name || result.video_id || "")}</strong></div>
            <div>shot ${escapeHtml(result.shot_id || "s0")}</div>
            <code>${escapeHtml(result.frame_id || "")}</code>
            ${subBadgesHtml}
            ${compHtml}
            ${textHtml}
          </div>
        </article>`;
      }).join("");
    }

  function revertCard(videoId, chainIdx, eventIdx) {
    const key = trakeKey(videoId, chainIdx, eventIdx);
    const st = thumbState[key];
    if (!st) return;
    st.currentUrl = st.originalUrl;
    st.currentTimestamp = st.originalTimestamp;
    st.currentFrameId = st.originalFrameId;
    refreshCard(videoId, chainIdx, eventIdx);
    statusEl.innerHTML = `<strong>Reverted</strong> Event ${eventIdx+1} to original <span class="pill">ORIGINAL</span>`;
    // sync modal if it's open on this card
    if (modal.open && modal.videoId===videoId && modal.chainIdx===chainIdx && modal.eventIdx===eventIdx) {
      eid("btn-revert-modal").style.display = "none";
      eid("btn-setthumb").textContent = "Làm thumbnail";
    }
  }

  function refreshCard(videoId, chainIdx, eventIdx) {
    const key = trakeKey(videoId, chainIdx, eventIdx);
    const st = thumbState[key];
    if (!st) return;
    const cardId = "evcard-" + key.replaceAll("::","__");
    const card = eid(cardId);
    if (!card) return;
    const isCustom = st.currentUrl !== st.originalUrl;
    const img = card.querySelector("img");
    if (img) img.src = st.currentUrl;
    // Update the info text (timestamp changes when thumbnail changes)
    const textSpan = eid("evinfo-text-" + key.replaceAll("::","__"));
    if (textSpan) {
      textSpan.innerHTML = `<strong>Event ${eventIdx+1}</strong> &middot; rank ${st.rank} &middot; ${fmtTime(st.currentTimestamp)}`;
    }
    const info = card.querySelector(".ev-info");
    if (info) {
      const existing = info.querySelector(".pill");
      if (isCustom && !existing) {
        const span = document.createElement("span");
        span.className = "pill"; span.style.fontSize = "10px"; span.textContent = "CUSTOM";
        info.appendChild(span);
      } else if (!isCustom && existing) {
        existing.remove();
      }
    }
    if (isCustom) card.classList.add("has-custom");
    else card.classList.remove("has-custom");
  }

  // ─── Modal state ──────────────────────────────────────────────────────────────
  const modal = {
    open: false,
    shots: [],        // [{shot_id, frames:[{frame_id, frame_index, timestamp_sec, image_url}]}]
    shotIdx: 0,
    frameIdx: 0,
    videoId: "",
    chainIdx: null,   // null for non-trake cards
    eventIdx: null,   // null for non-trake cards
  };

  // ─── Open modal from card (reads live thumbState, not stale render values) ───
  function openModalFromCard(safeKey) {
    const key = safeKey.replaceAll('__','::');
    const st = thumbState[key];
    if (!st) return;
    openModal(st.currentUrl, st.videoId, st.currentFrameId, st.eventIdx, null, st.chainIdx);
  }

  // ─── Modal open ───────────────────────────────────────────────────────────────
  function openModal(imgSrc, videoId, frameId, eventIdx, _unused, chainIdx) {
    // Show modal immediately with the clicked image
    modal.open = true;
    modal.videoId = videoId;
    modal.chainIdx = chainIdx != null ? chainIdx : null;
    modal.eventIdx = eventIdx;
    modal.shots = [];
    modal.shotIdx = 0;
    modal.frameIdx = 0;

    const modalEl = eid("frame-modal");
    modalEl.style.display = "flex";

    const imgEl = eid("modal-img");
    imgEl.src = imgSrc;

    eid("modal-title").textContent = "Loading shots…";
    eid("modal-time").textContent = "";
    eid("modal-footer").textContent = "";
    eid("modal-strip").innerHTML = "";
    eid("modal-prev").disabled = true;
    eid("modal-next").disabled = true;
    // Only show thumbnail controls for TRAKE event cards (eventIdx is a number)
    const isTrake = eventIdx !== null && eventIdx !== undefined;
    eid("btn-setthumb").style.display = isTrake ? "inline-block" : "none";
    eid("btn-revert-modal").style.display = "none";

    if (!videoId) { eid("modal-title").textContent = "No video ID"; return; }

    fetch("/api/video-shots?video_id=" + encodeURIComponent(videoId))
      .then(r => r.json())
      .then(data => {
        const shots = data.shots || [];
        if (!shots.length) { eid("modal-title").textContent = "No shots found"; return; }
        modal.shots = shots;
        // Find the shot + frame matching frameId
        let foundSi = 0, foundFi = 0, found = false;
        for (let si = 0; si < shots.length && !found; si++) {
          const frames = shots[si].frames || [];
          for (let fi = 0; fi < frames.length; fi++) {
            if (frames[fi].frame_id === frameId) {
              foundSi = si; foundFi = fi; found = true; break;
            }
          }
        }
        modal.shotIdx = foundSi;
        modal.frameIdx = foundFi;
        renderModal();
      })
      .catch(() => {
        eid("modal-title").textContent = "Could not load shots";
      });
  }

  // ─── Modal render (called after shots loaded, or after navigation) ────────────
  function renderModal() {
    if (!modal.shots.length) return;
    const shot = modal.shots[modal.shotIdx];
    const frame = shot.frames[modal.frameIdx];

    // Set main image
    eid("modal-img").src = frame.image_url;

    // Header
    eid("modal-title").textContent =
      `Shot ${modal.shotIdx+1}/${modal.shots.length}  (${shot.shot_id})`;
    eid("modal-time").textContent = fmtTime(frame.timestamp_sec);

    // Footer
    eid("modal-footer").textContent =
      `Frame ${modal.frameIdx+1}/${shot.frames.length} · ${fmtTime(frame.timestamp_sec)} · idx ${frame.frame_index}`;

    // Strip
    eid("modal-strip").innerHTML = shot.frames.map((f, fi) =>
      `<img src="${esc(f.image_url)}" class="${fi===modal.frameIdx?"active":""}"
        onclick="selectStrip(event,${fi})" title="${fmtTime(f.timestamp_sec)}">`
    ).join("");

    // Nav buttons disabled status
    const isFirst = modal.shotIdx <= 0 && modal.frameIdx <= 0;
    const currentShotFrames = shot.frames || [];
    const isLast = modal.shotIdx >= modal.shots.length - 1 && modal.frameIdx >= currentShotFrames.length - 1;
    eid("modal-prev").disabled = isFirst;
    eid("modal-next").disabled = isLast;

    // Thumbnail buttons (only for TRAKE)
    if (modal.eventIdx !== null && modal.eventIdx !== undefined) {
      const key = trakeKey(modal.videoId, modal.chainIdx, modal.eventIdx);
      const st = thumbState[key];
      const isCustom = st && st.currentUrl !== st.originalUrl;
      eid("btn-setthumb").textContent = isCustom ? "Cập nhật thumbnail" : "Làm thumbnail";
      eid("btn-revert-modal").style.display = isCustom ? "inline-block" : "none";
    }
  }

  // ─── Continuous Frame Navigation (Next / Previous Frame) ──────────────────────
  function prevFrame() {
    if (!modal.shots.length) return;
    if (modal.frameIdx > 0) {
      modal.frameIdx--;
    } else if (modal.shotIdx > 0) {
      modal.shotIdx--;
      const prevShotFrames = modal.shots[modal.shotIdx]?.frames || [];
      modal.frameIdx = Math.max(0, prevShotFrames.length - 1);
    }
    renderModal();
  }

  function nextFrame() {
    if (!modal.shots.length) return;
    const currentShotFrames = modal.shots[modal.shotIdx]?.frames || [];
    if (modal.frameIdx < currentShotFrames.length - 1) {
      modal.frameIdx++;
    } else if (modal.shotIdx < modal.shots.length - 1) {
      modal.shotIdx++;
      modal.frameIdx = 0;
    }
    renderModal();
  }

  // ─── Strip click ─────────────────────────────────────────────────────────────
  function selectStrip(e, fi) {
    if (e && e.stopPropagation) e.stopPropagation();
    modal.frameIdx = fi;
    renderModal();
  }

  // ─── Navigation button click handlers ───────────────────────────────────────
  eid("modal-prev").addEventListener("click", e => {
    e.stopPropagation();
    prevFrame();
  });
  eid("modal-next").addEventListener("click", e => {
    e.stopPropagation();
    nextFrame();
  });

  // ─── Set thumbnail ───────────────────────────────────────────────────────────
  eid("btn-setthumb").addEventListener("click", () => {
    if (!modal.shots.length || modal.eventIdx === null) return;
    const frame = modal.shots[modal.shotIdx].frames[modal.frameIdx];
    const key = trakeKey(modal.videoId, modal.chainIdx, modal.eventIdx);
    if (!thumbState[key]) return;
    thumbState[key].currentUrl = frame.image_url;
    thumbState[key].currentTimestamp = frame.timestamp_sec;
    thumbState[key].currentFrameId = frame.frame_id;
    refreshCard(modal.videoId, modal.chainIdx, modal.eventIdx);
    renderModal();
    statusEl.innerHTML = `<strong>Thumbnail updated</strong> — Event ${modal.eventIdx+1} <span class="pill">CUSTOM</span>`;
  });

  // ─── Revert from modal ───────────────────────────────────────────────────────
  eid("btn-revert-modal").addEventListener("click", () => {
    if (modal.eventIdx === null) return;
    revertCard(modal.videoId, modal.chainIdx, modal.eventIdx);
    renderModal();
  });

  // ─── Close modal ─────────────────────────────────────────────────────────────
  function closeModal() {
    eid("frame-modal").style.display = "none";
    eid("modal-img").src = "";
    modal.open = false;
    modal.shots = [];
  }
  eid("modal-close-x").addEventListener("click", closeModal);
  eid("btn-close-modal").addEventListener("click", closeModal);
  eid("frame-modal").addEventListener("click", e => {
    if (e.target === eid("frame-modal")) closeModal();
  });
  eid("modal-box").addEventListener("click", e => e.stopPropagation());

  // ─── Keyboard shortcuts ───────────────────────────────────────────────────────
  document.addEventListener("keydown", e => {
    if (!modal.open) return;
    if (e.key === "Escape") { closeModal(); return; }
    if (e.key === "ArrowLeft") { e.preventDefault(); prevFrame(); }
    if (e.key === "ArrowRight") { e.preventDefault(); nextFrame(); }
    if (e.key === "ArrowUp" && modal.shotIdx > 0) { e.preventDefault(); modal.shotIdx--; modal.frameIdx = 0; renderModal(); }
    if (e.key === "ArrowDown" && modal.shotIdx < modal.shots.length - 1) { e.preventDefault(); modal.shotIdx++; modal.frameIdx = 0; renderModal(); }
  });

  // Toggle VQA settings visibility
  form.track.addEventListener('change', (e) => {
      document.getElementById('vqa-settings').style.display = e.target.value === 'vqa' ? 'block' : 'none';
    });
    // Trigger initially
    if (form.track.value === 'vqa') {
        document.getElementById('vqa-settings').style.display = 'block';
    }

    // --- Mode switch: manual search vs agent chat ---
    function setMode(mode) {
      const isAgent = mode === "agent";
      document.getElementById("agent-chat").style.display = isAgent ? "" : "none";
      document.getElementById("search-form").style.display = isAgent ? "none" : "";
      document.querySelectorAll(".mode-btn").forEach(btn => {
        btn.classList.toggle("active", btn.dataset.mode === mode);
      });
      localStorage.setItem("ui_mode", mode);
    }
    document.querySelectorAll(".mode-btn").forEach(btn => {
      btn.addEventListener("click", () => setMode(btn.dataset.mode));
    });
    setMode(localStorage.getItem("ui_mode") || "manual");

    // --- Interactive agent chat (stateless server: we keep the history) ---
    const chatMessages = [];
    const chatBox = document.getElementById("chat-messages");
    const chatInput = document.getElementById("chat-input");
    const chatSend = document.getElementById("chat-send");
    const chatSuggestions = document.getElementById("chat-suggestions");

    function chatRender() {
      chatBox.innerHTML = chatMessages.map(m =>
        `<div style="margin-bottom:6px;"><b>${m.role === "user" ? "Bạn" : "Agent"}:</b> ${escapeHtml(m.content)}</div>`
      ).join("");
      chatBox.scrollTop = chatBox.scrollHeight;
    }

    async function chatSubmit(text) {
      if (!text.trim()) return;
      chatMessages.push({ role: "user", content: text.trim() });
      chatInput.value = "";
      chatSuggestions.innerHTML = "";
      chatRender();
      chatSend.disabled = true;
      statusEl.textContent = "Agent đang tìm...";
      try {
        const resp = await fetch("/api/agent/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ messages: chatMessages }),
        });
        const data = await resp.json();
        if (data.error) throw new Error(data.error);
        chatMessages.push({ role: "assistant", content: data.message || "" });
        chatRender();
        (data.suggestions || []).forEach(s => {
          const btn = document.createElement("button");
          btn.type = "button";
          btn.textContent = s;
          btn.style.cssText = "font-size:12px;padding:3px 8px;";
          btn.onclick = () => chatSubmit(s);
          chatSuggestions.appendChild(btn);
        });
        if (data.results && data.results.length) renderResults(data.results);
        statusEl.textContent = `Agent: ${data.results ? data.results.length : 0} kết quả.`;
      } catch (err) {
        statusEl.textContent = "Agent lỗi: " + err.message;
      } finally {
        chatSend.disabled = false;
      }
    }

    chatSend.addEventListener("click", () => chatSubmit(chatInput.value));
    chatInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") { e.preventDefault(); chatSubmit(chatInput.value); }
    });

    // --- Track Docs Modal handlers ---
    const trackDocsModal = document.getElementById("track-docs-modal");
    const openTrackDocsBtn = document.getElementById("open-track-docs");
    const closeTrackDocsBtn = document.getElementById("close-track-docs");
    if (openTrackDocsBtn && trackDocsModal) {
      openTrackDocsBtn.addEventListener("click", () => {
        trackDocsModal.style.display = "flex";
      });
    }
    if (closeTrackDocsBtn && trackDocsModal) {
      closeTrackDocsBtn.addEventListener("click", () => {
        trackDocsModal.style.display = "none";
      });
    }
    if (trackDocsModal) {
      trackDocsModal.addEventListener("click", (e) => {
        if (e.target === trackDocsModal) trackDocsModal.style.display = "none";
      });
    }
"""
