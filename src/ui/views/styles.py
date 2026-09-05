"""CSS styles for CodeNova UI."""

from __future__ import annotations

CSS_STYLES = r"""
    :root {
      --bg: #f7f7f4; --panel: #ffffff; --text: #1c1f24;
      --muted: #667085; --line: #d9dde3;
      --accent: #0f766e; --accent-strong: #115e59; --warn: #a16207;
      --accent-gradient: linear-gradient(135deg, #0f766e, #0d9488);
    }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: Inter, ui-sans-serif, system-ui, sans-serif; color: var(--text); background: var(--bg); }
    /* Modern App Header */
    .app-header {
      padding: 12px 24px;
      border-bottom: 1px solid var(--line);
      background: #ffffff;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      box-shadow: 0 2px 12px rgba(15, 118, 110, 0.05);
      position: sticky;
      top: 0;
      z-index: 100;
    }
    .brand-group {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .brand-logo {
      width: 40px;
      height: 40px;
      border-radius: 10px;
      background: #f0fdf8;
      border: 1px solid #ccfbf1;
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 2px 6px rgba(15, 118, 110, 0.08);
    }
    .header-title {
      margin: 0;
      font-size: 19px;
      font-weight: 800;
      letter-spacing: -0.02em;
      color: var(--text);
    }
    .header-title .accent-text {
      background: linear-gradient(135deg, #059669, #0f766e, #0284c7);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      font-weight: 850;
    }
    .header-subtitle {
      display: flex;
      align-items: center;
      gap: 6px;
      font-size: 11px;
      color: var(--muted);
      margin-top: 1px;
      font-weight: 500;
    }
    .status-dot {
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: #10b981;
      box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.25);
    }
    #mode-switch {
      display: flex;
      gap: 4px;
      background: #f1f5f9;
      padding: 4px;
      border-radius: 8px;
      border: 1px solid var(--line);
    }
    .mode-btn {
      width: auto;
      margin: 0;
      padding: 6px 14px;
      font-size: 12px;
      background: transparent;
      color: var(--muted);
      border: none;
      border-radius: 6px;
      cursor: pointer;
      font-weight: 600;
      transition: all .15s ease;
    }
    .mode-btn.active {
      background: #ffffff;
      color: var(--accent-strong);
      font-weight: 750;
      box-shadow: 0 2px 6px rgba(15,118,110,0.15);
    }
    main { display: grid; grid-template-columns: minmax(320px, 420px) 1fr; min-height: calc(100vh - 61px); }
    aside { padding: 18px; border-right: 1px solid var(--line); background: var(--panel); }
    section { padding: 18px; }
    label {
      display: block;
      margin: 14px 0 6px;
      color: var(--muted);
      font-size: 13px;
      font-weight: 650;
    }
    select, input, textarea {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      font: inherit;
      transition: border-color .15s ease, box-shadow .15s ease;
    }
    select:focus, input:focus, textarea:focus {
      outline: none;
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(15, 118, 110, 0.15);
    }
    select, input {
      padding: 10px 11px;
      background: #fff;
      color: var(--text);
    }
    textarea {
      min-height: 84px;
      resize: vertical;
      line-height: 1.45;
    }

    /* Modern Card Containers */
    .config-container {
      margin: 14px 0 10px;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .config-card {
      padding: 10px 12px;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #fafafa;
      transition: background-color .15s ease, border-color .15s ease;
    }
    .config-card:hover {
      background: #f4f6f6;
      border-color: #cbd5e1;
    }
    .config-card-header {
      margin-bottom: 8px;
    }
    .config-card-title {
      font-size: 12px;
      font-weight: 700;
      color: var(--accent-strong);
      text-transform: uppercase;
      letter-spacing: .03em;
    }

    /* Chip Badges for Embedding Models */
    .chips-group {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }
    .chip-item {
      cursor: pointer;
      margin: 0 !important;
    }
    .chip-item input[type="checkbox"] {
      position: absolute;
      opacity: 0;
      width: 0;
      height: 0;
    }
    .chip-badge {
      display: inline-block;
      padding: 5px 10px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: #fff;
      color: var(--muted);
      font-size: 12px;
      font-weight: 600;
      transition: all .15s ease;
      user-select: none;
    }
    .chip-item input[type="checkbox"]:checked + .chip-badge {
      background: #e6f5f3;
      border-color: var(--accent);
      color: var(--accent-strong);
      box-shadow: 0 1px 4px rgba(15, 118, 110, 0.15);
    }

    /* Check Switches for LLM & Reranker */
    .check-switch {
      display: flex;
      align-items: center;
      gap: 10px;
      margin: 2px 0 !important;
      cursor: pointer;
      user-select: none;
    }
    .check-switch input[type="checkbox"] {
      width: 18px;
      height: 18px;
      accent-color: var(--accent);
      cursor: pointer;
      margin: 0;
    }
    .switch-label {
      font-size: 13px;
      font-weight: 600;
      color: var(--text);
    }

    .row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      align-items: end;
    }

    /* Modern Buttons */
    .btn-guide {
      width: auto;
      padding: 3px 10px;
      margin: 0;
      font-size: 12px;
      background: #f0fdf8;
      color: var(--accent);
      border: 1px solid var(--accent);
      border-radius: 6px;
      cursor: pointer;
      font-weight: 600;
      transition: all .15s ease;
    }
    .btn-guide:hover {
      background: var(--accent);
      color: #fff;
    }
    .btn-submit {
      margin-top: 16px;
      padding: 11px 14px;
      border: none;
      border-radius: 8px;
      background: var(--accent-gradient);
      color: #fff;
      font-weight: 750;
      font-size: 14px;
      cursor: pointer;
      box-shadow: 0 4px 12px rgba(15, 118, 110, 0.25);
      transition: all .15s ease;
    }
    .btn-submit:hover {
      transform: translateY(-1px);
      box-shadow: 0 6px 16px rgba(15, 118, 110, 0.35);
    }
    .btn-submit:active {
      transform: translateY(0);
    }
    .btn-submit:disabled {
      opacity: .65;
      cursor: wait;
      transform: none;
    }
    .btn-secondary {
      width: auto;
      padding: 6px 14px;
      margin-top: 6px;
      font-size: 13px;
      background: #fff;
      color: var(--accent);
      border: 1px solid var(--line);
      border-radius: 6px;
      cursor: pointer;
      transition: all .15s ease;
    }
    .btn-secondary:hover {
      border-color: var(--accent);
      background: #f0fdf8;
    }

    .hint, .status { margin-top: 12px; color: var(--muted); font-size: 13px; line-height: 1.45; }
    .status strong { color: var(--text); }
    .status.warn { color: var(--warn); }
    .pill { display: inline-block; padding: 2px 7px; border-radius: 999px; background: #e6f5f3; color: var(--accent-strong); font-size: 12px; font-weight: 700; }

    /* Results grid - clean minimalist design */
    .results { display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr)); gap: 16px; }
    .card { overflow: hidden; border: 1px solid var(--line); border-radius: 10px; background: var(--panel); transition: transform .15s ease, box-shadow .15s ease; display: flex; flex-direction: column; }
    .card:hover { transform: translateY(-3px); box-shadow: 0 8px 24px rgba(15, 118, 110, 0.12); border-color: var(--accent); }
    .card img { width: 100%; aspect-ratio: 16/9; object-fit: cover; display: block; background: #e5e7eb; cursor: zoom-in; }
    .card-meta { padding: 10px 12px; display: flex; flex-direction: column; gap: 4px; font-size: 13px; }
    .card-header-row { display: flex; justify-content: space-between; align-items: center; }
    .card-rank { font-size: 11px; background: #e6f5f3; color: var(--accent-strong); font-weight: 750; padding: 2px 7px; border-radius: 999px; }
    .card-score { font-size: 12px; font-weight: 650; color: var(--accent); }
    .card-vname { font-size: 13px; font-weight: 700; color: var(--text); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; margin-top: 2px; }
    .card-info-row { display: flex; justify-content: space-between; align-items: center; font-size: 12px; color: var(--muted); margin-top: 2px; }
    .card-frame-badge { font-weight: 700; color: var(--accent-strong); background: #f0fdf8; padding: 1px 6px; border-radius: 4px; border: 1px solid #ccfbf1; }

    /* Answer / pipeline */
    .answer-box { margin-bottom: 18px; padding: 18px 20px; border: 2px solid var(--accent); border-radius: 10px; background: #f0fdf8; }
    .answer-box .label { font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: .05em; color: var(--muted); }
    .answer-box .answer-text { margin-top: 6px; font-size: 18px; font-weight: 700; color: var(--accent-strong); line-height: 1.45; }
    .pipeline-toggle { margin-top: 10px; background: none; border: 1px solid var(--line); padding: 6px 12px; border-radius: 6px; color: var(--muted); cursor: pointer; font-size: 12px; }
    .pipeline-toggle:hover { background: var(--panel); }
    .pipeline-detail { display: none; margin-top: 10px; padding: 14px 16px; border: 1px solid var(--line); border-radius: 8px; background: var(--panel); font-size: 13px; line-height: 1.5; }
    .pipeline-detail.open { display: block; }
    .pipeline-detail code { display: block; white-space: pre-wrap; font-size: 12px; color: var(--muted); }

    /* TRAKE event card */
    .video-block { margin-bottom: 20px; padding: 14px 16px; border: 2px solid var(--accent); border-radius: 10px; background: var(--panel); }
    .video-block-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
    .event-grid { display: grid; gap: 10px; }

    /* Each event card */
    .ev-card { position: relative; border-radius: 8px; overflow: hidden; border: 1px solid var(--line); background: #f0fdf8; }
    .ev-card img { width: 100%; aspect-ratio: 16/9; object-fit: cover; display: block; background: #e5e7eb; cursor: zoom-in; transition: opacity .15s; }
    .ev-card img:hover { opacity: .88; }
    .ev-card .ev-info { padding: 5px 8px 6px; font-size: 12px; color: var(--muted); display: flex; justify-content: space-between; align-items: center; gap: 6px; }
    .ev-card .revert-badge {
      display: none; position: absolute; top: 5px; right: 5px;
      background: rgba(0,0,0,.65); color: #fff; border: none;
      border-radius: 5px; padding: 3px 8px; font-size: 11px; font-weight: 600;
      cursor: pointer; margin-top: 0; width: auto;
    }
    .ev-card .revert-badge:hover { background: rgba(0,0,0,.85); }
    .ev-card.has-custom .revert-badge { display: block; }
    .ev-card.has-custom { border-color: var(--accent); }

    /* Modal - White & Teal theme */
    #frame-modal {
      display: none; position: fixed; inset: 0; z-index: 999;
      background: rgba(15, 23, 42, 0.85); backdrop-filter: blur(4px); align-items: center; justify-content: center;
    }
    #frame-modal.open { display: flex; }
    .modal-box {
      display: flex; flex-direction: column;
      width: 95vw; height: 92vh; max-width: 1350px;
      border-radius: 14px; overflow: hidden;
      background: #ffffff; box-shadow: 0 20px 50px rgba(0,0,0,0.3); border: 1px solid var(--line);
    }
    .modal-top {
      display: flex; justify-content: space-between; align-items: center;
      padding: 10px 18px; background: var(--accent-gradient); color: #ffffff; font-size: 14px; flex-shrink: 0;
    }
    .modal-title-group { display: flex; align-items: center; gap: 10px; font-weight: 700; }
    .sub-id-badge { font-size: 11px; background: rgba(255,255,255,0.2); padding: 2px 8px; border-radius: 999px; font-weight: 500; }
    .modal-top .time-badge { background: #ffffff; color: var(--accent-strong); font-weight: 750; padding: 3px 10px; border-radius: 999px; font-size: 12px; }
    .modal-top .close-x { background: none; border: none; color: #ffffff; font-size: 24px; cursor: pointer; padding: 0 4px; margin-top: 0; width: auto; opacity: 0.85; }
    .modal-top .close-x:hover { opacity: 1; }

    .modal-mid { flex: 1; display: flex; align-items: stretch; padding: 12px; gap: 12px; min-height: 0; background: #f8fafc; }
    .modal-mid .img-area { flex: 1.2; display: flex; justify-content: center; align-items: center; min-height: 0; overflow: hidden; background: #0f172a; border-radius: 10px; padding: 8px; }
    .modal-mid .img-area img { max-width: 100%; max-height: 100%; object-fit: contain; border-radius: 6px; display: block; }
    .modal-nav { flex-shrink: 0; background: #ffffff; border: 1px solid var(--line); color: var(--text); font-size: 20px; cursor: pointer; padding: 0 14px; border-radius: 8px; display: flex; align-items: center; justify-content: center; margin-top: 0; width: auto; transition: all .15s ease; }
    .modal-nav:hover { background: #e6f5f3; border-color: var(--accent); color: var(--accent-strong); }
    .modal-nav:disabled { opacity: .3; cursor: default; background: #f1f5f9; }

    .modal-details-side {
      flex: 1; min-width: 280px; max-width: 440px; overflow-y: auto;
      display: flex; flex-direction: column; gap: 10px; padding: 12px;
      background: #ffffff; border: 1px solid var(--line); border-radius: 10px;
    }
    .modal-section { background: #f0fdf8; border-left: 4px solid var(--accent); border-radius: 6px; padding: 10px 12px; }
    .modal-section-title { font-size: 11px; font-weight: 750; color: var(--accent-strong); text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 4px; }
    .modal-section-content { font-size: 13px; color: var(--text); line-height: 1.5; white-space: pre-wrap; word-break: break-word; }

    .modal-strip { display: flex; justify-content: flex-start; gap: 8px; padding: 10px 16px; background: #ffffff; border-top: 1px solid var(--line); flex-shrink: 0; overflow-x: auto; }
    .modal-strip img { width: 96px; height: 54px; object-fit: cover; border-radius: 6px; border: 2px solid transparent; cursor: pointer; flex-shrink: 0; transition: all .15s ease; }
    .modal-strip img:hover { border-color: #99f6e4; }
    .modal-strip img.active { border-color: var(--accent); box-shadow: 0 0 0 2px rgba(15,118,110,0.3); transform: scale(1.04); }

    .modal-bot { display: flex; justify-content: space-between; align-items: center; padding: 10px 16px; background: #ffffff; border-top: 1px solid var(--line); color: var(--muted); font-size: 12px; gap: 8px; flex-wrap: wrap; flex-shrink: 0; }
    .modal-bot .actions { display: flex; gap: 6px; }
    .btn-setthumb { background: #0f766e; color: #fff; border: none; border-radius: 6px; padding: 6px 14px; font-size: 12px; font-weight: 600; cursor: pointer; margin-top: 0; width: auto; }
    .btn-setthumb:hover { background: #115e59; }
    .btn-revert-modal { background: #555; color: #fff; border: none; border-radius: 6px; padding: 6px 14px; font-size: 12px; font-weight: 600; cursor: pointer; margin-top: 0; width: auto; }
    .btn-revert-modal:hover { background: #777; }
    .btn-close-modal { background: #333; color: #ccc; border: none; border-radius: 6px; padding: 6px 14px; font-size: 12px; cursor: pointer; margin-top: 0; width: auto; }
    .btn-close-modal:hover { background: #444; }
    @media (max-width: 860px) { main { grid-template-columns: 1fr; } aside { border-right: 0; border-bottom: 1px solid var(--line); } }
"""
