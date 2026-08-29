"""Full INDEX_HTML page assembly component."""

from __future__ import annotations

from ui.views.main_grid import MAIN_GRID_HTML
from ui.views.scripts import APP_JS
from ui.views.sidebar import SIDEBAR_HTML
from ui.views.styles import CSS_STYLES

INDEX_HTML = (
    r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CodeNova Multimodal Retriever</title>
  <style>"""
    + CSS_STYLES
    + r"""
  </style>
</head>
<body>
<header class="app-header">
  <div class="brand-group">
    <div class="brand-logo">
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M12 2L2 7L12 12L22 7L12 2Z" stroke="#0f766e" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>
        <path d="M2 17L12 22L22 17" stroke="#0f766e" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>
        <path d="M2 12L12 17L22 12" stroke="#0d9488" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>
      </svg>
    </div>
    <div class="brand-text">
      <h1 class="header-title">CodeNova <span class="accent-text">Multimodal Retriever</span></h1>
      <div class="header-subtitle">
        <span>Experiment: __ACTIVE_EXPERIMENT__</span>
        <span>&middot;</span>
        <span>Models: __ACTIVE_MODELS__</span>
      </div>
    </div>
  </div>
  <div class="header-actions">
    <div id="mode-switch">
      <button type="button" class="mode-btn active" data-mode="manual">🛠️ Search Studio</button>
      <button type="button" class="mode-btn" data-mode="agent">🤖 ReAct Agent</button>
    </div>
  </div>
</header>
<main>
"""
    + SIDEBAR_HTML
    + MAIN_GRID_HTML
    + r"""
  </main>
  <script>"""
    + APP_JS
    + r"""
  </script>
</body>
</html>
"""
)
