"""Full INDEX_HTML page assembly component."""

from __future__ import annotations

from ui.views.main_grid import MAIN_GRID_HTML
from ui.views.scripts import APP_JS
from ui.views.sidebar import SIDEBAR_HTML
from ui.views.styles import CSS_STYLES

INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CodeNova Retrieval UI</title>
  <style>""" + CSS_STYLES + r"""
  </style>
</head>
<body>
<header>
  <h1>CodeNova Retrieval UI</h1>
  <div id="mode-switch">
    <button type="button" class="mode-btn active" data-mode="manual">🛠 Thủ công</button>
    <button type="button" class="mode-btn" data-mode="agent">🤖 Agent</button>
  </div>
</header>
<main>
""" + SIDEBAR_HTML + MAIN_GRID_HTML + r"""
  </main>
  <script>""" + APP_JS + r"""
  </script>
</body>
</html>
"""
