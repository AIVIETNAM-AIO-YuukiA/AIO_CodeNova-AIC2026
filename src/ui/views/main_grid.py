"""Main results view section HTML component."""

from __future__ import annotations

from ui.views.modal import MODAL_HTML

MAIN_GRID_HTML = r"""    <section>
      <div id="answer-box"></div>
      <div id="events-box"></div>
      <div id="pipeline-box"></div>
      <div id="results" class="results"></div>
""" + MODAL_HTML + r"""
    </section>"""
