"""FastAPI application factory.

Replaces ui/server.py's hand-rolled ThreadingHTTPServer + if/elif routing.
The one-time startup sequence — validate readiness, ensure manifests, build
the retriever, warm every model — is unchanged from ``serve_ui()``; it just
runs in a ``lifespan`` context manager and stores its results on
``app.state`` instead of being captured by a request-handler closure.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from api.routers import agent, media, models, search, vqa
from config.settings import Experiment
from core.logging import get_logger
from ui.api import ensure_manifests
from ui.server import _validate_experiment_for_serving, _warmup_models
from ui.views.page import INDEX_HTML

LOGGER = get_logger(__name__)


def _render_index_html(experiment: Experiment, default_top_k: int) -> str:
    import html as html_lib

    return INDEX_HTML.replace('value="20"', f'value="{default_top_k}"').replace(
        "__ACTIVE_EXPERIMENT__", html_lib.escape(experiment.name)
    )


def create_app(
    experiment: Experiment,
    default_top_k: int = 20,
    reranker=None,
    reranker_top_k: int = 10,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        from core.paths import set_project_root
        from retrieval.vqa import _get_retriever

        readiness = _validate_experiment_for_serving(experiment)
        LOGGER.info(
            "[readiness] Validated experiment=%s generated_at=%s",
            experiment.name,
            readiness.get("generated_at"),
        )
        ensure_manifests(experiment)
        # frame_path values in manifests are relative to the working directory
        # the pipeline was run from, not to experiment.run_dir.
        set_project_root(Path.cwd())

        retriever = _get_retriever(experiment)
        warmup = _warmup_models(reranker, experiment, retriever)

        app.state.experiment = experiment
        app.state.retriever = retriever
        app.state.reranker = warmup.ui_reranker
        app.state.default_top_k = default_top_k
        app.state.reranker_top_k = reranker_top_k

        yield

    app = FastAPI(title="CodeNova Retrieval UI", lifespan=lifespan)

    app.include_router(search.router)
    app.include_router(vqa.router)
    app.include_router(agent.router)
    app.include_router(media.router)
    app.include_router(models.router)

    @app.get("/", response_class=HTMLResponse)
    def index():
        return _render_index_html(experiment, default_top_k)

    @app.get("/health")
    def health():
        return {"ok": True, "experiment": experiment.name}

    return app
