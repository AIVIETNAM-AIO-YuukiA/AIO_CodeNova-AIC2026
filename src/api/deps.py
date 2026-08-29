"""FastAPI dependencies for reading server-lifetime state.

Experiment, Retriever, and the UI reranker are all built once at server
startup (see api/app.py's lifespan) — the same one-time-warmup shape
ui/server.py used before this refactor, just exposed through app.state
instead of a closure captured by a hand-rolled request handler.
"""

from __future__ import annotations

from fastapi import Request

from config.settings import Experiment


def get_experiment(request: Request) -> Experiment:
    return request.app.state.experiment


def get_retriever(request: Request):
    return request.app.state.retriever


def get_reranker(request: Request):
    return request.app.state.reranker


def get_default_top_k(request: Request) -> int:
    return request.app.state.default_top_k


def get_reranker_top_k(request: Request) -> int:
    return request.app.state.reranker_top_k
