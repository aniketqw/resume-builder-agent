"""The FastAPI application.

Served as one process with one worker. That is not laziness: the session store,
the ONNX InferenceSession, the dense memo, the rate limiters and several
`lru_cache`s all live in process memory. A second worker would duplicate the
model, halve the cache hit rate and -- worst -- give each worker its own rate
limiter, so the Groq budget would be exceeded by exactly the worker count.
Scale inside the process, via the thread pools.
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .deps import AppState
from .routes import documents, github, meta, render, score

log = logging.getLogger(__name__)

def web_root() -> Path:
    """Where the static frontend lives.

    ``parents[2]`` is right for a source checkout (``src/rba_api/app.py`` ->
    repo root) and wrong for an installed wheel, where it points outside
    site-packages -- the same trap `rba.config` documents for config.yaml. The
    container sets RBA_WEB_ROOT; RBA_ROOT covers it too, so a deployment that
    has already set one does not need both.
    """
    explicit = os.environ.get("RBA_WEB_ROOT")
    if explicit:
        return Path(explicit).expanduser()
    root = os.environ.get("RBA_ROOT")
    if root:
        return Path(root).expanduser() / "web"
    return Path(__file__).resolve().parents[2] / "web"


def create_app(*, testing: bool = False) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state = AppState(testing=testing)
        app.state.rba = state
        # Warm on the scoring thread so a slow model load does not hold the
        # event loop before the server reports ready.
        if not testing:
            await state.pools.score(state.warm)
        log.info("rba api ready (dense=%s)", state.dense_ready)
        try:
            yield
        finally:
            state.shutdown()

    app = FastAPI(title="Resume Builder Agent", version="2.0.0",
                  lifespan=lifespan, docs_url="/api/docs",
                  openapi_url="/api/openapi.json")

    for router in (meta.router, documents.router, score.router,
                   render.router, github.router):
        app.include_router(router, prefix="/api")

    @app.exception_handler(ValueError)
    async def _value_error(_request, exc: ValueError):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    # Mounted last so it cannot shadow /api. html=True serves index.html for
    # unknown paths, which is what a client-routed single page needs.
    root = web_root()
    if root.is_dir():
        app.mount("/", StaticFiles(directory=str(root), html=True), name="web")
    else:   # pragma: no cover - only in a broken install
        log.warning("web root %s is missing; serving API only", root)

    return app


app = create_app()
