"""Where CPU-bound work runs, so the event loop stays free.

`rescore`, `score_many`, `attribute` and `to_pdf` are synchronous NumPy / ONNX
/ fpdf2 code. Awaiting them directly would block every other request for the
duration, which on a 15-candidate batch is a third of a second.

Two pools, deliberately sized:

* **Scoring: one worker.** ONNX already parallelises internally via
  ``intra_op_num_threads``, so a second concurrent batch does not go faster --
  it competes for the same cores and doubles peak memory. `dense.BATCH` keeps
  peak RSS flat *within* one call; it does nothing about two calls at once, and
  the OOM reaper's symptom is a blank page with no traceback. Serialising here
  is what keeps that from happening.
* **Rendering: two workers.** PDF generation is independent of ONNX and should
  not queue behind a scoring batch just to produce a download.
"""
from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any


class Pools:
    def __init__(self, *, score_workers: int = 1, render_workers: int = 2) -> None:
        self._scoring = ThreadPoolExecutor(max_workers=score_workers,
                                           thread_name_prefix="rba-score")
        self._render = ThreadPoolExecutor(max_workers=render_workers,
                                          thread_name_prefix="rba-render")

    async def score(self, fn: Callable[..., Any], *args, **kwargs) -> Any:
        """Run scoring work on the single scoring thread."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._scoring,
                                          lambda: fn(*args, **kwargs))

    async def render(self, fn: Callable[..., Any], *args, **kwargs) -> Any:
        """Run a renderer off the event loop, without queueing behind scoring."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._render,
                                          lambda: fn(*args, **kwargs))

    def shutdown(self) -> None:
        self._scoring.shutdown(wait=False, cancel_futures=True)
        self._render.shutdown(wait=False, cancel_futures=True)
