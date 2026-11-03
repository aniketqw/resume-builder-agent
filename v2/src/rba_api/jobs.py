"""Work that takes minutes, driven over HTTP.

A real GitHub import is not a request. `pipeline.build_from_github`'s own
module docstring cites a measured full run of 1,433 seconds, and it reports
progress through a synchronous `progress(message, fraction)` callback at eight
fixed checkpoints. So: start it as a task, return an id, and stream the
callback's output.

The callback being *synchronous* is the constraint that shapes this module. It
is invoked from inside the pipeline, which is itself running in the event loop,
so it must never block and never await -- it only appends to a list and sets an
`asyncio.Event`. The SSE endpoint waits on that event.
"""
from __future__ import annotations

import asyncio
import logging
import secrets
import time
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

PENDING, RUNNING, DONE, ERROR, CANCELLED = (
    "pending", "running", "done", "error", "cancelled")

#: How long a finished job's result is kept so a client that reconnects can
#: still collect it.
RESULT_TTL = 10 * 60


@dataclass
class Update:
    message: str
    fraction: float

    def to_dict(self) -> dict:
        return {"message": self.message, "fraction": self.fraction}


@dataclass
class Job:
    id: str
    state: str = PENDING
    created: float = field(default_factory=time.time)
    finished: float = 0.0
    updates: list[Update] = field(default_factory=list)
    result: Any = None
    error: str = ""
    task: asyncio.Task | None = None
    _bell: asyncio.Event = field(default_factory=asyncio.Event)

    # -- producer side (called from inside the pipeline) ------------------
    def report(self, message: str, fraction: float) -> None:
        """The `Progress` callback. Must not block and must not await."""
        self.updates.append(Update(message, max(0.0, min(1.0, fraction))))
        self._bell.set()

    def finish(self, result: Any) -> None:
        self.state, self.result, self.finished = DONE, result, time.time()
        self._bell.set()

    def fail(self, message: str) -> None:
        self.state, self.error, self.finished = ERROR, message, time.time()
        self._bell.set()

    def cancel(self) -> None:
        if self.task is not None:
            self.task.cancel()
        self.state, self.finished = CANCELLED, time.time()
        self._bell.set()

    # -- consumer side ----------------------------------------------------
    async def stream(self):
        """Yield every update, then one terminal event.

        Replays from index 0, so a client that connects late -- or reconnects
        -- still sees the whole run rather than joining midway.
        """
        sent = 0
        while True:
            while sent < len(self.updates):
                yield "progress", self.updates[sent].to_dict()
                sent += 1
            if self.state in (DONE, ERROR, CANCELLED):
                break
            self._bell.clear()
            try:
                await asyncio.wait_for(self._bell.wait(), timeout=20)
            except TimeoutError:
                # A keep-alive, so proxies do not reap an idle connection
                # during the long summarise step.
                yield "ping", {}
        if self.state == DONE:
            yield "done", self.result if self.result is not None else {}
        elif self.state == ERROR:
            yield "error", {"message": self.error}
        else:
            yield "cancelled", {}

    @property
    def progress(self) -> float:
        return self.updates[-1].fraction if self.updates else 0.0

    def to_dict(self, include_result: bool = True) -> dict:
        payload = {"job_id": self.id, "state": self.state,
                   "progress": self.progress,
                   "message": self.updates[-1].message if self.updates else "",
                   "error": self.error}
        if include_result and self.state == DONE:
            payload["result"] = self.result
        return payload


class JobStore:
    def __init__(self, ttl: float = RESULT_TTL) -> None:
        self.ttl = ttl
        self._jobs: dict[str, Job] = {}

    def create(self) -> Job:
        self.sweep()
        job = Job(id=f"j_{secrets.token_urlsafe(10)}")
        self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def sweep(self) -> int:
        cutoff = time.time() - self.ttl
        stale = [k for k, j in self._jobs.items()
                 if j.finished and j.finished < cutoff]
        for key in stale:
            del self._jobs[key]
        return len(stale)

    def __len__(self) -> int:
        return len(self._jobs)
