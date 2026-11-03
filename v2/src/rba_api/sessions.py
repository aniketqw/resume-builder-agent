"""The working document, held server-side between requests.

Why not stateless. The batch scorer's whole job is to answer "what would this
score if I changed one line?" about a hundred times a view, and each of those
needs the base document to apply ops to. Posting a 25 KB document with every
one of those requests would dwarf the payload that matters. Three more reasons,
all structural:

* the dense memo is only worth having if it outlives a request;
* a GitHub import runs for minutes and its result has to land somewhere;
* `attribute`, `variants._measure_gain` and `additions._measure` all toggle the
  live document and restore it in a ``finally``, which is safe for one caller
  and wrong for two -- so there has to be one owner with one lock.

`rev` is the optimistic-concurrency token. A mutation carries the rev it was
computed against; if the document has moved on, the write is refused with 409
and the client resyncs rather than silently clobbering a newer edit.
"""
from __future__ import annotations

import asyncio
import secrets
import time
from dataclasses import dataclass, field
from typing import Any

from rba.ats.batch import DenseCache
from rba.resume.document import ResumeDoc
from rba.resume.selection import signature

#: Two hours. Long enough to survive lunch, short enough that abandoned
#: sessions do not pin documents and embedding caches in memory for ever.
DEFAULT_TTL = 2 * 60 * 60

#: Refuse new sessions past this rather than letting memory grow unbounded.
#: A 503 the operator can see beats an OOM kill they cannot.
DEFAULT_MAX_SESSIONS = 200


class SessionNotFound(KeyError):
    pass


class RevisionMismatch(Exception):
    """The client's `rev` is stale; it must re-read before writing."""

    def __init__(self, expected: int, got: int) -> None:
        super().__init__(f"document is at rev {expected}, client sent {got}")
        self.expected = expected
        self.got = got


class SessionLimit(Exception):
    """Too many live sessions."""


@dataclass
class Session:
    id: str
    created: float
    touched: float

    doc: ResumeDoc | None = None
    rev: int = 0

    # Target
    jd_text: str = ""
    role_key: str = ""
    required_skills: list[str] = field(default_factory=list)
    preferred_skills: list[str] = field(default_factory=list)

    # Derived artefacts, each invalidated by the thing it depends on.
    baseline_doc: ResumeDoc | None = None
    attribution: Any = None
    attribution_signature: str = ""
    versions: Any = None
    matches: list = field(default_factory=list)
    evidence: dict = field(default_factory=dict)
    repos: list = field(default_factory=list)
    excluded: list = field(default_factory=list)
    facts: dict = field(default_factory=dict)
    summaries: dict = field(default_factory=dict)
    variants: Any = None
    additions: Any = None
    pathways: dict = field(default_factory=dict)
    stats: dict = field(default_factory=dict)

    dense_cache: DenseCache = field(default_factory=DenseCache)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    # -- document ---------------------------------------------------------
    def set_document(self, doc: ResumeDoc, *, baseline: ResumeDoc | None = None) -> int:
        self.doc = doc
        if baseline is not None:
            self.baseline_doc = baseline
        self.rev += 1
        self._invalidate_derived()
        return self.rev

    def check_rev(self, got: int | None) -> None:
        if got is None:
            return
        if got != self.rev:
            raise RevisionMismatch(self.rev, got)

    def bump(self) -> int:
        self.rev += 1
        self._invalidate_derived()
        return self.rev

    def set_target(self, *, jd_text: str, role_key: str,
                   required: list[str], preferred: list[str]) -> None:
        """Changing the target invalidates every derived number but NOT the
        document -- the user's edits survive a change of job description."""
        self.jd_text = jd_text
        self.role_key = role_key
        self.required_skills = list(required)
        self.preferred_skills = list(preferred)
        self._invalidate_derived()
        # A new job description means a new embedding target, so cached
        # cosines keyed on the old one are dead weight.
        self.dense_cache = DenseCache()

    def _invalidate_derived(self) -> None:
        self.attribution = None
        self.attribution_signature = ""
        self.versions = None

    @property
    def signature(self) -> str:
        return signature(self.doc) if self.doc is not None else ""

    def snapshot(self) -> dict:
        return {
            "session_id": self.id,
            "rev": self.rev,
            "has_document": self.doc is not None,
            "signature": self.signature,
            "target": {"jd_text": self.jd_text, "role_key": self.role_key,
                       "required_skills": self.required_skills,
                       "preferred_skills": self.preferred_skills},
            "ready": {
                "attribution": self.attribution is not None,
                "versions": self.versions is not None,
                "evidence": bool(self.evidence),
                "pathways": sorted(self.pathways),
            },
        }


class SessionStore:
    def __init__(self, *, ttl: float = DEFAULT_TTL,
                 max_sessions: int = DEFAULT_MAX_SESSIONS) -> None:
        self.ttl = ttl
        self.max_sessions = max_sessions
        self._data: dict[str, Session] = {}

    def create(self) -> Session:
        self.sweep()
        if len(self._data) >= self.max_sessions:
            raise SessionLimit(
                f"{len(self._data)} live sessions is the configured maximum")
        now = time.time()
        session = Session(id=f"s_{secrets.token_urlsafe(12)}",
                          created=now, touched=now)
        self._data[session.id] = session
        return session

    def get(self, session_id: str) -> Session:
        session = self._data.get(session_id)
        if session is None:
            raise SessionNotFound(session_id)
        if time.time() - session.touched > self.ttl:
            self._data.pop(session_id, None)
            raise SessionNotFound(session_id)
        session.touched = time.time()
        return session

    def delete(self, session_id: str) -> None:
        self._data.pop(session_id, None)

    def sweep(self) -> int:
        cutoff = time.time() - self.ttl
        stale = [k for k, v in self._data.items() if v.touched < cutoff]
        for key in stale:
            del self._data[key]
        return len(stale)

    def __len__(self) -> int:
        return len(self._data)
