"""Describing a candidate document as a small patch, not a whole document.

The v2 UI asks "what would the score be if I turned this line off?" about a
hundred times per view. Shipping a full `ResumeDoc` for each would make the
batch-scoring request enormous for no reason, so the client sends the base
document once and then a list of ops per candidate.

Two properties matter more than the convenience:

**Ops never mutate.** `apply_ops` works on a copy. The scoring helpers it feeds
-- `attribution.attribute`, `variants._measure_gain`, `additions._measure` --
all toggle the live document and restore it in a `finally`, which is fine for
one caller in one thread and wrong the moment a server runs two requests at
once. Copy-on-apply is what makes concurrent scoring safe.

**Entries are addressed the way the rest of the codebase already addresses
them**: ``f"{section.title}#{index}"``. That key is what `matching.py`,
`additions.py`, `variants.py` and `grounding.py` all use, so there is no second
addressing scheme to keep in sync.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

from .document import Bullet, ResumeDoc

OpKind = Literal["enable", "disable", "set_text", "add_bullet", "remove_bullet"]


class UnknownTarget(KeyError):
    """An op referenced a bullet or entry that is not in the document.

    Raised rather than ignored: a silently dropped op would score a document
    the user is not looking at, and the number would be wrong in a way nothing
    downstream could detect.
    """


@dataclass(frozen=True)
class Op:
    kind: OpKind
    bullet_id: str = ""
    entry_key: str = ""
    text: str = ""

    @classmethod
    def from_dict(cls, payload: dict) -> Op:
        kind = payload.get("op") or payload.get("kind") or ""
        if kind not in ("enable", "disable", "set_text", "add_bullet",
                        "remove_bullet"):
            raise ValueError(f"unknown op {kind!r}")
        return cls(kind=kind, bullet_id=payload.get("bullet_id", ""),
                   entry_key=payload.get("entry_key", ""),
                   text=payload.get("text", ""))

    def to_dict(self) -> dict:
        return {"op": self.kind, "bullet_id": self.bullet_id,
                "entry_key": self.entry_key, "text": self.text}


def entry_key(section_title: str, index: int) -> str:
    """The one addressing scheme. See the module docstring."""
    return f"{section_title}#{index}"


def entries_by_key(doc: ResumeDoc) -> dict[str, object]:
    return {entry_key(section.title, index): entry
            for section in doc.sections
            for index, entry in enumerate(section.entries)}


def copy_doc(doc: ResumeDoc) -> ResumeDoc:
    """A deep copy via the document's own round-trip, so a copy can never
    diverge from what `from_dict` would accept."""
    return ResumeDoc.from_dict(doc.to_dict())


def apply_ops(doc: ResumeDoc, ops) -> ResumeDoc:
    """`doc` with every op applied, as a new document. `doc` is untouched."""
    work = copy_doc(doc)
    for raw in ops or []:
        op = raw if isinstance(raw, Op) else Op.from_dict(raw)
        _apply_one(work, op)
    return work


def _apply_one(doc: ResumeDoc, op: Op) -> None:
    if op.kind in ("enable", "disable", "set_text", "remove_bullet"):
        bullet = doc.by_id(op.bullet_id)
        if bullet is None:
            raise UnknownTarget(f"no bullet {op.bullet_id!r}")
        if op.kind == "enable":
            bullet.enabled = True
        elif op.kind == "disable":
            bullet.enabled = False
        elif op.kind == "set_text":
            # The id stays put. `Bullet.__post_init__` only derives an id when
            # one is absent, so editing wording does not rename the line out
            # from under the selection the client is holding.
            bullet.text = op.text
        else:
            _remove(doc, op.bullet_id)
        return

    if op.kind == "add_bullet":
        entry = entries_by_key(doc).get(op.entry_key)
        if entry is None:
            raise UnknownTarget(f"no entry {op.entry_key!r}")
        entry.bullets.append(Bullet(text=op.text, origin="assumed"))
        return

    raise ValueError(f"unhandled op {op.kind!r}")


def _remove(doc: ResumeDoc, bullet_id: str) -> None:
    for section in doc.sections:
        for entry in section.entries:
            for index, bullet in enumerate(entry.bullets):
                if bullet.id == bullet_id:
                    del entry.bullets[index]
                    return


def signature(doc: ResumeDoc) -> str:
    """What a set of deltas was computed against.

    Per-line deltas stay valid only while the enabled lines and their wording
    are unchanged. Comparing this to the live document is how the UI knows to
    mark numbers as needing a refresh rather than quietly showing values that
    describe a document the user has already edited away from. Same contract as
    v1's `app._signature`, hashed so it can travel in a header.
    """
    parts = [f"{b.id}\x1f{b.text}\x1f{int(b.enabled)}" for b in doc.all_bullets()]
    digest = hashlib.sha256("\x1e".join(parts).encode("utf-8"))
    return digest.hexdigest()[:16]
