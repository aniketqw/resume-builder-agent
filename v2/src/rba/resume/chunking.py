"""Chunking a resume into stable units for a token-limited encoder.

`bge-small-en-v1.5` truncates at 512 tokens; a real one-page resume measures
~1,150. Embedding the whole document as one string leaves the back half
invisible to the dense channel -- see docs/literature-survey.md §8.1, which
names this the largest correctness gap in the scorer and proposes exactly
this fix: embed per section or per entry, and aggregate.

Chunk identity is content-derived from enabled bullet ids (themselves
content-derived, see `document.Bullet._bullet_id`), not an object reference --
so a chunk's cached embedding is reusable across separate `ResumeDoc`
instances (every leave-one-out variant is a distinct in-memory document) as
long as the chunk's actual rendered content is unchanged. Toggling one bullet
changes only the one chunk it lives in; every other chunk's id is identical
to before, and its embedding is a cache hit.
"""
from __future__ import annotations

from .document import ResumeDoc


def document_chunks(doc: ResumeDoc) -> list[tuple[str, str]]:
    """(chunk_id, chunk_text) pairs: one per section's own content, plus one
    per entry inside it -- the units a resume is already organised into.
    Empty entries/sections (nothing currently enabled) contribute no chunk,
    matching `ResumeDoc.as_text()`'s own rule that an empty section must not
    pad the scored text.
    """
    chunks: list[tuple[str, str]] = []

    header_parts: list[str] = []
    if doc.header is not None:
        header_parts.append(doc.header.as_text())
    header_parts += [p for p in (doc.headline, doc.contact, doc.summary) if p]
    if header_parts:
        chunks.append(("header", "\n".join(p for p in header_parts if p)))

    for section in doc.sections:
        own_bullets = [b for b in section.bullets if b.enabled]
        body_text = section.body.as_text() if section.body is not None else ""
        if own_bullets or body_text or section.text:
            parts = [section.title]
            if section.text:
                parts.append(section.text)
            if body_text:
                parts.append(body_text)
            parts.extend(b.text for b in own_bullets)
            ids = ",".join(b.id for b in own_bullets)
            chunks.append((f"section:{section.title}:{ids}", "\n".join(parts)))

        for entry in section.entries:
            enabled = entry.enabled_bullets()
            if not enabled:
                continue
            parts = [entry.heading()]
            if entry.tech:
                parts.append("Tech: " + ", ".join(entry.tech))
            parts.extend(b.text for b in enabled)
            ids = ",".join(b.id for b in enabled)
            chunks.append((f"entry:{section.title}:{entry.title}:{ids}",
                          "\n".join(parts)))

    return chunks
