"""The editable resume document.

Designed around one requirement: a user toggles a single bullet and we must
re-score *exactly*, regenerate the narrative without it, and still be able to
say which repository and which line the bullet came from.

That forces three properties:

* **Stable ids.** A bullet's id survives reordering, enabling and disabling, so
  the UI can address it and a selection can be stored. Ids are derived from
  content, so the same bullet regenerated from the same evidence keeps its id.
* **Evidence travels with the bullet.** Dropping provenance at the document
  boundary would make the resume unfalsifiable, which is the thing this whole
  project exists to avoid.
* **Enabled is a property of the bullet, not a separate selection list.** One
  source of truth; `as_text()` always reflects what the user actually sees.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field


def _bullet_id(text: str, source: str) -> str:
    """Content-derived id, so regenerating the same bullet keeps its identity."""
    digest = hashlib.sha256(f"{source}\x1f{text}".encode()).hexdigest()
    return digest[:12]


@dataclass
class Bullet:
    text: str
    source_repo: str = ""
    evidence: str = ""
    enabled: bool = True
    # Set by the extraction/verification pipeline. A bullet that failed
    # verification can still be shown, but it is marked, never silently kept.
    verified: bool = True
    measure_hint: str = ""
    # Where this line came from, which drives the provenance colouring and
    # lets a reader tell a measured fact from an assumed one:
    #   "yours"   -- parsed from the resume the user supplied
    #   "repo"    -- generated from repository evidence
    #   "assumed" -- contains something not traceable to either
    origin: str = "yours"
    id: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            self.id = _bullet_id(self.text, self.source_repo)

    def to_dict(self) -> dict:
        return {
            "id": self.id, "text": self.text, "source_repo": self.source_repo,
            "evidence": self.evidence, "enabled": self.enabled,
            "verified": self.verified, "measure_hint": self.measure_hint,
            "origin": self.origin,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> Bullet:
        return cls(**{k: v for k, v in payload.items()
                      if k in {"id", "text", "source_repo", "evidence",
                               "enabled", "verified", "measure_hint",
                               "origin"}})


@dataclass
class Entry:
    """One dated thing: an internship, a project, a position of responsibility.

    A real resume entry is not a bare list of bullets -- it carries a title, an
    organisation, dates and often a `Tech:` line, and a renderer that drops any
    of those produces a document the user does not recognise as theirs.
    """

    title: str = ""
    subtitle: str = ""        # "LLM Agents" / "Marvell Semiconductors"
    dates: str = ""           # "Oct 2026 - Present"
    tech: list[str] = field(default_factory=list)
    bullets: list[Bullet] = field(default_factory=list)

    def enabled_bullets(self) -> list[Bullet]:
        return [b for b in self.bullets if b.enabled]

    def heading(self) -> str:
        left = " - ".join(p for p in (self.title, self.subtitle) if p)
        return f"{left}  {self.dates}".strip() if self.dates else left

    def to_dict(self) -> dict:
        return {"title": self.title, "subtitle": self.subtitle,
                "dates": self.dates, "tech": list(self.tech),
                "bullets": [b.to_dict() for b in self.bullets]}

    @classmethod
    def from_dict(cls, payload: dict) -> Entry:
        return cls(title=payload.get("title", ""),
                   subtitle=payload.get("subtitle", ""),
                   dates=payload.get("dates", ""),
                   tech=list(payload.get("tech", [])),
                   bullets=[Bullet.from_dict(b) for b in payload.get("bullets", [])])


@dataclass
class Table:
    """A grid, as in ACADEMIC DETAILS and CERTIFICATIONS."""

    headers: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)

    def as_text(self) -> str:
        return "\n".join(" ".join(c for c in row if c) for row in self.rows)

    def to_dict(self) -> dict:
        return {"kind": "table", "headers": list(self.headers),
                "rows": [list(r) for r in self.rows]}

    @classmethod
    def from_dict(cls, payload: dict) -> Table:
        return cls(headers=list(payload.get("headers", [])),
                   rows=[list(r) for r in payload.get("rows", [])])


@dataclass
class KeyValues:
    """Label-to-value rows, as in TECHNICAL SKILLS."""

    rows: list[tuple[str, str]] = field(default_factory=list)

    def as_text(self) -> str:
        return "\n".join(f"{k} {v}".strip() for k, v in self.rows)

    def to_dict(self) -> dict:
        return {"kind": "keyvalues", "rows": [[k, v] for k, v in self.rows]}

    @classmethod
    def from_dict(cls, payload: dict) -> KeyValues:
        return cls(rows=[(r[0], r[1]) for r in payload.get("rows", []) if len(r) >= 2])


@dataclass
class Header:
    """The block above the first heading: name plus labelled contact fields."""

    name: str = ""
    fields: list[tuple[str, str]] = field(default_factory=list)

    def as_text(self) -> str:
        parts = [self.name] if self.name else []
        parts += [f"{k} : {v}".strip(" :") for k, v in self.fields]
        return "\n".join(parts)

    def to_dict(self) -> dict:
        return {"name": self.name, "fields": [[k, v] for k, v in self.fields]}

    @classmethod
    def from_dict(cls, payload: dict) -> Header:
        return cls(name=payload.get("name", ""),
                   fields=[(f[0], f[1]) for f in payload.get("fields", [])
                           if len(f) >= 2])


def _body_from_dict(payload: dict | None) -> Table | KeyValues | None:
    if not payload:
        return None
    if payload.get("kind") == "table":
        return Table.from_dict(payload)
    if payload.get("kind") == "keyvalues":
        return KeyValues.from_dict(payload)
    return None


@dataclass
class Section:
    """A top-level heading.

    Three shapes, and a section uses whichever ones it needs:
    `bullets` for a plain list, `entries` for dated items, `body` for a table
    or a set of label-value rows. Keeping `bullets` first and defaulted means
    every `Section(title, bullets)` call written before this existed still
    works unchanged.
    """

    title: str
    bullets: list[Bullet] = field(default_factory=list)
    entries: list[Entry] = field(default_factory=list)
    body: Table | KeyValues | None = None
    text: str = ""            # free prose, as under SUMMARY

    def enabled_bullets(self) -> list[Bullet]:
        own = [b for b in self.bullets if b.enabled]
        return own + [b for e in self.entries for b in e.enabled_bullets()]

    def all_bullets(self) -> list[Bullet]:
        return list(self.bullets) + [b for e in self.entries for b in e.bullets]

    def is_empty(self) -> bool:
        return not (self.enabled_bullets() or self.body or self.text)

    def to_dict(self) -> dict:
        out: dict = {"title": self.title,
                     "bullets": [b.to_dict() for b in self.bullets]}
        if self.entries:
            out["entries"] = [e.to_dict() for e in self.entries]
        if self.body is not None:
            out["body"] = self.body.to_dict()
        if self.text:
            out["text"] = self.text
        return out

    @classmethod
    def from_dict(cls, payload: dict) -> Section:
        return cls(title=payload.get("title", ""),
                   bullets=[Bullet.from_dict(b) for b in payload.get("bullets", [])],
                   entries=[Entry.from_dict(e) for e in payload.get("entries", [])],
                   body=_body_from_dict(payload.get("body")),
                   text=payload.get("text", ""))


@dataclass
class ResumeDoc:
    """A whole resume: header, optional narrative, and sections of bullets."""

    name: str = ""
    headline: str = ""
    contact: str = ""
    summary: str = ""
    sections: list[Section] = field(default_factory=list)
    # The labelled contact block of a structured resume (Course / Email /
    # Mobile / CGPA / Links). `contact` stays for documents parsed the flat way.
    header: Header | None = None

    # -- access -----------------------------------------------------------
    def all_bullets(self) -> list[Bullet]:
        return [b for section in self.sections for b in section.all_bullets()]

    def enabled(self) -> list[Bullet]:
        return [b for b in self.all_bullets() if b.enabled]

    def by_id(self, bullet_id: str) -> Bullet | None:
        for bullet in self.all_bullets():
            if bullet.id == bullet_id:
                return bullet
        return None

    def toggle(self, bullet_id: str, enabled: bool | None = None) -> bool:
        """Enable/disable one bullet. Returns the new state."""
        bullet = self.by_id(bullet_id)
        if bullet is None:
            raise KeyError(f"no bullet {bullet_id!r}")
        bullet.enabled = (not bullet.enabled) if enabled is None else enabled
        return bullet.enabled

    # -- rendering --------------------------------------------------------
    def as_text(self, include_summary: bool = True) -> str:
        """The text that gets scored. Only enabled bullets appear.

        Tables, key-value rows and the header block are included: a resume's
        skills and education live there, and omitting them would quietly drop
        every one of those terms out of the keyword and skill channels.
        """
        parts: list[str] = []
        if self.header is not None:
            parts.append(self.header.as_text())
        parts += [p for p in (self.headline, self.contact) if p]
        if include_summary and self.summary:
            parts.append(self.summary)
        for section in self.sections:
            bullets = section.enabled_bullets()
            body_text = section.body.as_text() if section.body is not None else ""
            if not (bullets or body_text or section.text):
                continue          # an empty section should not pad the text
            parts.append(section.title)
            if section.text:
                parts.append(section.text)
            if body_text:
                parts.append(body_text)
            for entry in section.entries:
                entry_bullets = entry.enabled_bullets()
                if not entry_bullets:
                    continue
                parts.append(entry.heading())
                if entry.tech:
                    parts.append("Tech: " + ", ".join(entry.tech))
                parts.extend(b.text for b in entry_bullets)
            parts.extend(b.text for b in section.bullets if b.enabled)
        return "\n".join(p for p in parts if p)

    def as_markdown(self) -> str:
        lines: list[str] = []
        if self.name:
            lines.append(f"# {self.name}")
        if self.headline:
            lines.append(f"**{self.headline}**")
        if self.contact:
            lines.append(self.contact)
        if self.summary:
            lines += ["", self.summary]
        for section in self.sections:
            if section.is_empty():
                continue
            lines += ["", f"## {section.title}"]
            if section.text:
                lines += [section.text]
            if isinstance(section.body, KeyValues):
                lines += [f"- **{k}** — {v}" for k, v in section.body.rows]
            elif isinstance(section.body, Table):
                if section.body.headers:
                    lines += ["", "| " + " | ".join(section.body.headers) + " |",
                              "|" + "---|" * len(section.body.headers)]
                    lines += ["| " + " | ".join(r) + " |" for r in section.body.rows]
                else:
                    lines += [f"- {' — '.join(c for c in r if c)}"
                              for r in section.body.rows]
            for entry in section.entries:
                bullets = entry.enabled_bullets()
                if not bullets:
                    continue
                lines += ["", f"### {entry.heading()}"]
                if entry.tech:
                    lines += [f"*Tech: {', '.join(entry.tech)}*"]
                lines += [f"- {b.text}" for b in bullets]
            lines += [f"- {b.text}" for b in section.bullets if b.enabled]
        return "\n".join(lines)

    def skills(self) -> list[str]:
        """Skills a scorer may count: only those on enabled bullets."""
        from ..ats.normalise import normalise_all
        found: list[str] = []
        for bullet in self.enabled():
            found.extend(_skill_candidates(bullet.text))
        # A TECHNICAL SKILLS block is a declaration of skills; ignoring it
        # would under-count the very channel it exists to feed.
        for section in self.sections:
            if isinstance(section.body, KeyValues):
                for _label, value in section.body.rows:
                    found.extend(_skill_candidates(value))
        return normalise_all(found)

    # -- persistence ------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "name": self.name, "headline": self.headline, "contact": self.contact,
            "summary": self.summary,
            "sections": [s.to_dict() for s in self.sections],
            "header": self.header.to_dict() if self.header else None,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=1)

    @classmethod
    def from_dict(cls, payload: dict) -> ResumeDoc:
        return cls(
            name=payload.get("name", ""), headline=payload.get("headline", ""),
            contact=payload.get("contact", ""), summary=payload.get("summary", ""),
            sections=[Section.from_dict(s) for s in payload.get("sections", [])],
            header=(Header.from_dict(payload["header"])
                    if payload.get("header") else None),
        )

    @classmethod
    def from_json(cls, text: str) -> ResumeDoc:
        return cls.from_dict(json.loads(text))


def _skill_candidates(text: str) -> list[str]:
    """Known skill names appearing in a bullet.

    Only aliases we already recognise. Guessing new skill names from prose is
    exactly the inflation the verification step exists to stop.
    """
    from ..ats.normalise import ALIASES, canonical

    lowered = f" {text.lower()} "
    hits: list[str] = []
    for canon, aliases in ALIASES.items():
        for surface in (canon, *aliases):
            if f" {surface} " in lowered or f" {surface}," in lowered:
                hits.append(canonical(canon))
                break
    return hits


def from_summaries(summaries, name: str = "", headline: str = "",
                   contact: str = "", summary: str = "") -> ResumeDoc:
    """Club per-repository summaries into one resume.

    One section per project, preserving which repository each bullet came from
    and whether it survived verification.
    """
    sections: list[Section] = []
    for repo, repo_summary in (summaries or {}).items():
        bullets: list[Bullet] = []
        for claim in getattr(repo_summary, "claims", []):
            status = getattr(claim.status, "value", str(claim.status))
            if status == "rejected":
                continue          # refused claims never enter the document
            evidence = claim.evidence[0].quote if claim.evidence else ""
            bullets.append(Bullet(
                text=claim.text, source_repo=repo, evidence=evidence,
                verified=status == "verified",
                measure_hint=getattr(claim, "measure_hint", ""),
            ))
        if bullets:
            sections.append(Section(title=repo, bullets=bullets))
    return ResumeDoc(name=name, headline=headline, contact=contact,
                     summary=summary, sections=sections)
