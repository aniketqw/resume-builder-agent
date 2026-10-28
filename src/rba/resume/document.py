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
    id: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            self.id = _bullet_id(self.text, self.source_repo)

    def to_dict(self) -> dict:
        return {
            "id": self.id, "text": self.text, "source_repo": self.source_repo,
            "evidence": self.evidence, "enabled": self.enabled,
            "verified": self.verified, "measure_hint": self.measure_hint,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> Bullet:
        return cls(**{k: v for k, v in payload.items()
                      if k in {"id", "text", "source_repo", "evidence",
                               "enabled", "verified", "measure_hint"}})


@dataclass
class Section:
    title: str
    bullets: list[Bullet] = field(default_factory=list)

    def enabled_bullets(self) -> list[Bullet]:
        return [b for b in self.bullets if b.enabled]

    def to_dict(self) -> dict:
        return {"title": self.title, "bullets": [b.to_dict() for b in self.bullets]}

    @classmethod
    def from_dict(cls, payload: dict) -> Section:
        return cls(title=payload.get("title", ""),
                   bullets=[Bullet.from_dict(b) for b in payload.get("bullets", [])])


@dataclass
class ResumeDoc:
    """A whole resume: header, optional narrative, and sections of bullets."""

    name: str = ""
    headline: str = ""
    contact: str = ""
    summary: str = ""
    sections: list[Section] = field(default_factory=list)

    # -- access -----------------------------------------------------------
    def all_bullets(self) -> list[Bullet]:
        return [b for section in self.sections for b in section.bullets]

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
        """The text that gets scored. Only enabled bullets appear."""
        parts: list[str] = [p for p in (self.headline, self.contact) if p]
        if include_summary and self.summary:
            parts.append(self.summary)
        for section in self.sections:
            bullets = section.enabled_bullets()
            if not bullets:
                continue          # an empty section should not pad the text
            parts.append(section.title)
            parts.extend(b.text for b in bullets)
        return "\n".join(parts)

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
            bullets = section.enabled_bullets()
            if not bullets:
                continue
            lines += ["", f"## {section.title}"]
            lines += [f"- {b.text}" for b in bullets]
        return "\n".join(lines)

    def skills(self) -> list[str]:
        """Skills a scorer may count: only those on enabled bullets."""
        from ..ats.normalise import normalise_all
        found: list[str] = []
        for bullet in self.enabled():
            found.extend(_skill_candidates(bullet.text))
        return normalise_all(found)

    # -- persistence ------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "name": self.name, "headline": self.headline, "contact": self.contact,
            "summary": self.summary,
            "sections": [s.to_dict() for s in self.sections],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=1)

    @classmethod
    def from_dict(cls, payload: dict) -> ResumeDoc:
        return cls(
            name=payload.get("name", ""), headline=payload.get("headline", ""),
            contact=payload.get("contact", ""), summary=payload.get("summary", ""),
            sections=[Section.from_dict(s) for s in payload.get("sections", [])],
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
