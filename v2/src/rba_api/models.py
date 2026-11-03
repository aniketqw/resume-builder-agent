"""Wire shapes.

`ResumeDoc` already round-trips through `to_dict`/`from_dict`, so the document
itself travels as a plain object rather than being re-declared here -- a second
declaration would be one more thing to drift.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Target(BaseModel):
    """What the resume is being scored against."""

    jd_text: str = ""
    role_key: str = ""
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)


class ParseRequest(BaseModel):
    text: str
    name: str = ""


class OpModel(BaseModel):
    op: Literal["enable", "disable", "set_text", "add_bullet", "remove_bullet"]
    bullet_id: str = ""
    entry_key: str = ""
    text: str = ""


class PatchRequest(BaseModel):
    ops: list[OpModel] = Field(default_factory=list)
    if_match: int | None = None


class ScoreRequest(BaseModel):
    """Either a session or an inline document.

    The inline path is not only for tests: it keeps the scoring endpoints
    honest as pure functions of their input, which is what lets the exactness
    suite compare them against `rescore` directly.
    """

    session: str = ""
    document: dict | None = None
    target: Target | None = None
    use_dense: bool = True
    explain: bool = False


class VariantSpec(BaseModel):
    key: str
    ops: list[OpModel] = Field(default_factory=list)


class BatchScoreRequest(BaseModel):
    session: str = ""
    document: dict | None = None
    target: Target | None = None
    use_dense: bool = True
    base_ops: list[OpModel] = Field(default_factory=list)
    variants: list[VariantSpec] = Field(default_factory=list)


class ScoreBlock(BaseModel):
    raw: float
    score_100: float
    band: str
    channels: dict[str, float]
    weights_used: dict[str, float] = Field(default_factory=dict)


class VariantScore(BaseModel):
    key: str
    raw: float
    score_100: float
    delta: float


class BatchScoreResponse(BaseModel):
    base: ScoreBlock
    variants: list[VariantScore]
    stats: dict[str, Any]
    signature: str = ""
    rev: int = 0
    delta_basis: Literal["raw"] = "raw"


class RenderRequest(BaseModel):
    session: str = ""
    document: dict | None = None
    style: Literal["bits", "clean"] = "bits"
    show_numbers: bool = True
    colour_provenance: bool = False
    include_disabled: bool = False
    max_pages: int = 1
