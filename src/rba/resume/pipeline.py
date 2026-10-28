"""Running the GitHub pipeline from a UI, with progress and a hard cap.

The CLI path runs the full LangGraph. A web request cannot: Groq's free tier is
8,000 tokens per minute, a measured full run took 1,433 s including 642 s and
616 s of rate-limit backoff, and Streamlit Community Cloud drops the websocket
after roughly 10-15 minutes. So this runs the same nodes with a repo cap small
enough to finish inside that window, and reports progress as it goes.
"""
from __future__ import annotations

import logging
from collections.abc import Callable

from ..config import Config
from ..github_tools import make_backend
from ..groq_client import GroqClient
from ..jev_client import JevClient
from ..nodes.fetch import fetch_facts
from ..nodes.select import select_repos
from ..nodes.summarize import summarize_repos
from ..nodes.verify import verify_claims
from .document import ResumeDoc, from_summaries

log = logging.getLogger(__name__)

Progress = Callable[[str, float], None]


def _noop(_message: str, _fraction: float) -> None:
    return None


async def build_from_github(username: str, *, github_token: str = "",
                            groq_key: str = "", jev_key: str = "",
                            max_repos: int = 6,
                            progress: Progress | None = None,
                            config: Config | None = None) -> tuple[ResumeDoc, dict]:
    """Public repos -> verified claims -> an editable resume document."""
    report = progress or _noop
    cfg = config or Config.load()
    cfg.raw["selection"]["max_repos"] = max_repos

    report("Connecting to GitHub…", 0.05)
    backend = await make_backend(cfg, token=github_token or None)
    stats: dict = {}
    try:
        repos = await backend.list_repos(username)
        public = [r for r in repos if not r.is_private]
        stats["repos_found"] = len(repos)
        stats["public"] = len(public)
        report(f"Found {len(public)} public repositories", 0.15)

        state: dict = {"username": username, "repos": public}
        # Jev is optional: without a key, select_repos falls back to a
        # heuristic rather than failing the whole import.
        jev = JevClient(cfg, api_key=jev_key or "") if jev_key else None
        try:
            state |= await select_repos(state, config=cfg, backend=backend,
                                        jev=jev)
        finally:
            if jev is not None:
                await jev.aclose()
        chosen = state.get("selected", [])
        stats["selected"] = len(chosen)
        report(f"Selected {len(chosen)} worth describing", 0.3)

        state |= await fetch_facts(state, backend=backend, config=cfg)
        report("Read READMEs, trees and commits", 0.45)
    finally:
        await backend.aclose()

    async with GroqClient(cfg, api_key=groq_key or None) as client:
        report("Summarising projects…", 0.55)
        state |= await summarize_repos(state, client=client, config=cfg)
        report("Verifying every claim against its evidence…", 0.85)
        state |= await verify_claims(state)
        usage = client.usage_report()
        stats["groq_tokens"] = sum(v["total"] for v in usage.values())

    claims = state.get("claims") or []
    stats["claims_total"] = len(claims)
    stats["claims_verified"] = sum(
        1 for c in claims if getattr(c.status, "value", "") == "verified")
    stats["claims_rejected"] = sum(
        1 for c in claims if getattr(c.status, "value", "") == "rejected")

    # Per-repo failures are collected rather than raised, so one dead
    # repository cannot lose the whole run. They still have to reach the
    # caller: an invalid Groq key fails every summary, and without this the
    # user gets an empty resume and no reason for it.
    stats["errors"] = list(state.get("errors") or [])
    stats["summarised"] = len(state.get("summaries") or {})

    doc = from_summaries(state.get("summaries") or {}, name=username,
                         headline="", contact=f"github.com/{username}")
    report("Done", 1.0)
    return doc, stats


def parse_resume_text(text: str, name: str = "") -> ResumeDoc:
    """Turn a pasted resume into an editable document, deterministically.

    No LLM. Lines that look like headings become sections, bullet-ish lines
    become bullets. Imperfect, but it never invents content and it is instant;
    the user is about to edit it anyway.
    """
    from .document import Bullet, Section

    sections: list[Section] = []
    current = Section(title="Experience")
    header: list[str] = []
    seen_bullet = False

    for raw in (text or "").splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped:
            continue
        is_bullet = stripped[0] in "-*•·—" or stripped[:2].rstrip(".").isdigit()
        looks_heading = (
            not is_bullet
            and len(stripped) < 60
            and (stripped.isupper() or stripped.endswith(":")
                 or stripped.istitle())
        )
        if is_bullet:
            seen_bullet = True
            current.bullets.append(
                Bullet(stripped.lstrip("-*•·— ").strip(), source_repo="uploaded"))
        elif looks_heading and seen_bullet:
            if current.bullets:
                sections.append(current)
            current = Section(title=stripped.rstrip(":"))
        elif seen_bullet:
            current.bullets.append(Bullet(stripped, source_repo="uploaded"))
        else:
            header.append(stripped)

    if current.bullets:
        sections.append(current)
    if not sections:
        sections = [Section("Experience", [
            Bullet(line.strip(), source_repo="uploaded")
            for line in (text or "").splitlines() if line.strip()])]

    return ResumeDoc(
        name=name or (header[0] if header else ""),
        headline=header[1] if len(header) > 1 else "",
        contact=next((h for h in header if "@" in h or "github" in h.lower()), ""),
        sections=sections)
