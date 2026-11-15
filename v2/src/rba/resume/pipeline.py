"""Running the GitHub pipeline from a browser request.

This used to hand-sequence four of the graph's nodes, because a full CLI run
measured 1,433 s -- 642 s and 616 s of it rate-limit backoff -- which a
browser request cannot sit through. That is still true, but the fix is not a
second workflow: it is stopping at the right node. The graph itself now
branches (see `graph.build_graph`), and the browser path is simply the graph
run with `full_run` unset, which skips the per-role resumes, the comparison
table, the plan and the report -- the seven extra Groq calls that caused the
1,433 s figure in the first place. So this function now *is* a graph driver:
same twelve nodes, same evidence, same verifier, just a smaller cap on how
many repos get summarised, and a shorter branch.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from ..config import Config
from ..preflight import blocking_failures, check_credentials
from .document import ResumeDoc

log = logging.getLogger(__name__)

Progress = Callable[[str, float], None]


@dataclass
class ImportResult:
    """Everything one import produced.

    This used to be ``tuple[ResumeDoc, dict]``, and the rest of `state` -- the
    repository facts, the summaries, which repos were chosen and which were
    passed over -- was dropped on the floor at the end of the function. Every
    one of those is needed downstream: `match_projects` needs `repos`,
    `build_evidence` needs `facts`, and "why is this repo not on my resume?"
    needs `excluded`. Recovering them meant a second pass over the GitHub API
    for data this function already had in hand.

    Unpacks like the old tuple, so existing `doc, stats = await ...` callers
    keep working.
    """

    doc: ResumeDoc
    stats: dict
    facts: dict = field(default_factory=dict)
    summaries: dict = field(default_factory=dict)
    claims: list = field(default_factory=list)
    repos: list = field(default_factory=list)
    selected: list = field(default_factory=list)
    excluded: list = field(default_factory=list)
    repo_graph: dict = field(default_factory=dict)

    def __iter__(self):
        return iter((self.doc, self.stats))


def _noop(_message: str, _fraction: float) -> None:
    return None


#: The message/fraction shown for each node as the graph streams past it.
#: The one place the import progress bar's wording lives. A node renamed in
#: `graph.py` without a matching update here is caught by
#: `test_graph.py::test_node_progress_covers_every_browser_branch_node`.
NODE_PROGRESS: dict[str, tuple[str, float]] = {
    "connect_github": ("Connecting to GitHub…", 0.05),
    "list_repos": ("Listing repositories…", 0.15),
    "select_repos": ("Selecting repositories worth describing…", 0.30),
    "fetch_facts": ("Reading READMEs, trees and commits…", 0.45),
    "prepare_evidence": ("Indexing code structure…", 0.50),
    "safety_scan": ("Scanning for secrets…", 0.55),
    "summarize_repo": ("Summarising projects…", 0.80),
    "verify_claims": ("Verifying every claim against its evidence…", 0.90),
    "build_document": ("Done", 1.0),
}


async def build_from_github(username: str, *, github_token: str = "",
                            groq_key: str = "", jev_key: str = "",
                            max_repos: int = 6, resume_text: str = "",
                            limiter=None, cache=None,
                            progress: Progress | None = None,
                            config: Config | None = None) -> ImportResult:
    """Public repos -> verified claims -> an editable resume document.

    Runs the same graph the CLI does, stopped at `build_document` (no
    `full_run` in the input) -- see the module docstring for why that is
    enough to fix the 1,433 s figure without a second workflow.
    """
    # Local import: `graph` imports `resume.document`, and this package's
    # `__init__.py` imports this module -- a module-level import here would
    # be a real cycle, not just a slow one. Reproduced directly: `from
    # rba.graph import Runner` as the first import in a fresh process raises
    # ImportError without this being deferred.
    from ..graph import Runner, build_graph

    report = progress or _noop
    cfg = config or Config.load()

    # Check credentials before spending anything, and outside the graph:
    # this must fail before a single node runs, not partway through one.
    # Discovering a bad key after ten repositories have been fetched -- with
    # cached ones masking it -- is how a stale key in the UI once looked
    # like a per-repository failure.
    report("Checking credentials…", 0.02)
    checks = await check_credentials(groq_key=groq_key or None,
                                     typesafe_key=jev_key or None,
                                     github_token=github_token or None,
                                     config=cfg)
    stats: dict = {"credentials": [c.line() for c in checks]}
    failed = blocking_failures(checks)
    if failed:
        raise RuntimeError("; ".join(c.line() for c in failed))

    runner = Runner(cfg, github_token=github_token, groq_key=groq_key,
                    jev_key=jev_key, limiter=limiter, cache=cache,
                    max_repos=max_repos)
    graph = build_graph(runner)
    final: dict = {}
    try:
        async for mode, chunk in graph.astream(
                {"username": username, "resume_text": resume_text},
                stream_mode=["updates", "values"]):
            if mode == "updates":
                for node in chunk:
                    if node in NODE_PROGRESS:
                        report(*NODE_PROGRESS[node])
            elif mode == "values":
                final = chunk
    finally:
        await runner.aclose()

    stats["repos_found"] = final.get("repos_found", 0)
    stats["public"] = len(final.get("repos") or [])
    stats["selected"] = len(final.get("selected") or [])
    if runner.client is not None:
        usage = runner.client.usage_report()
        stats["groq_tokens"] = sum(v["total"] for v in usage.values())

    claims = final.get("claims") or []
    stats["claims_total"] = len(claims)
    stats["claims_verified"] = sum(
        1 for c in claims if getattr(c.status, "value", "") == "verified")
    stats["claims_rejected"] = sum(
        1 for c in claims if getattr(c.status, "value", "") == "rejected")

    # Per-repo failures are collected rather than raised, so one dead
    # repository cannot lose the whole run. They still have to reach the
    # caller: an invalid Groq key fails every summary, and without this the
    # user gets an empty resume and no reason for it.
    stats["errors"] = list(final.get("errors") or [])
    stats["summarised"] = len(final.get("summaries") or {})

    public = final.get("repos") or []
    chosen = final.get("selected") or []
    return ImportResult(
        doc=final.get("document") or ResumeDoc(name=username),
        stats=stats,
        facts=dict(final.get("facts") or {}),
        summaries=dict(final.get("summaries") or {}),
        claims=list(claims),
        repos=list(public),
        selected=list(chosen),
        excluded=[r for r in public
                  if r.name not in {c.name for c in chosen}],
        repo_graph=dict(final.get("repo_graph") or {}),
    )


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
