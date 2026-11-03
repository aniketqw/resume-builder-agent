"""Report rendering: Markdown, LaTeX, and PDF when a TeX toolchain exists.

The report leads with provenance, not with the resume. Anyone reading it should
be able to see which claims are proven, which are only measurable, and what the
agent refused to say — that refusal list is the most informative part.
"""
from __future__ import annotations

import logging
import re
import shutil
import subprocess
from pathlib import Path

from .state import ClaimStatus

log = logging.getLogger(__name__)

LATEX_ESCAPES = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
                 "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
                 "^": r"\textasciicircum{}", "\\": r"\textbackslash{}"}


def tex_escape(text: str) -> str:
    return "".join(LATEX_ESCAPES.get(ch, ch) for ch in text)


def render_markdown(state: dict) -> str:
    resumes = state.get("resumes") or {}
    scores = state.get("scores") or {}
    comparison = state.get("comparison") or {}
    plan = state.get("plan") or []
    claims = state.get("claims") or []
    safety = state.get("safety") or []
    usage = state.get("usage") or {}
    summaries = state.get("summaries") or {}

    verified = [c for c in claims if c.status is ClaimStatus.VERIFIED]
    needs = [c for c in claims if c.status is ClaimStatus.NEEDS_MEASURE]
    rejected = [c for c in claims if c.status is ClaimStatus.REJECTED]

    out: list[str] = []
    w = out.append
    w(f"# Resume report — {state.get('username', 'unknown')}\n")
    w(f"Built from {len(summaries)} repositories. "
      f"**{len(verified)} claims verified**, {len(needs)} need measurement, "
      f"{len(rejected)} rejected.\n")

    if safety:
        w("\n## Security findings in your repositories\n")
        w("These were found while reading your repositories. Fix these first.\n")
        for f in safety:
            w(f"- **[{f.severity}]** `{f.repo}/{f.path}` — {f.detail}")
        w("")

    if comparison.get("rows"):
        w("\n## Role comparison\n")
        w("| Role | ATS score | Keywords | Role fit | Numbers | Recoverable |")
        w("|---|---:|---:|---:|---:|---:|")
        for r in comparison["rows"]:
            c = r["components"]
            w(f"| {r['title']} | **{r['total']}** | {c.get('keywords', 0)} | "
              f"{c.get('role_fit', 0)} | {c.get('numbers', 0)} | +{r['potential_gain']} |")
        w(f"\nBest fit: **{comparison.get('best_role', '')}** "
          f"(spread across roles: {comparison.get('spread', 0)} points).")
        if comparison.get("rationale"):
            w(f"\n{comparison['rationale']}")
        w("")

    for key, resume in resumes.items():
        s = scores.get(key)
        w(f"\n## Resume — {key}" + (f" ({s.total}/100)" if s else "") + "\n")
        w(f"**{resume.headline}**\n")
        if resume.summary:
            w(f"{resume.summary}\n")
        for section, bullets in resume.sections.items():
            if section == "summary" or not bullets:
                continue
            w(f"### {section.title()}\n")
            for b in bullets:
                w(f"- {b}")
            w("")

    if plan:
        w("\n## Improvement plan\n")
        total_hours = sum(s.hours for s in plan)
        total_gain = sum(s.ats_gain for s in plan)
        w(f"{len(plan)} steps, ~{total_hours:.0f} hours, "
          f"+{total_gain:.1f} ATS points if all are completed.\n")
        for i, s in enumerate(plan, 1):
            w(f"### {i}. {s.title}  \n"
              f"*{s.hours:.0f}h · +{s.ats_gain} ATS · role: {s.role}"
              + (f" · repo: `{s.repo}`" if s.repo else " · new work") + "*\n")
            w(f"- **Gap:** {s.gap}")
            w(f"- **Evidence found:** {s.evidence_found}")
            w(f"- **Action:** {s.action}")
            w(f"- **Done when:** {s.measurable_output}")
            w("")

    if needs:
        w("\n## Measure these\n")
        w("Each of these is plausible but unproven. The agent refused to put a "
          "number on it; run the command and the number becomes claimable.\n")
        for c in needs:
            w(f"- `{c.repo}` — {c.text}")
            w(f"  - *why:* {c.reason}")
            w(f"  - **`{c.measure_hint}`**")
        w("")

    if rejected:
        w("\n## Refused claims\n")
        w("The model proposed these; the evidence did not support them. "
          "They are listed so you can see what was filtered, not hidden.\n")
        for c in rejected:
            w(f"- `{c.repo}` — {c.text}  \n  *{c.reason}*")
        w("")

    w("\n## Evidence index\n")
    for repo, summary in summaries.items():
        ok = [c for c in summary.claims if c.status is ClaimStatus.VERIFIED]
        if not ok:
            continue
        w(f"\n### {repo}\n")
        w(f"{summary.one_line}\n")
        for c in ok:
            src = c.evidence[0] if c.evidence else None
            w(f"- {c.text}")
            if src:
                quote = src.quote.replace("\n", " ")[:160]
                w(f"  - `{src.path}`: \"{quote}\"")
        w("")

    if usage:
        w("\n## Run cost\n")
        groq = usage.get("groq") or {}
        jev = usage.get("jev") or {}
        if groq:
            w(f"- Groq: {groq.get('total', 0)} tokens across "
              f"{groq.get('calls', 0)} calls")
        if jev:
            w(f"- Jev: {jev.get('input_tokens', 0)} input tokens, "
              f"${jev.get('cost_usd', 0):.5f} (output tokens are free)")
        if usage.get("cache_hit_rate") is not None:
            w(f"- Cache hit rate: {usage['cache_hit_rate']:.0%}")
        if usage.get("elapsed_s"):
            w(f"- Wall clock: {usage['elapsed_s']:.0f}s")
    return "\n".join(out) + "\n"


LATEX_PREAMBLE = r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=0.75in]{geometry}
\usepackage{enumitem}
\usepackage[hidelinks]{hyperref}
\setlist[itemize]{leftmargin=*,topsep=2pt,itemsep=1pt}
\pagestyle{empty}
\begin{document}
"""


def render_latex(state: dict, role: str) -> str:
    resume = (state.get("resumes") or {}).get(role)
    if resume is None:
        raise KeyError(f"no resume for role {role!r}")
    contact = state.get("contact", "")
    parts = [LATEX_PREAMBLE,
             r"\begin{center}{\LARGE\bfseries " + tex_escape(resume.headline) + "}\\\\[4pt]",
             tex_escape(contact) + r"\end{center}", ""]
    if resume.summary:
        parts += [r"\section*{Summary}", tex_escape(resume.summary), ""]
    for section, bullets in resume.sections.items():
        if section == "summary" or not bullets:
            continue
        parts.append(r"\section*{" + tex_escape(section.title()) + "}")
        parts.append(r"\begin{itemize}")
        parts += [r"  \item " + tex_escape(b) for b in bullets]
        parts.append(r"\end{itemize}")
        parts.append("")
    parts.append(r"\end{document}")
    return "\n".join(parts)


def write_reports(state: dict, out_dir: str | Path = "out") -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    md = out / "report.md"
    md.write_text(render_markdown(state))
    written["markdown"] = str(md)

    for role in (state.get("resumes") or {}):
        tex = out / f"resume_{role}.tex"
        tex.write_text(render_latex(state, role))
        written[f"latex_{role}"] = str(tex)
        pdf = compile_pdf(tex)
        if pdf:
            written[f"pdf_{role}"] = pdf
    return written


def compile_pdf(tex_path: Path) -> str | None:
    """Compile with pdflatex if it exists. Absence is normal, not an error."""
    engine = shutil.which("pdflatex") or shutil.which("tectonic")
    if not engine:
        log.info("no LaTeX engine found; wrote %s without a PDF "
                 "(install MacTeX/TeX Live for PDFs)", tex_path.name)
        return None
    try:
        subprocess.run(
            [engine, "-interaction=nonstopmode", "-halt-on-error",
             f"-output-directory={tex_path.parent}", str(tex_path)],
            capture_output=True, check=True, timeout=120)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        log.warning("LaTeX failed for %s: %s", tex_path.name, exc)
        return None
    pdf = tex_path.with_suffix(".pdf")
    for junk in (".aux", ".log", ".out"):
        tex_path.with_suffix(junk).unlink(missing_ok=True)
    return str(pdf) if pdf.exists() else None


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
