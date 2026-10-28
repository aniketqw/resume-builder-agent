"""Streamlit entry point. Run from the repository root.

Start from the resume you already have, pull candidate lines out of your
GitHub projects, and watch a grounded score move as you accept or reject them.

Two things make this different from the usual resume scorer:

* **The +/- beside each line is exact.** It is leave-one-out: the score
  genuinely recomputed without that line. Affordable only because scoring makes
  no LLM call -- ONNX embeddings and arithmetic. Measured on a 662-word
  one-page resume: 293 ms for one re-score, 4.8 s for all fifteen deltas.
  So the score moves on every click; the per-line figures refresh on request.
* **Nothing is invented.** Generated lines arrive switched off and carry the
  repository evidence they came from. Your own resume is the baseline and is
  never rewritten.
"""
from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from rba.ats.fusion import CHANNEL_WEIGHTS  # noqa: E402
from rba.config import Config  # noqa: E402
from rba.groq_client import GroqClient  # noqa: E402
from rba.nodes.variants import generate_variants  # noqa: E402
from rba.resume import (  # noqa: E402
    ResumeDoc,
    attribute,
    build_from_github,
    rescore,
)
from rba.resume.diagnostics import diagnose  # noqa: E402
from rba.resume.ingest import parse_resume  # noqa: E402
from rba.resume.merge import baseline, candidates, merge_candidates  # noqa: E402
from rba.resume.render import (  # noqa: E402
    RenderOptions,
    to_html,
    to_latex,
    to_pdf,
    to_plain,
)
from rba.resume.render.options import ORIGIN_LABELS  # noqa: E402
from rba.resume.render.pdf import pages_used  # noqa: E402
from rba.scoring import load_roles  # noqa: E402

st.set_page_config(page_title="Resume Builder — evidence-grounded",
                   page_icon="📄", layout="wide")

SAMPLE_JD = """AI Engineer

Required: Python, LangChain, LangGraph, RAG, evaluation, Docker
Preferred: AWS, Kafka, vLLM, fine-tuning

Build and evaluate LLM agent systems end to end: retrieval, tool use,
guardrails, and measurement of cost and quality."""

STATUS_ICON = {"ok": "\u2705", "warn": "\u26a0\ufe0f", "info": "\u2139\ufe0f"}


def state():
    st.session_state.setdefault("doc", None)
    st.session_state.setdefault("stats", {})
    st.session_state.setdefault("attribution", None)
    st.session_state.setdefault("baseline_score", None)
    st.session_state.setdefault("attr_signature", None)
    st.session_state.setdefault("variants", None)
    return st.session_state


# ----------------------------------------------------------------- inputs --
def sidebar() -> dict:
    st.sidebar.header("1 · Your resume")
    uploaded = st.sidebar.file_uploader(
        "Upload a PDF, .txt or .md", type=["pdf", "txt", "md"],
        help="A BITS-format PDF is parsed into its tables and dated entries. "
             "Any other layout falls back to a simpler parser.")
    pasted = st.sidebar.text_area("…or paste it", height=120)

    st.sidebar.header("2 · Add from GitHub")
    use_github = st.sidebar.checkbox(
        "Pull candidate lines from my repositories", value=False,
        help="Generated lines arrive switched off. Nothing is added to your "
             "resume until you turn it on.")
    config: dict = {"uploaded": uploaded, "pasted": pasted,
                    "use_github": use_github}
    if use_github:
        config["username"] = st.sidebar.text_input("GitHub username", "aniketqw")
        config["github_token"] = st.sidebar.text_input(
            "GitHub token", type="password",
            help="Without one GitHub allows 60 requests an hour, shared.")
        config["groq_key"] = st.sidebar.text_input("Groq API key", type="password")
        config["max_repos"] = st.sidebar.slider(
            "Repositories to read", 1, 25, 6,
            help="Measured at roughly 3,000 tokens and one request per "
                 "repository. See evals/budget.md.")

    st.sidebar.header("3 · Target")
    roles = load_roles()
    mode = st.sidebar.radio("Target", ["Paste a job description", "Pick a role"],
                            label_visibility="collapsed")
    if mode == "Paste a job description":
        config["jd"] = st.sidebar.text_area("Job description", SAMPLE_JD, height=180)
        config["required"] = st.sidebar.text_input("Required skills (comma separated)")
        config["preferred"] = st.sidebar.text_input("Preferred skills")
        config["role_key"] = st.sidebar.selectbox(
            "Role profile for the advice panel", list(roles),
            format_func=lambda k: roles[k].title)
    else:
        key = st.sidebar.selectbox("Role", list(roles),
                                   format_func=lambda k: roles[k].title)
        role = roles[key]
        top = [t for t, _ in sorted(role.keywords.items(),
                                    key=lambda kv: -kv[1])]
        config["role_key"] = key
        config["jd"] = f"{role.title}. {role.headline}. " + ", ".join(top[:18])
        config["required"] = ", ".join(top[:8])
        config["preferred"] = ", ".join(top[8:14])
        st.sidebar.caption(
            "No job description, so this uses the role's keyword profile as a "
            "stand-in. A real posting scores more meaningfully.")
    return config


# Credential-shaped text must never reach the screen: this app is public.
_TOKENISH = re.compile(r"\b(?:gsk|ghp|gho|hf|lsv2_pt)[_-][A-Za-z0-9_-]+")


def _diagnose(exc: Exception) -> str:
    """Turn an exception into something the user can act on.

    The first version printed only `type(exc).__name__`, so a missing GitHub
    token surfaced as a bare "Import failed: RuntimeError" -- true, and useless.
    """
    message = _TOKENISH.sub("[redacted]", str(exc))
    lowered = message.lower()
    if "github_personal_access_token" in lowered or "github token" in lowered:
        return ("No GitHub token reached the importer. Paste a token in the "
                "sidebar — unauthenticated GitHub allows only 60 requests an "
                "hour, shared across everyone using this app.")
    if "groq_api_key" in lowered:
        return "Paste a Groq API key in the sidebar; summarising needs one."
    if "401" in message or "bad credentials" in lowered:
        return ("GitHub rejected the token. Check it has not expired and that "
                "it carries at least `public_repo` scope.")
    if "403" in message or "rate limit" in lowered:
        return ("GitHub rate-limited the request. With a token you get 5,000 "
                "requests an hour; without one, 60.")
    if "404" in message or "not found" in lowered:
        return "That GitHub username does not exist, or has no public repositories."
    if "429" in message:
        return ("Groq rate-limited the request. The free tier allows 8,000 "
                "tokens a minute — try fewer repositories.")
    return f"Import failed: {message[:200]}"


def split(value: str) -> list[str]:
    return [part.strip() for part in (value or "").split(",") if part.strip()]


# ------------------------------------------------------------------ build --
def build(config: dict) -> None:
    store = state()
    raw: bytes | str = ""
    if config.get("uploaded") is not None:
        raw = config["uploaded"].read()
    elif config.get("pasted", "").strip():
        raw = config["pasted"]

    if raw:
        store.doc = parse_resume(raw)
    elif not config.get("use_github"):
        st.error("Upload or paste a resume, or tick **Add from GitHub**.")
        return
    else:
        store.doc = ResumeDoc()
    store.stats = {}

    if config.get("use_github"):
        if not config.get("username"):
            st.error("Enter a GitHub username.")
            return
        bar = st.progress(0.0, text="Starting…")

        def report(message: str, fraction: float) -> None:
            bar.progress(min(max(fraction, 0.0), 1.0), text=message)

        try:
            generated, stats = asyncio.run(build_from_github(
                config["username"], github_token=config.get("github_token", ""),
                groq_key=config.get("groq_key", ""),
                max_repos=config.get("max_repos", 6), progress=report))
        except Exception as exc:
            bar.empty()
            st.error(_diagnose(exc))
            with st.expander("Technical detail"):
                st.code(f"{type(exc).__name__}: {exc}"[:800])
            return
        bar.empty()
        _report_partial_failures(stats)
        if not store.doc.sections and not store.doc.name:
            # Nothing to merge into: the generated document is the resume.
            store.doc = generated
            for bullet in store.doc.all_bullets():
                bullet.origin = "repo"
        else:
            merge_candidates(store.doc, generated)
        store.stats = stats
    store.attribution = None
    store.baseline_score = None


def _report_partial_failures(stats: dict) -> None:
    """Say what went wrong per repository instead of returning a bare resume.

    The failure that prompted this: an expired Groq key made every summary
    return 401, and the app showed an empty document with no explanation.
    """
    errors = stats.get("errors") or []
    if not errors:
        return
    joined = " ".join(errors).lower()
    if "invalid_api_key" in joined or "invalid api key" in joined or "401" in joined:
        st.error("Groq rejected the API key, so no repository could be "
                 "summarised. Paste a current key in the sidebar — keys are "
                 "revoked when they leak into a public repository.")
    elif not stats.get("summarised"):
        st.error("No repository could be summarised. The detail is below.")
    else:
        st.warning(f"{len(errors)} repository/repositories could not be read; "
                   f"the rest were summarised.")
    with st.expander(f"Detail ({len(errors)})"):
        for line in errors[:20]:
            st.code(_TOKENISH.sub("[redacted]", line)[:300], language="text")


# ------------------------------------------------------------- scoring ----
def _signature(doc: ResumeDoc) -> tuple:
    """What the deltas were computed against.

    Per-point deltas stay valid only while the set of enabled lines and their
    wording are unchanged. Comparing this to the live document is how the UI
    knows to label them as needing a refresh rather than quietly showing
    numbers that describe a document the user has already edited away from.
    """
    return tuple((b.id, b.text, b.enabled) for b in doc.all_bullets())


def sync_from_widgets(doc: ResumeDoc) -> None:
    """Apply the widget state to the document before anything reads it.

    Streamlit widgets keep their value in `session_state` under their key, and
    `editor()` only copies that back onto the bullets when it runs -- which is
    *after* the score is computed. Without this the score was always one click
    behind: untick a line and nothing moved until the next interaction.
    """
    for bullet in doc.all_bullets():
        enabled = st.session_state.get(f"on_{bullet.id}")
        if enabled is not None:
            bullet.enabled = enabled
        text = st.session_state.get(f"tx_{bullet.id}")
        if text is not None and text != bullet.text:
            bullet.text = text


def live_score(config: dict) -> tuple[float, dict]:
    """The exact score of the document as it stands. One embedding, ~293 ms."""
    store = state()
    raw, channels = rescore(store.doc, config["jd"],
                            required_skills=split(config.get("required", "")),
                            preferred_skills=split(config.get("preferred", "")))
    return raw * 100, channels


def refresh_deltas(config: dict) -> None:
    """Recompute every per-point delta. ~4.8 s, so never on a toggle."""
    store = state()
    store.attribution = attribute(
        store.doc, config["jd"],
        required_skills=split(config.get("required", "")),
        preferred_skills=split(config.get("preferred", "")))
    store.attr_signature = _signature(store.doc)


def recompute(config: dict) -> None:
    """Full refresh: deltas, and the baseline the headline delta compares to."""
    store = state()
    if store.doc is None:
        return
    with st.spinner("Scoring every line…"):
        refresh_deltas(config)
        base = baseline(store.doc)
        raw, _ = rescore(base, config["jd"],
                         required_skills=split(config.get("required", "")),
                         preferred_skills=split(config.get("preferred", "")))
        store.baseline_score = raw * 100 if base.enabled() else None


# ------------------------------------------------------------- left rail --
def score_panel(config: dict, score: float, channels: dict) -> None:
    store = state()
    att = store.attribution

    st.metric("Match score", f"{score:.1f}",
              delta=(None if store.baseline_score is None
                     else f"{score - store.baseline_score:+.1f} vs your "
                          f"uploaded resume ({store.baseline_score:.0f})"))
    st.caption("Recomputed in full on every click — not an estimate. "
               "Fused across three channels; not an ATS number.")

    for name, value in (channels or {}).items():
        st.progress(min(max(value, 0.0), 1.0), text=f"{name} · {value:.2f}")

    if att and att.harmful():
        st.markdown("##### Lines that are costing you")
        st.caption("Removing these would *raise* the score against this job.")
        for impact in att.harmful()[:4]:
            st.markdown(f"- :red[{impact.label}] {impact.text[:80]}…")

    st.markdown("##### Advice")
    st.caption("Not score components. The seven weighted bars this replaces "
               "were invented; these are checks you can act on.")
    role = load_roles()[config["role_key"]]
    for finding in diagnose(store.doc, role):
        icon = STATUS_ICON.get(finding.status, "·")
        with st.expander(f"{icon} {finding.label}", expanded=False):
            if finding.detail:
                st.caption(finding.detail)
            if finding.items:
                st.write(", ".join(finding.items[:30])
                         if len(" ".join(finding.items)) < 400
                         else "\n".join(f"- {i}" for i in finding.items[:8]))

    if store.stats.get("credentials"):
        with st.expander("Credentials"):
            for line in store.stats["credentials"]:
                st.caption(line)


# ------------------------------------------------------------ right rail --
def render_panel(config: dict) -> None:
    store = state()
    doc: ResumeDoc = store.doc

    controls = st.columns([1.3, 1, 1])
    style = controls[0].radio("Format", ["BITS", "Clean"], horizontal=True,
                              label_visibility="collapsed", key="fmt")
    numbers = controls[1].toggle("Numbers", value=True, key="nums",
                                 help="Hide every figure to check the resume "
                                      "still reads without them.")
    provenance = controls[2].toggle("Provenance", value=False, key="prov",
                                    help="Colour each line by where it came from.")
    opts = RenderOptions(style=style.lower(), show_numbers=numbers,
                         colour_provenance=provenance)

    if provenance:
        st.caption(" · ".join(
            f":{'green' if k == 'yours' else 'gray' if k == 'repo' else 'orange'}[{v}]"
            for k, v in ORIGIN_LABELS.items()))

    pdf = to_pdf(doc, opts)
    pages = pages_used(pdf)
    st.caption(f"{len(to_plain(doc, opts).split())} words · {pages} page"
               + ("" if pages == 1 else "s"))

    downloads = st.columns(4)
    downloads[0].download_button(f"{style} PDF", pdf,
                                 file_name=f"resume_{style.lower()}.pdf",
                                 mime="application/pdf", use_container_width=True)
    downloads[1].download_button("LaTeX", to_latex(doc, opts),
                                 file_name="resume.tex", use_container_width=True)
    downloads[2].download_button("Plain text", to_plain(doc, opts),
                                 file_name="resume.txt", use_container_width=True,
                                 help="The ATS-safe version: no tables, no "
                                      "columns, one bullet per line.")
    downloads[3].download_button("JSON", doc.to_json(), file_name="resume.json",
                                 use_container_width=True,
                                 help="Everything, including evidence and "
                                      "which lines are switched off.")
    st.html(to_html(doc, opts))


# ----------------------------------------------------------------- editor --
def _origin_chip(origin: str) -> str:
    colour = {"yours": "green", "repo": "gray", "assumed": "orange"}.get(origin, "gray")
    return f":{colour}[{origin}]"


def _delta_cell(impact, stale: bool, enabled: bool) -> None:
    if impact is None:
        st.caption("—" if enabled else "off")
        return
    mark = "~" if stale else ""
    if impact.delta > 0:
        st.markdown(f"**:green[{mark}{impact.label}]**")
    elif impact.delta < 0:
        st.markdown(f"**:red[{mark}{impact.label}]**")
    else:
        st.caption(f"{mark}0.0")


def _suggestion(bullet) -> None:
    """Offer the best verified rewrite of this line, with its measured gain.

    The gain is not the model's opinion. Every rewrite was substituted into the
    document and re-scored, and any rewrite that introduced a number, a
    technology or an outcome its evidence does not support was rejected before
    it ever got scored.
    """
    store = state()
    variants = store.variants
    if variants is None:
        return
    best = variants.best(bullet.id)
    if best is None or best.gain <= 0:
        return
    with st.container(border=True):
        st.caption(f"Suggested rewrite · :green[{best.gain:+.1f}] measured")
        st.markdown(best.text)
        if st.button("Use this wording", key=f"use_{bullet.id}",
                     use_container_width=False):
            bullet.text = best.text
            st.session_state[f"tx_{bullet.id}"] = best.text
            st.rerun(scope="fragment")


def _bullet_rows(bullets, att, stale: bool) -> None:
    for bullet in bullets:
        impact = att.by_id(bullet.id) if att else None
        toggle, body, score = st.columns([0.5, 9, 1.5])
        with toggle:
            bullet.enabled = st.checkbox(
                "on", value=bullet.enabled, key=f"on_{bullet.id}",
                label_visibility="collapsed")
        with body:
            bullet.text = st.text_input("bullet", bullet.text,
                                        key=f"tx_{bullet.id}",
                                        label_visibility="collapsed")
            caption = [_origin_chip(bullet.origin)]
            if bullet.source_repo:
                caption.append(f"`{bullet.source_repo}`")
            if not bullet.verified:
                caption.append("⚠ needs measuring")
            if bullet.evidence:
                caption.append(f"evidence: “{bullet.evidence[:70]}”")
            st.caption(" · ".join(caption))
        with score:
            _delta_cell(impact, stale, bullet.enabled)
        if bullet.enabled:
            _suggestion(bullet)


def suggest_rewrites(config: dict) -> None:
    """Ask one worker per project for rephrasings, then measure them here.

    This is the only control on the page that spends tokens, which is why it is
    a button and not automatic: one structured call per project, so a
    four-project resume costs four requests against a measured budget of ~3,058
    tokens per call (`evals/budget.md`). The scoring and the verification that
    follow are pure Python.
    """
    store = state()
    try:
        with st.spinner("Rewriting each line, then measuring every rewrite…"):
            async def run():
                async with GroqClient(
                        Config.load(),
                        api_key=config.get("groq_key") or None) as client:
                    return await generate_variants(
                        store.doc, config["jd"], client=client,
                        required_skills=split(config.get("required", "")),
                        preferred_skills=split(config.get("preferred", "")))

            store.variants = asyncio.run(run())
    except Exception as exc:
        st.error(_diagnose(exc))
        return
    kept, rejected = store.variants.kept(), store.variants.rejected()
    st.success(f"{len(kept)} rewrite(s) kept, {len(rejected)} rejected for "
               "claiming something the evidence does not support.")


def editor(config: dict, stale: bool) -> None:
    store = state()
    doc: ResumeDoc = store.doc
    att = store.attribution
    new = candidates(doc)
    if new:
        st.caption(f"{len(new)} candidate line(s) from your repositories, "
                   "switched off. Turn one on and the score above moves "
                   "immediately.")

    for section in doc.sections:
        if not (section.entries or section.bullets):
            continue
        st.markdown(f"#### {section.title}")
        for entry in section.entries:
            label = " — ".join(p for p in (entry.title, entry.subtitle) if p)
            st.markdown(f"**{label}**  ·  {entry.dates}" if entry.dates
                        else f"**{label}**")
            _bullet_rows(entry.bullets, att, stale)
        _bullet_rows(section.bullets, att, stale)


# -------------------------------------------------------------- workspace --
@st.fragment
def workspace(config: dict) -> None:
    """Everything that reacts to a click, in one fragment.

    Streamlit reruns the whole script on any widget change, which meant a
    toggle redrew the sidebar, the uploader and the downloads before the score
    moved. Keeping the checkboxes, the score and the preview inside a single
    `st.fragment` scopes that rerun to this function, so a click costs one
    document embedding (~293 ms measured) and nothing else.
    """
    store = state()
    sync_from_widgets(store.doc)
    score, channels = live_score(config)
    stale = store.attr_signature != _signature(store.doc)

    left, right = st.columns([1, 1.45], gap="large")
    with left:
        score_panel(config, score, channels)
    with right:
        render_panel(config)

    st.divider()
    if stale and store.attribution is not None:
        bar = st.columns([3, 1])
        bar[0].caption("The per-line +/- figures below were measured against "
                       "an earlier version of this resume — shown with `~`. "
                       "The score above is always current.")
        if bar[1].button("Refresh +/-", use_container_width=True):
            with st.spinner("Re-measuring every line…"):
                refresh_deltas(config)
            st.rerun(scope="fragment")

    controls = st.columns([1.1, 3])
    if controls[0].button("Suggest better wording", use_container_width=True,
                          help="One model call per project. Every rewrite is "
                               "then verified against its evidence and scored "
                               "by real re-scoring."):
        suggest_rewrites(config)
    if store.variants is not None:
        rejected = store.variants.rejected()
        controls[1].caption(
            f"{len(store.variants.kept())} verified rewrite(s) available · "
            f"{len(rejected)} rejected for claiming something unsupported"
            if rejected else
            f"{len(store.variants.kept())} verified rewrite(s) available")

    editor(config, stale)


# ------------------------------------------------------------------- main --
def main() -> None:
    st.title("Resume Builder")
    st.caption("Starts from the resume you already have, proposes lines drawn "
               "from your repositories, and shows exactly what each one is "
               "worth. Never claims anything your evidence does not support.")

    config = sidebar()
    if st.sidebar.button("Build", type="primary", use_container_width=True):
        build(config)
        if state().doc is not None:
            recompute(config)

    store = state()
    if store.doc is None:
        st.info("Upload your resume in the sidebar and press **Build**.")
        st.markdown(
            "- **Upload** a PDF and it is parsed into its real structure — "
            "tables, dated entries, `Tech:` lines — not flattened to bullets.\n"
            "- **Add from GitHub** reads your public repositories, summarises "
            "each one, and discards any claim its evidence does not support.\n"
            "- Candidate lines arrive **switched off**. Your resume is the "
            "baseline and is never rewritten.\n"
            "- Tick or untick any line and the score moves immediately.")
        return

    workspace(config)

    with st.expander("How the number is produced"):
        st.markdown(
            "Three channels, fused by weighted sum with theoretical min-max "
            "normalisation — **not** Reciprocal Rank Fusion, which Bruch et al. "
            "(arXiv 2210.11934, TOIS) beat with a weighted sum on all five "
            "datasets they tested.\n\n"
            "**Every number here is measured, never estimated.** The score "
            "recomputes in full on each click (~293 ms: one document "
            "embedding, with the job description's cached). The per-line +/- "
            "is exact leave-one-out — the score with that line removed — which "
            "costs ~4.8 s for a fifteen-line resume, so it refreshes on "
            "request rather than on every click.\n\n"
            "An approximation would have been 4,253x faster, but measured "
            "against ground truth it got the *sign* wrong on 6 lines in 20 — "
            "saying a line helped when it hurt — so it is not used.\n\n"
            "The per-line figures do **not** sum to the total. The score is "
            "non-linear: BM25 saturates and cosine normalises by length, so "
            "contributions are not additive. Each delta answers only *“what "
            "happens if I remove this line”* — which is exactly what the "
            "checkbox does.\n\n"
            "The advice panel deliberately carries **no weights**. The seven "
            "weighted bars it replaces summed to 100 by construction; nobody "
            "measured that a missing phone number is worth six points.")
        st.code("\n".join(f"{name:<9} weight {weight}"
                          for name, weight in CHANNEL_WEIGHTS.items()),
                language="text")


if __name__ == "__main__":
    main()
