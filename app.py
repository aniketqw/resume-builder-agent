"""Streamlit entry point. Run from the repository root.

Three steps: get a resume (from GitHub or an upload), say what job it is for,
then edit it while a grounded score moves.

The per-bullet +/- is the point of the thing. It is exact leave-one-out: each
number is what the score really loses if you remove that line, verified in
tests against an independent rescore. That is affordable only because scoring
makes no LLM call -- one ONNX embedding plus arithmetic, ~150 ms per toggle.
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
from rba.resume import (  # noqa: E402
    ResumeDoc,
    attribute,
    build_from_github,
    parse_resume_text,
)
from rba.scoring import load_roles  # noqa: E402

st.set_page_config(page_title="Resume Builder — evidence-grounded",
                   page_icon="📄", layout="wide")

SAMPLE_JD = """Backend Engineer

Required: Python, Docker, Kubernetes, REST APIs, SQL
Preferred: AWS, Kafka, Terraform

Design and operate microservices in production, own services end to end."""


def state():
    st.session_state.setdefault("doc", None)
    st.session_state.setdefault("stats", {})
    st.session_state.setdefault("attribution", None)
    return st.session_state


def _roles() -> dict:
    return load_roles()


# ----------------------------------------------------------------- inputs --
def sidebar() -> dict:
    st.sidebar.header("1 · Where does the resume come from?")
    source = st.sidebar.radio("Source", ["GitHub", "Paste or upload"],
                              label_visibility="collapsed")

    config: dict = {"source": source}
    if source == "GitHub":
        config["username"] = st.sidebar.text_input("GitHub username", "aniketqw")
        config["github_token"] = st.sidebar.text_input(
            "GitHub token", type="password",
            help="Unauthenticated GitHub allows only 60 requests/hour, shared "
                 "across everyone using this app. A token gives you 5,000 and "
                 "spends your quota, not ours. Never stored.")
        config["groq_key"] = st.sidebar.text_input(
            "Groq API key", type="password",
            help="Used to summarise your projects. Never stored.")
        config["max_repos"] = st.sidebar.slider(
            "Repositories to describe", 2, 12, 5,
            help="Groq's free tier allows 8,000 tokens a minute, so each repo "
                 "costs real time. Five finishes in a couple of minutes.")
    else:
        uploaded = st.sidebar.file_uploader("Upload a resume (.txt or .md)",
                                            type=["txt", "md"])
        config["uploaded"] = uploaded
        config["pasted"] = st.sidebar.text_area("…or paste it", height=200)

    st.sidebar.header("2 · What job is it for?")
    mode = st.sidebar.radio("Target", ["Paste a job description",
                                       "Pick a role archetype"],
                            label_visibility="collapsed")
    if mode == "Paste a job description":
        config["jd"] = st.sidebar.text_area("Job description", SAMPLE_JD,
                                            height=180)
        config["required"] = st.sidebar.text_input(
            "Required skills", "Python, Docker, Kubernetes, REST API, SQL")
        config["preferred"] = st.sidebar.text_input(
            "Preferred skills", "AWS, Kafka, Terraform")
    else:
        roles = _roles()
        key = st.sidebar.selectbox("Role", list(roles),
                                   format_func=lambda k: roles[k].title)
        role = roles[key]
        top = sorted(role.keywords, key=lambda k: -role.keywords[k])
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
    if config["source"] == "GitHub":
        if not config.get("username"):
            st.error("Enter a GitHub username.")
            return
        bar = st.progress(0.0, text="Starting…")

        def report(message: str, fraction: float) -> None:
            bar.progress(min(max(fraction, 0.0), 1.0), text=message)

        try:
            doc, stats = asyncio.run(build_from_github(
                config["username"], github_token=config.get("github_token", ""),
                groq_key=config.get("groq_key", ""),
                max_repos=config.get("max_repos", 5), progress=report))
        except Exception as exc:
            bar.empty()
            st.error(_diagnose(exc))
            with st.expander("Technical detail"):
                st.code(f"{type(exc).__name__}: {exc}"[:800])
            return
        bar.empty()
        store.doc, store.stats = doc, stats
    else:
        raw = ""
        if config.get("uploaded") is not None:
            raw = config["uploaded"].read().decode("utf-8", "replace")
        raw = raw or config.get("pasted", "")
        if not raw.strip():
            st.error("Upload or paste a resume first.")
            return
        store.doc, store.stats = parse_resume_text(raw), {}
    store.attribution = None


def recompute(config: dict) -> None:
    store = state()
    if store.doc is None:
        return
    with st.spinner("Scoring every bullet…"):
        store.attribution = attribute(
            store.doc, config["jd"],
            required_skills=split(config.get("required", "")),
            preferred_skills=split(config.get("preferred", "")))


# ------------------------------------------------------------------- view --
def show_resume(config: dict) -> None:
    store = state()
    doc: ResumeDoc = store.doc
    att = store.attribution

    head, meta = st.columns([3, 2])
    with head:
        doc.name = st.text_input("Name", doc.name)
        doc.headline = st.text_input("Headline", doc.headline)
        doc.contact = st.text_input("Contact", doc.contact)
    with meta:
        if att:
            st.metric("Score", f"{att.total:.0f}", help="Fused across three "
                                                        "channels; not an ATS number.")
            st.metric("Band", att.band)
        if store.stats:
            s = store.stats
            st.caption(
                f"{s.get('claims_verified', 0)}/{s.get('claims_total', 0)} claims "
                f"verified · {s.get('claims_rejected', 0)} refused · "
                f"{s.get('groq_tokens', 0):,} Groq tokens")

    if att and att.harmful():
        st.warning("These lines are **lowering** your score against this job — "
                   "consider removing them: "
                   + "; ".join(f"“{i.text[:60]}” ({i.label})"
                               for i in att.harmful()[:3]))

    st.divider()
    for section in doc.sections:
        st.subheader(section.title)
        for bullet in section.bullets:
            impact = att.by_id(bullet.id) if att else None
            toggle, body, score = st.columns([0.6, 9, 1.6])
            with toggle:
                bullet.enabled = st.checkbox(
                    "on", value=bullet.enabled, key=f"on_{bullet.id}",
                    label_visibility="collapsed")
            with body:
                bullet.text = st.text_input(
                    "bullet", bullet.text, key=f"tx_{bullet.id}",
                    label_visibility="collapsed")
                caption = []
                if bullet.source_repo:
                    caption.append(f"`{bullet.source_repo}`")
                if not bullet.verified:
                    caption.append("⚠ unverified — needs measuring")
                if bullet.evidence:
                    caption.append(f"evidence: “{bullet.evidence[:70]}”")
                if caption:
                    st.caption(" · ".join(caption))
            with score:
                if impact is None:
                    st.caption("—")
                elif impact.delta > 0:
                    st.markdown(f"**:green[{impact.label}]**")
                elif impact.delta < 0:
                    st.markdown(f"**:red[{impact.label}]**")
                else:
                    st.caption("0.0")

    st.divider()
    left, right = st.columns(2)
    if left.button("Rescore after my edits", type="primary",
                   use_container_width=True):
        recompute(config)
        st.rerun()
    right.download_button("Download resume (Markdown)", doc.as_markdown(),
                          file_name="resume.md", use_container_width=True)

    with st.expander("How the number is produced"):
        st.markdown(
            "Three channels, fused by weighted sum with theoretical min-max "
            "normalisation — **not** Reciprocal Rank Fusion, which Bruch et al. "
            "(arXiv 2210.11934, TOIS) beat with a weighted sum on all five "
            "datasets they tested.\n\n"
            "The **+/- beside each bullet is exact**: it is the score with that "
            "line removed, recomputed in full. Scoring makes no LLM call, so "
            "this costs about 150 ms. An approximation would have been 4,253x "
            "faster, but measured against ground truth it got the *sign* wrong "
            "on 6 bullets in 20 — saying a line helped when it hurt — so it is "
            "not used.\n\n"
            "No applicant tracking system shows a candidate a 0-100 score. "
            "Greenhouse reports Strong/Good/Partial, Workday reports A-D. This "
            "is our number, which is why the formula is printed here.")
        st.code("\n".join(f"{name:<9} weight {weight}"
                          for name, weight in CHANNEL_WEIGHTS.items()),
                language="text")


# ------------------------------------------------------------------- main --
def main() -> None:
    st.title("Resume Builder")
    st.caption("Builds a resume from your GitHub projects, then shows what "
               "every line is actually worth. Never claims anything your "
               "repositories do not support.")

    config = sidebar()
    if st.sidebar.button("Build resume", type="primary", use_container_width=True):
        build(config)
        if state().doc is not None:
            recompute(config)

    store = state()
    if store.doc is None:
        st.info("Choose a source in the sidebar and press **Build resume**.")
        st.markdown(
            "- **GitHub** reads your public repositories, summarises each one, "
            "and drops any claim its evidence does not support.\n"
            "- **Paste or upload** starts from a resume you already have.\n\n"
            "Either way you get a per-line score showing exactly what each "
            "bullet contributes."
        )
        return
    show_resume(config)


if __name__ == "__main__":
    main()
