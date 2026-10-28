"""Streamlit entry point. Must run from the repository root.

Scope is deliberately the **scorer**, not the full GitHub agent. A full agent
run takes 8-25 minutes against Groq's 8,000-tokens-per-minute free tier, and
Streamlit Community Cloud drops the websocket after roughly 10-15 minutes of
quiet. Scoring is instant, needs no API key, and cannot be rate limited.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from rba.ats import score_resume_against_jd  # noqa: E402
from rba.ats.fusion import CHANNEL_WEIGHTS  # noqa: E402
from rba.ats.normalise import ALIASES  # noqa: E402

st.set_page_config(page_title="Resume Scorer — evidence-grounded",
                   page_icon="📄", layout="wide")

SAMPLE_JD = """Backend Engineer

Required: Python, Docker, Kubernetes, REST APIs, SQL
Preferred: AWS, Kafka, Terraform

You will design and operate microservices in production, own services
end to end, and take part in an on-call rotation."""

SAMPLE_RESUME = """Software Engineer

Built Python services with FastAPI, containerised with Docker and deployed on
Kubernetes. Designed REST APIs consumed by three internal teams. Wrote the
PostgreSQL schema and the migration tooling. Set up CI with GitHub Actions and
added the first integration test suite."""


@st.cache_resource(show_spinner=False)
def _warm_dense() -> bool:
    """Load the embedding model once per container."""
    from rba.ats.dense import available
    return available()


def main() -> None:
    st.title("Resume Scorer")
    st.caption("Three independent channels, fused. The formula is published "
               "below — no hidden rubric.")

    with st.expander("What this number is, and what it is not", expanded=False):
        st.markdown(
            "**No applicant tracking system shows a candidate a 0-100 score.** "
            "Greenhouse reports Strong / Good / Partial bands, Workday reports "
            "A-D letter grades, and both are driven by criteria the *employer* "
            "configures per job. So this is our number, not an ATS's, and the "
            "band beside it exists to stay recognisable.\n\n"
            "The design follows the second-placed system at **TalentCLEF 2026** "
            "(a CLEF shared task whose Task A is exactly this problem): parse "
            "documents into structured fields, then fuse several retrieval "
            "views. That approach scored MAP 0.6632 against the winner's "
            "0.7140, with no training.\n\n"
            "**There is no ground truth for an absolute resume score.** The "
            "channel weights below are a judgement call. What we can honestly "
            "offer is that they are *visible*, and that each channel is shown "
            "separately so you can disagree with the mix."
        )

    left, right = st.columns(2)
    with left:
        jd_text = st.text_area("Job description", SAMPLE_JD, height=260)
        required = st.text_input("Required skills (comma separated)",
                                 "Python, Docker, Kubernetes, REST API, SQL")
    with right:
        resume_text = st.text_area("Resume", SAMPLE_RESUME, height=260)
        preferred = st.text_input("Preferred skills (comma separated)",
                                  "AWS, Kafka, Terraform")

    resume_skills = st.text_input(
        "Skills evidenced in the resume (comma separated)",
        "Python, Docker, Kubernetes, REST API, PostgreSQL, CI/CD",
        help="Normally produced by the extraction step. Typed here so the "
             "scoring stays inspectable.")

    use_dense = st.toggle("Use the semantic channel (downloads a 130MB model "
                          "on first run)", value=True)

    if not st.button("Score", type="primary"):
        st.stop()

    def split(value: str) -> list[str]:
        return [part.strip() for part in value.split(",") if part.strip()]

    if use_dense:
        with st.spinner("Loading the embedding model…"):
            if not _warm_dense():
                st.warning("The embedding model could not be loaded, so the "
                           "semantic channel was skipped and its weight "
                           "redistributed.")

    report = score_resume_against_jd(
        resume_text, jd_text,
        resume_skills=split(resume_skills),
        required_skills=split(required),
        preferred_skills=split(preferred),
        use_dense=use_dense,
    )

    st.divider()
    headline, band = st.columns([1, 2])
    headline.metric("Score", f"{report.score_100:.0f}/100")
    band.metric("Band", report.band,
                help="Mirrors Greenhouse's Strong/Good/Partial/Limited.")

    st.subheader("How that number was reached")
    for line in report.explain():
        st.markdown(f"- {line}")

    cols = st.columns(3)
    labels = {"dense": "Semantic", "lexical": "Keyword (BM25)", "skills": "Skills"}
    for column, name in zip(cols, ("dense", "lexical", "skills"), strict=False):
        value = report.fused.channels.get(name)
        column.metric(labels[name],
                      "—" if value is None else f"{value * 100:.0f}/100",
                      help="Dropped; weight redistributed." if value is None else None)

    if report.skills.missing_required or report.skills.matched_required:
        st.subheader("Skill coverage")
        st.write(report.skills.summary)
        if report.skills.missing_required:
            st.error("Required, not evidenced: "
                     + ", ".join(report.skills.missing_required))
        if report.skills.matched_required:
            st.success("Required, evidenced: "
                       + ", ".join(report.skills.matched_required))

    if report.missing_terms:
        st.subheader("Job-description terms absent from the resume")
        st.caption("Lexical only. Some of these are filler; BM25 already "
                   "discounts common words by how often they appear across a "
                   "background corpus of 22 job descriptions.")
        st.write(", ".join(report.missing_terms[:40]))

    with st.expander("The formula"):
        st.markdown(
            "Convex combination with theoretical min-max normalisation — "
            "**not** Reciprocal Rank Fusion. Bruch et al. (arXiv 2210.11934, "
            "TOIS) found a weighted sum beat RRF on all five datasets they "
            "tested, and that RRF's tuned parameters do not transfer between "
            "domains.")
        st.code("\n".join(f"{name:<9} weight {weight}"
                          for name, weight in CHANNEL_WEIGHTS.items())
                + "\n\nscore = sum(channel * weight) / sum(weights of available channels)",
                language="text")
        st.caption(f"Skill aliases recognised: {len(ALIASES)} canonical names. "
                   "An unknown skill passes through unchanged rather than "
                   "being forced onto a nearest match.")


if __name__ == "__main__":
    main()
