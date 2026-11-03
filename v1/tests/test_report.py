from rba.report import render_latex, render_markdown, tex_escape, write_reports
from rba.state import (
    Claim,
    ClaimStatus,
    Evidence,
    RepoSummary,
    Resume,
    RoleScore,
    SafetyFinding,
)


def sample_state():
    return {
        "username": "aniketqw",
        "claims": [
            Claim(id="a", repo="demo", text="Ships 34 tests",
                  evidence=[Evidence(path="README.md", quote="34 tests, about 2 s")]),
            Claim(id="b", repo="demo", text="Reduced latency",
                  status=ClaimStatus.NEEDS_MEASURE, reason="no quantity",
                  measure_hint="time it before and after"),
            Claim(id="c", repo="demo", text="Losing 745 of 1484",
                  status=ClaimStatus.REJECTED,
                  reason="contains number(s) absent from the evidence: 745"),
        ],
        "summaries": {"demo": RepoSummary(
            repo="demo", one_line="A demo project.",
            claims=[Claim(id="a", repo="demo", text="Ships 34 tests",
                          evidence=[Evidence(path="README.md", quote="34 tests")])])},
        "resumes": {"balanced": Resume(
            role="balanced", headline="Software Engineer", summary="Builds things.",
            sections={"summary": ["Builds things."], "projects": ["Shipped 34 tests"]})},
        "scores": {"balanced": RoleScore(role="balanced", total=61.5,
                                         components={"keywords": 20.0})},
        "comparison": {"rows": [{"role": "balanced", "title": "Software Engineer",
                                 "total": 61.5, "components": {"keywords": 20.0},
                                 "top_gaps": ["docker"], "potential_gain": 4.4}],
                       "best_role": "balanced", "spread": 0.0},
        "safety": [SafetyFinding(repo="leaky", path=".env", kind="env-file",
                                 detail="An .env file is committed.", severity="high")],
        "contact": "a@b.com | github.com/x",
        "usage": {"groq": {"total": 2074, "calls": 1}},
    }


def test_markdown_reports_all_three_claim_states():
    md = render_markdown(sample_state())
    assert "1 claims verified" in md or "**1 claims verified**" in md
    assert "Measure these" in md
    assert "time it before and after" in md
    assert "Refused claims" in md
    assert "745" in md, "a refused claim must stay visible, not be hidden"


def test_security_findings_come_first():
    md = render_markdown(sample_state())
    assert md.index("Security findings") < md.index("Role comparison")
    assert ".env file is committed" in md


def test_evidence_index_cites_sources():
    md = render_markdown(sample_state())
    assert "Evidence index" in md
    assert "`README.md`" in md


def test_markdown_survives_an_empty_state():
    md = render_markdown({})
    assert "Resume report" in md


def test_latex_escapes_special_characters():
    assert tex_escape("100% C&C_test #1") == r"100\% C\&C\_test \#1"


def test_latex_renders_a_compilable_document():
    tex = render_latex(sample_state(), "balanced")
    assert tex.startswith(r"\documentclass")
    assert tex.rstrip().endswith(r"\end{document}")
    assert r"\item Shipped 34 tests" in tex


def test_write_reports_emits_markdown_and_latex(tmp_path):
    written = write_reports(sample_state(), tmp_path)
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "resume_balanced.tex").exists()
    assert "markdown" in written and "latex_balanced" in written
