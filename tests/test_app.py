"""The Streamlit app must render, build and score without raising.

A broken `app.py` fails only at runtime on Community Cloud, where
`showErrorDetails = "none"` hides the traceback and viewers get a blank page.
"""
import pytest

pytest.importorskip("streamlit")
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")

RESUME = ("Jane Doe\nBackend Engineer\njane@example.com\n\nExperience:\n"
          "- Built Python FastAPI services with Docker\n"
          "- Deployed on Kubernetes and wrote REST APIs\n"
          "- Organised the office social committee\n")

BACKEND_JD = ("Backend Engineer\n\nRequired: Python, Docker, Kubernetes, "
              "REST APIs, SQL\n\nDesign and operate microservices in "
              "production, own services end to end.")


def built_app(role_mode: bool = False, jd: str = "") -> AppTest:
    app = AppTest.from_file(APP, default_timeout=240)
    app.run()
    if role_mode:
        app.sidebar.radio[0].set_value("Pick a role").run()
    elif jd:
        # The job-description box is the second text area; the first is the
        # resume. Setting it explicitly keeps these tests independent of
        # whatever sample JD the app happens to ship with.
        app.sidebar.text_area[1].set_value(jd).run()
    app.sidebar.text_area[0].set_value(RESUME).run()
    app.sidebar.button[0].click().run()
    return app


def bullet_toggles(app) -> list:
    """Only the per-bullet checkboxes, not the sidebar's GitHub switch."""
    return [c for c in app.checkbox if (c.key or "").startswith("on_")]


def test_app_renders_without_exception():
    app = AppTest.from_file(APP, default_timeout=120)
    app.run()
    assert not app.exception, app.exception
    assert app.title[0].value == "Resume Builder"


def test_pasted_resume_becomes_an_editable_scored_document():
    app = built_app()
    assert not app.exception, app.exception
    assert "Match score" in {m.label for m in app.metric}
    # One checkbox per parsed bullet, so each line can be toggled.
    assert len(bullet_toggles(app)) == 3


def test_every_bullet_shows_a_signed_impact():
    app = built_app()
    deltas = [m.value for m in app.markdown if m.value.startswith("**:")]
    assert len(deltas) >= 3
    assert all(("+" in d) or ("-" in d) for d in deltas)


def test_an_irrelevant_bullet_is_flagged_as_harmful():
    """Against a backend job description, an office-social-committee line
    lowers the score. A grounded scorer can say so; a guessed one cannot."""
    app = built_app(jd=BACKEND_JD)
    red = [m.value for m in app.markdown if ":red[" in m.value]
    assert red, "expected at least one negative-impact bullet"
    headings = [m.value for m in app.markdown]
    assert any("costing you" in h for h in headings), \
        "harmful bullets should be surfaced, not buried"


def test_role_archetype_mode_works_without_a_job_description():
    app = built_app(role_mode=True)
    assert not app.exception, app.exception
    assert "Match score" in {m.label for m in app.metric}


def test_building_with_no_input_reports_an_error_rather_than_crashing():
    app = AppTest.from_file(APP, default_timeout=120)
    app.run()
    app.sidebar.button[0].click().run()
    assert not app.exception
    assert app.error, "an empty build should explain itself"


def test_the_formula_is_published_in_the_ui():
    app = built_app()
    rendered = " ".join(block.value for block in app.code)
    from rba.ats.fusion import CHANNEL_WEIGHTS
    for name in CHANNEL_WEIGHTS:
        assert name in rendered


def test_downloads_offer_every_format():
    app = built_app()
    labels = {b.label for b in app.get("download_button")}
    assert {"LaTeX", "Plain text", "JSON"} <= labels
    assert any("PDF" in label for label in labels)


def test_the_advice_panel_carries_no_invented_weights():
    """The seven-bar rubric is gone; nothing may reintroduce its numbers."""
    app = built_app()
    text = " ".join(m.value for m in app.markdown) + \
        " ".join(c.value for c in app.caption)
    assert "out of 100" not in text
    assert "Not score components" in text


def test_clicking_a_point_moves_the_score_without_a_rescore_button():
    """The whole point of the rewrite: a toggle is a control, not a form.

    Untick a line and the score must change on that interaction alone -- no
    'Rescore' press in between.
    """
    app = built_app(jd=BACKEND_JD)
    before = next(m.value for m in app.metric if m.label == "Match score")
    toggles = bullet_toggles(app)
    assert toggles, "expected per-line checkboxes"
    toggles[0].uncheck().run()
    after = next(m.value for m in app.metric if m.label == "Match score")
    assert before != after, f"score did not move on toggle ({before} -> {after})"


def test_stale_deltas_are_marked_rather_than_silently_wrong():
    """Per-line figures describe the document they were measured against.

    After an edit they are shown with a '~' and a refresh offered, instead of
    quietly presenting numbers for a resume the user has moved away from.
    """
    app = built_app(jd=BACKEND_JD)
    bullet_toggles(app)[0].uncheck().run()
    marked = [m.value for m in app.markdown if "~" in m.value and ":" in m.value]
    labels = [b.label for b in app.button]
    assert marked or "Refresh +/-" in labels, \
        "edited document should mark its deltas stale"
