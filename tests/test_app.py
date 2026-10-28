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


def built_app() -> AppTest:
    app = AppTest.from_file(APP, default_timeout=240)
    app.run()
    app.sidebar.radio[0].set_value("Paste or upload").run()
    app.sidebar.text_area[0].set_value(RESUME).run()
    app.sidebar.button[0].click().run()
    return app


def test_app_renders_without_exception():
    app = AppTest.from_file(APP, default_timeout=120)
    app.run()
    assert not app.exception, app.exception
    assert app.title[0].value == "Resume Builder"


def test_pasted_resume_becomes_an_editable_scored_document():
    app = built_app()
    assert not app.exception, app.exception
    labels = {m.label: m.value for m in app.metric}
    assert "Score" in labels and "Band" in labels
    # One checkbox per parsed bullet, so each line can be toggled.
    assert len(app.checkbox) == 3


def test_every_bullet_shows_a_signed_impact():
    app = built_app()
    deltas = [m.value for m in app.markdown if m.value.startswith("**:")]
    assert len(deltas) == 3
    assert all(("+" in d) or ("-" in d) for d in deltas)


def test_an_irrelevant_bullet_is_flagged_as_harmful():
    """Against a backend job description, an office-social-committee line
    lowers the score. A grounded scorer can say so; a guessed one cannot."""
    app = built_app()
    red = [m.value for m in app.markdown if m.value.startswith("**:red[")]
    assert red, "expected at least one negative-impact bullet"
    assert app.warning, "harmful bullets should be surfaced, not buried"
    assert "lowering" in app.warning[0].value


def test_role_archetype_mode_works_without_a_job_description():
    app = AppTest.from_file(APP, default_timeout=240)
    app.run()
    app.sidebar.radio[0].set_value("Paste or upload").run()
    app.sidebar.radio[1].set_value("Pick a role archetype").run()
    app.sidebar.text_area[0].set_value(RESUME).run()
    app.sidebar.button[0].click().run()
    assert not app.exception, app.exception
    assert {m.label for m in app.metric} >= {"Score", "Band"}


def test_building_with_no_input_reports_an_error_rather_than_crashing():
    app = AppTest.from_file(APP, default_timeout=120)
    app.run()
    app.sidebar.radio[0].set_value("Paste or upload").run()
    app.sidebar.button[0].click().run()
    assert not app.exception
    assert app.error, "an empty build should explain itself"


def test_the_formula_is_published_in_the_ui():
    app = built_app()
    rendered = " ".join(block.value for block in app.code)
    from rba.ats.fusion import CHANNEL_WEIGHTS
    for name in CHANNEL_WEIGHTS:
        assert name in rendered
