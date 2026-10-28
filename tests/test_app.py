"""The Streamlit app must render and score without raising.

A broken `app.py` fails only at runtime on Community Cloud, where the traceback
is hidden from viewers by `showErrorDetails = "none"`. Catching it here is the
difference between a failing test and a blank public page.
"""
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from rba.ats.fusion import CHANNEL_WEIGHTS


def _app() -> AppTest:
    app = AppTest.from_file(str(__import__("pathlib").Path(__file__).parents[1]
                                / "app.py"), default_timeout=180)
    app.run()
    return app


def test_app_renders_without_exception():
    app = _app()
    assert not app.exception, app.exception
    assert app.title[0].value == "Resume Scorer"


def test_app_scores_without_exception():
    app = _app()
    # Semantic channel off: keeps the test offline and fast.
    app.toggle[0].set_value(False)
    app.button[0].click().run()
    assert not app.exception, app.exception
    labels = {m.label: m.value for m in app.metric}
    # The headline is a percentile when the reference set matches the channels
    # used, and a raw score otherwise. Both are valid; neither is "/100 ATS".
    assert "Percentile" in labels or "Score (raw)" in labels
    assert labels["Band"] in {"Strong match", "Good match",
                              "Partial match", "Limited match"}


def test_a_disabled_channel_is_shown_as_dropped_not_zero():
    app = _app()
    app.toggle[0].set_value(False)
    app.button[0].click().run()
    labels = {m.label: m.value for m in app.metric}
    assert labels["Semantic"] == "—", "a dropped channel must not read as 0"


def test_app_publishes_the_formula():
    """The differentiator is that the weights are visible, so they must be."""
    app = _app()
    app.toggle[0].set_value(False)
    app.button[0].click().run()
    rendered = " ".join(block.value for block in app.code)
    for name in CHANNEL_WEIGHTS:
        assert name in rendered
