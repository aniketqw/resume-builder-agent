"""HTTP layer for the resume builder.

v1 had none: its Streamlit page called the engine in-process and nothing else
could drive it. This package puts one documented surface in front of the same
functions so the browser, curl and the tests all go through the same door.

Nothing in here computes a score. Every number it returns comes from
`rba.ats` / `rba.resume`, unchanged.
"""
__all__ = ["create_app"]


def create_app(*args, **kwargs):
    from .app import create_app as _create_app
    return _create_app(*args, **kwargs)
