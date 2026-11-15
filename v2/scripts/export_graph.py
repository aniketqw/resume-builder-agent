#!/usr/bin/env python3
"""Export the real LangGraph pipeline as a diagram.

Generated directly from the compiled graph (`build_graph(Runner(...))`), not
hand-drawn -- a node added or renamed in `graph.py` shows up here on the next
run instead of silently going stale in a picture nobody updates.

    .venv/bin/python scripts/export_graph.py

Writes `docs/v2-graph.mmd` (Mermaid source, embedded directly in
`ANALYSIS.md` -- GitHub renders a ```mermaid fence natively, so no image file
or external rendering service is required to view it). Also tries to render
`docs/v2-graph.png` via the same `draw_mermaid_png()` LangGraph exposes, which
calls out to the mermaid.ink API; if that network call fails (it did once in
this environment -- a DNS/reachability issue, not a code bug) the PNG is
skipped and only the `.mmd` source is written, since the Mermaid source alone
is already enough to view the diagram on GitHub.
"""
from __future__ import annotations

import pathlib

from rba.config import Config
from rba.graph import Runner, build_graph

DOCS = pathlib.Path(__file__).resolve().parents[2] / "docs"


def main() -> None:
    runner = Runner(Config.load())
    compiled = build_graph(runner)
    mermaid = compiled.get_graph().draw_mermaid()

    mmd_path = DOCS / "v2-graph.mmd"
    mmd_path.write_text(mermaid)
    print(f"wrote {mmd_path}")

    try:
        png = compiled.get_graph().draw_mermaid_png()
        png_path = DOCS / "v2-graph.png"
        png_path.write_bytes(png)
        print(f"wrote {png_path}")
    except Exception as exc:
        print(f"PNG render skipped ({type(exc).__name__}: {exc}); "
              "the .mmd source above is enough to view the diagram on GitHub.")


if __name__ == "__main__":
    main()
