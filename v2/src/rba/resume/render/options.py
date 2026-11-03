"""What every renderer is allowed to vary."""
from __future__ import annotations

from dataclasses import dataclass

# How a bullet's provenance is coloured, matching the legend in the UI.
ORIGIN_COLOURS = {
    "yours": "#1a7f37",     # green  -- came from the resume you supplied
    "repo": "#57606a",      # grey   -- generated from repository evidence
    "assumed": "#9a6700",   # amber  -- contains something unproven
}
ORIGIN_LABELS = {
    "yours": "from your resume",
    "repo": "from repository evidence",
    "assumed": "assumed — needs measuring",
}


@dataclass(frozen=True)
class RenderOptions:
    """Switches shared by every output format.

    `style` picks the layout: "bits" reproduces the one-page academic format
    with its tables, "clean" is a plainer single-column CV.
    """

    style: str = "bits"
    show_numbers: bool = True
    colour_provenance: bool = False
    include_disabled: bool = False
    # Resumes for early-career roles are expected on one page. 0 disables the
    # shrink-to-fit pass entirely.
    max_pages: int = 1

    @property
    def is_bits(self) -> bool:
        return self.style == "bits"
