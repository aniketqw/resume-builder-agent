"""Turning a fused similarity into a number that means something.

A raw fused score of 0.63 is not interpretable. Neither is "63/100" — it
*looks* interpretable, which is worse, because it implies a calibration nobody
has done. Proper calibration would mean "candidates scoring 80 reach the next
stage about 80% of the time", measured against interview outcomes. That needs
labelled hiring data, which does not exist publicly in English; the one public
person-job dataset with real labels is Chinese.

What *is* defensible without labels is a percentile against a stated reference
set: **"this pair scores higher than 63% of the reference pairs"**. That claim
is true by construction, needs no outcome labels, and makes the reference set
the explicit assumption — which is the honest place to put the uncertainty.

The empirical CDF is stored as a short list of quantile breakpoints, so
scoring at runtime is a binary search over a hundred floats.
"""
from __future__ import annotations

import bisect
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).parent / "data" / "calibration.json"


@dataclass(frozen=True)
class Calibration:
    """Quantile breakpoints of raw fused scores over a reference set."""

    breakpoints: tuple[float, ...]
    reference_size: int
    description: str = ""
    # Which channels were active when the reference scores were computed.
    # A percentile is only meaningful against a like-for-like distribution:
    # scoring with the semantic channel off produces systematically lower raw
    # values, and comparing those to a reference built with it on floored
    # every result at the 0th percentile.
    channels: tuple[str, ...] = ()

    def percentile(self, raw: float) -> float:
        """Percentage of the reference set this score exceeds."""
        if not self.breakpoints:
            return raw * 100.0
        index = bisect.bisect_right(self.breakpoints, raw)
        return round(100.0 * index / len(self.breakpoints), 1)

    def matches(self, channels) -> bool:
        """Whether a scoring run used the same channels as this reference."""
        if not self.channels:
            return True          # legacy reference set; assume compatible
        return tuple(sorted(channels)) == tuple(sorted(self.channels))

    @property
    def usable(self) -> bool:
        # Below roughly this many pairs the quantiles are noise and a
        # percentile would imply more precision than exists.
        return self.reference_size >= 30 and len(self.breakpoints) >= 10


@lru_cache(maxsize=1)
def load_calibration() -> Calibration:
    if not DATA.is_file():
        return Calibration(breakpoints=(), reference_size=0,
                           description="no reference set built")
    payload = json.loads(DATA.read_text())
    return Calibration(
        breakpoints=tuple(payload.get("breakpoints", [])),
        reference_size=int(payload.get("reference_size", 0)),
        description=payload.get("description", ""),
        channels=tuple(payload.get("channels", [])),
    )


def build_calibration(raw_scores: list[float], quantiles: int = 100,
                      description: str = "",
                      channels: tuple[str, ...] = ()) -> Calibration:
    """Reduce observed raw scores to evenly spaced quantile breakpoints."""
    ordered = sorted(float(s) for s in raw_scores)
    if not ordered:
        return Calibration(breakpoints=(), reference_size=0,
                           description=description, channels=channels)
    steps = min(quantiles, len(ordered))
    breakpoints = []
    for i in range(1, steps + 1):
        position = round(i * len(ordered) / steps) - 1
        breakpoints.append(ordered[max(0, min(position, len(ordered) - 1))])
    return Calibration(breakpoints=tuple(breakpoints),
                       reference_size=len(ordered), description=description,
                       channels=channels)


def save_calibration(calibration: Calibration, path: Path | None = None) -> Path:
    target = path or DATA
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({
        "description": calibration.description,
        "reference_size": calibration.reference_size,
        "channels": list(calibration.channels),
        "breakpoints": [round(b, 6) for b in calibration.breakpoints],
    }, indent=1) + "\n")
    return target
