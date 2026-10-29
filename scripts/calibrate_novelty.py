"""Measure whether duplicate detection actually separates duplicates.

The thresholds in `rba.nodes.additions` come from this, not the other way
round. Each pair below is a real resume bullet and a hand-written rewording of
that same claim -- what the detector must catch -- scored against every pair of
genuinely different real bullets, which it must not.

    python scripts/calibrate_novelty.py
"""
from __future__ import annotations

import itertools
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from rba.ats.dense import embed
from rba.nodes.additions import DUPLICATE_COSINE, DUPLICATE_OVERLAP, word_overlap
from rba.resume.ingest import parse_bits

FIXTURE = pathlib.Path(__file__).resolve().parent.parent / "fixtures" / "aniket_agents_clean.txt"

REWORDINGS = {
    "LangGraph agent":
        "Engineered a LangGraph agent executing code inside an isolated Docker "
        "sandbox, with tooling for shell, Python, file I/O, HTTP and Playwright.",
    "34 unit tests":
        "Wrote 34 unit tests for the engine covering hand-counted timelines and "
        "trap policies designed to fail.",
    "discrete-event simulator":
        "Created a Python discrete-event simulation contrasting four text and "
        "voice channels for a debugging agent across 5 scripted sessions.",
    "PyTorch Lightning":
        "Implemented a PyTorch Lightning callback that records CIFAR-10 "
        "misclassifications and groups them by distortion type.",
    "action ledger":
        "Built an action ledger deciding whether to stop, hold or undo agent "
        "work on user interrupt.",
}


def main() -> int:
    real = [b.text for b in parse_bits(FIXTURE.read_text()).all_bullets()]
    pairs = []
    for fragment, reworded in REWORDINGS.items():
        original = next((t for t in real if fragment in t), None)
        if original:
            pairs.append((original, reworded))

    texts = real + [b for _a, b in pairs]
    matrix = embed(texts)
    if matrix is None:
        print("embedding model unavailable")
        return 1
    matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    index = {t: i for i, t in enumerate(texts)}

    def cosine(a: str, b: str) -> float:
        return float(np.dot(matrix[index[a]], matrix[index[b]]))

    dup = [(cosine(a, b), word_overlap(a, b)) for a, b in pairs]
    distinct = [(cosine(a, b), word_overlap(a, b))
                for a, b in itertools.combinations(real, 2)]

    print("same claim, reworded (must be caught):")
    for (c, o), (a, _b) in zip(dup, pairs):
        caught = c >= DUPLICATE_COSINE or o >= DUPLICATE_OVERLAP
        print(f"  {'OK ' if caught else 'MISS'}  cos {c:.3f}  overlap {o:.3f}  {a[:48]}")

    distinct.sort(key=lambda x: -x[0])
    print("\nmost similar genuinely-different pairs (must NOT be caught):")
    for c, o in distinct[:5]:
        flagged = c >= DUPLICATE_COSINE or o >= DUPLICATE_OVERLAP
        print(f"  {'FALSE' if flagged else 'ok   '} cos {c:.3f}  overlap {o:.3f}")

    dc = [d[0] for d in dup]
    sc = [d[0] for d in distinct]
    do = [d[1] for d in dup]
    so = [d[1] for d in distinct]
    print(f"\nduplicates  cos {min(dc):.3f}-{max(dc):.3f}  overlap {min(do):.3f}-{max(do):.3f}")
    print(f"distinct    cos {min(sc):.3f}-{max(sc):.3f}  overlap {min(so):.3f}-{max(so):.3f}")
    print(f"\ncosine gap  {min(dc) - max(sc):+.3f}   (threshold {DUPLICATE_COSINE})")
    print(f"overlap gap {min(do) - max(so):+.3f}   (threshold {DUPLICATE_OVERLAP})")

    missed = sum(1 for c, o in dup if c < DUPLICATE_COSINE and o < DUPLICATE_OVERLAP)
    false = sum(1 for c, o in distinct if c >= DUPLICATE_COSINE or o >= DUPLICATE_OVERLAP)
    print(f"\nmissed duplicates: {missed}   false duplicates: {false}")
    return 1 if (missed or false) else 0


if __name__ == "__main__":
    raise SystemExit(main())
