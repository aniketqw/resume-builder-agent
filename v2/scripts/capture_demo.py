#!/usr/bin/env python3
"""Record the README's demo by driving the real application.

Committed so the assets can be regenerated rather than being mystery binaries
someone has to take on trust. Run it and you get exactly what is in `docs/`.

    .venv/bin/python scripts/capture_demo.py

Two deliberate choices:

**Screenshot frames, not `record_video_dir`.** Playwright can record video, but
the point of this UI is numbers changing, and a frame-by-frame script lets each
beat hold for exactly as long as it needs to be readable. A video of a cursor
moving is worth less than a still of a score that just moved.

**A synthetic resume.** `fixtures/demo_resume.txt` is an invented person. The
real fixture carries a phone number and an institutional email, and a GIF burns
text into a binary that the repository's secret scanner cannot read -- `grep -I`
skips binaries, and its pattern only matches API-key shapes anyway. So the demo
never loads the real one.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
REPO = V2.parent
DOCS = REPO / "docs"
FIXTURE = V2 / "fixtures" / "demo_resume.txt"

# Big enough that the two-pane layout is the real one (it collapses below
# 980px), small enough that the GIF stays inside its size budget.
VIEWPORT = {"width": 1240, "height": 780}

JD = ("Backend Engineer, Platform. Python, Go, Docker, Postgres, Redis, "
      "CI/CD, observability, testing, distributed systems, API design.")

GIF_FPS = 8
GIF_WIDTH = 1000
GIF_COLORS = 64
#: The repository's largest tracked file is 60 KB and there is no LFS, so
#: whatever lands here sits in history verbatim and for ever.
GIF_BUDGET = 2 * 1024 * 1024
PNG_BUDGET = 400 * 1024


def chromium_path() -> str | None:
    """Prefer the cached headless shell; fall back to Playwright's default."""
    cache = Path.home() / "Library/Caches/ms-playwright"
    for pattern in ("chromium_headless_shell-*/chrome-headless-shell-mac-arm64/"
                    "chrome-headless-shell",
                    "chromium-*/chrome-mac-arm64/Chromium.app/Contents/MacOS/Chromium"):
        for found in sorted(cache.glob(pattern), reverse=True):
            if found.exists():
                return str(found)
    return None


def ffmpeg_path() -> str | None:
    found = shutil.which("ffmpeg")
    if found:
        return found
    cache = Path.home() / "Library/Caches/ms-playwright"
    for candidate in sorted(cache.glob("ffmpeg-*/ffmpeg-mac"), reverse=True):
        if candidate.exists():
            return str(candidate)
    return None


class Recorder:
    """Screenshots on a fixed cadence, so timing is in frames not guesswork."""

    def __init__(self, page, outdir: Path) -> None:
        self.page = page
        self.dir = outdir
        self.n = 0

    def hold(self, frames: int = 1) -> None:
        """Capture `frames` identical frames: a pause the viewer can read."""
        for _ in range(frames):
            self.page.screenshot(path=str(self.dir / f"f{self.n:04d}.png"))
            self.n += 1

    def over(self, seconds: float, fps: int = GIF_FPS) -> None:
        """Capture continuously for `seconds` -- for things that animate."""
        deadline = time.time() + seconds
        interval = 1.0 / fps
        while time.time() < deadline:
            start = time.time()
            self.page.screenshot(path=str(self.dir / f"f{self.n:04d}.png"))
            self.n += 1
            time.sleep(max(0, interval - (time.time() - start)))


def script(page, rec: Recorder) -> None:
    """The demo, following what the product actually does."""
    page.wait_for_selector("#drop", timeout=20_000)
    rec.hold(8)                                     # the empty state

    # 1. Upload a resume. No LLM, no network -- it is a parser.
    page.set_input_files("#fileInput", str(FIXTURE))
    page.wait_for_selector("#jdText", timeout=20_000)
    rec.over(2.0)
    rec.hold(8)                                     # the parsed paper preview

    # 2. Pick a role. No job description yet -- the server synthesises one from
    #    the role's keyword profile and says so.
    page.click('[data-role="backend_platform_engineer"]')
    rec.over(4.5)
    rec.hold(12)                                    # score + stand-in caveat

    # 3. Paste a real posting. The score moves to something meaningful.
    page.fill("#jdText", JD)
    rec.over(5.0)
    rec.hold(10)

    # 4. Untick a line. The headline moves and every per-line delta stays live.
    checkbox = page.locator("[data-toggle]").first
    checkbox.uncheck()
    rec.over(5.0)
    rec.hold(14)

    # 5. Put it back -- served from cache, so it snaps.
    checkbox.check()
    rec.over(3.0)
    rec.hold(10)


def stills(page, context) -> None:
    """Light and dark, plus the import panel."""
    DOCS.mkdir(parents=True, exist_ok=True)
    # The script leaves the page scrolled part-way down; a hero shot wants the
    # top of the workbench, with the scorecard and the paper both in frame.
    page.evaluate("window.scrollTo(0, 0)")
    page.wait_for_timeout(400)
    page.screenshot(path=str(DOCS / "v2-workbench-light.png"))

    page.click("#themeBtn")
    page.wait_for_timeout(700)
    page.screenshot(path=str(DOCS / "v2-workbench-dark.png"))
    page.click("#themeBtn")
    page.wait_for_timeout(500)

    panel = page.locator("#ghBox")
    panel.scroll_into_view_if_needed()
    page.wait_for_timeout(500)
    panel.screenshot(path=str(DOCS / "v2-github-import.png"))


def build_gif(frames: Path, out: Path) -> None:
    """Palettised GIF. Two passes, because one-pass quantisation on a flat UI
    produces visible banding in the score bars."""
    ffmpeg = ffmpeg_path()
    if ffmpeg is None:
        raise SystemExit("ffmpeg not found; install it or use Playwright's bundled copy")
    palette = frames / "palette.png"
    common = ["-y", "-v", "error", "-framerate", str(GIF_FPS),
              "-i", str(frames / "f%04d.png")]
    scale = f"fps={GIF_FPS},scale={GIF_WIDTH}:-1:flags=lanczos"
    subprocess.run([ffmpeg, *common, "-vf",
                    f"{scale},palettegen=max_colors={GIF_COLORS}:stats_mode=diff",
                    str(palette)], check=True)
    subprocess.run([ffmpeg, *common, "-i", str(palette), "-lavfi",
                    f"{scale}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3",
                    str(out)], check=True)


def report(path: Path, budget: int) -> bool:
    size = path.stat().st_size
    ok = size <= budget
    print(f"  {'ok ' if ok else 'OVER'} {path.relative_to(REPO)}  "
          f"{size / 1024:.0f} KB (budget {budget / 1024:.0f} KB)")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=os.environ.get("RBA_URL",
                                                        "http://127.0.0.1:8000"))
    parser.add_argument("--keep-frames", action="store_true")
    args = parser.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise SystemExit("pip install playwright  (browsers are already cached)") from None

    frames = V2 / ".demo-frames"
    if frames.exists():
        shutil.rmtree(frames)
    frames.mkdir(parents=True)
    DOCS.mkdir(parents=True, exist_ok=True)

    errors: list[str] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=chromium_path())
        context = browser.new_context(viewport=VIEWPORT, device_scale_factor=1)
        page = context.new_page()
        page.on("pageerror", lambda e: errors.append(str(e)))
        print(f"driving {args.url}")
        page.goto(args.url, wait_until="networkidle")

        rec = Recorder(page, frames)
        script(page, rec)
        print(f"  captured {rec.n} frames")
        stills(page, context)
        browser.close()

    if errors:
        # A demo of a page that is throwing is a demo of a bug.
        print("page errors during capture:", *errors, sep="\n  ")
        return 1

    gif = DOCS / "v2-demo.gif"
    build_gif(frames, gif)
    if not args.keep_frames:
        shutil.rmtree(frames)

    print("\nwrote:")
    ok = report(gif, GIF_BUDGET)
    for png in sorted(DOCS.glob("*.png")):
        ok = report(png, PNG_BUDGET) and ok
    if not ok:
        print("\nover budget. Lower GIF_FPS, GIF_WIDTH or GIF_COLORS, or trim a beat.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
