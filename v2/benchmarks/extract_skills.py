"""Fill the on-disk JD-skills cache (one LLM call per unique job description).

    RBA_SKILLS_CACHE=benchmarks/results/jd_skills_cache.json \
        uv run python -m benchmarks.extract_skills --n-jds 60

The job descriptions are public dataset text (nothing from a real user), so
sending them to Ollama's cloud models is fine; the resumes are never sent.
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor

from rba.ats import improved

from . import data


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jds", type=int, default=60)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args(argv)
    base = data.load_0xnbk()
    pairs = base + data.load_cnamuangtoun(base) + data.load_netsol()
    jds: dict[str, str] = {}
    for ds in ("0xnbk", "cnamuangtoun", "netsol"):
        sub = data.jd_subset([p for p in pairs if p.dataset == ds], args.n_jds)
        for p in sub:
            jds.setdefault(p.group, p.jd)
    print(f"{len(jds)} unique JDs to extract; cache {improved.CACHE_PATH}", flush=True)
    t0, done, failed = time.time(), 0, 0

    def work(jd):
        return improved.llm_jd_skills(jd)

    with ThreadPoolExecutor(args.workers) as ex:
        for res in ex.map(work, jds.values()):
            done += 1
            failed += res is None
            if done % 10 == 0:
                print(f"  {done}/{len(jds)} failed={failed} {time.time() - t0:.0f}s", flush=True)
    print(f"done: {done} JDs, {failed} failed, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
