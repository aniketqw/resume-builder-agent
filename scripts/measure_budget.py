"""Measure what one repo actually costs, then say whether 110 fit in a day.

The plan's 275K-token figure was an estimate of mine. This replaces it with a
measurement: run the real pipeline over N repos, read the usage Groq itself
reported, and extrapolate against the two daily quotas.

    python scripts/measure_budget.py --repos 10 --user aniketqw

Writes evals/budget.json and prints the verdict.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from rba.config import Config
from rba.resume.pipeline import build_from_github

# Groq free tier. RPD and TPM are confirmed in response headers; TPD is from
# the docs only, which is why it is labelled as such in the output.
LIMIT_RPD = 1_000
LIMIT_TPD = 200_000
LIMIT_TPM = 8_000


def _load_env() -> None:
    env = pathlib.Path(__file__).resolve().parent.parent / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


async def main() -> int:
    _load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos", type=int, default=10)
    ap.add_argument("--user", default=os.environ.get("GITHUB_USERNAME") or "aniketqw")
    ap.add_argument("--target", type=int, default=110,
                    help="repo count to extrapolate to")
    ap.add_argument("--out", default="evals/budget.json")
    ap.add_argument("--cold", action="store_true",
                    help="bypass the local response cache, to measure a "
                         "first-ever run rather than a warm re-run")
    args = ap.parse_args()

    cfg = Config.load()

    started = time.time()
    usage_holder: dict = {}

    # build_from_github closes its client, so reach the per-node report through
    # the stats dict it returns plus a monkeypatched report capture.
    from rba import groq_client as gc

    real_report = gc.GroqClient.usage_report

    def capture(self):
        out = real_report(self)
        usage_holder.update(out)
        return out

    gc.GroqClient.usage_report = capture
    real_cache = gc.Cache
    if args.cold:
        gc.Cache = lambda *a, **k: real_cache(enabled=False)
    try:
        _doc, stats = await build_from_github(
            args.user,
            github_token=os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN", ""),
            groq_key=os.environ.get("GROQ_API_KEY", ""),
            jev_key=os.environ.get("TYPESAFE_API_KEY", ""),
            max_repos=args.repos,
            config=cfg,
            progress=lambda m, f: print(f"  [{f:>5.0%}] {m}", flush=True),
        )
    finally:
        gc.GroqClient.usage_report = real_report
        gc.Cache = real_cache

    elapsed = time.time() - started
    summarised = max(stats.get("selected", 0), 1)

    tot_req = sum(v.get("requests", 0) for v in usage_holder.values())
    tot_tok = sum(v.get("total", 0) for v in usage_holder.values())
    tot_prompt = sum(v.get("prompt", 0) for v in usage_holder.values())
    tot_cached = sum(v.get("cached_tokens", 0) for v in usage_holder.values())

    per_repo_req = tot_req / summarised
    per_repo_tok = tot_tok / summarised
    proj_req = per_repo_req * args.target
    proj_tok = per_repo_tok * args.target

    result = {
        "measured": {
            "user": args.user,
            "repos_summarised": summarised,
            "repos_found": stats.get("repos_found"),
            "elapsed_s": round(elapsed, 1),
            "cold": bool(args.cold),
            "requests": tot_req,
            "tokens": tot_tok,
            "prompt_tokens": tot_prompt,
            "cached_prompt_tokens": tot_cached,
            "by_node": usage_holder,
        },
        "per_repo": {
            "requests": round(per_repo_req, 2),
            "tokens": round(per_repo_tok, 1),
        },
        "projected": {
            "repos": args.target,
            "requests": round(proj_req),
            "tokens": round(proj_tok),
            "pct_of_rpd": round(100 * proj_req / LIMIT_RPD, 1),
            "pct_of_tpd_documented": round(100 * proj_tok / LIMIT_TPD, 1),
            "minutes_at_tpm_floor": round(proj_tok / LIMIT_TPM, 1),
        },
        "limits": {"rpd_header_confirmed": LIMIT_RPD,
                   "tpd_docs_only": LIMIT_TPD,
                   "tpm_header_confirmed": LIMIT_TPM},
        "fits_in_one_day": bool(proj_req <= LIMIT_RPD and proj_tok <= LIMIT_TPD),
    }

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    p = result["projected"]
    print("\n--- measured ---")
    print(f"  {summarised} repos -> {tot_req} requests, {tot_tok} tokens "
          f"in {elapsed:.0f}s")
    for node, v in sorted(usage_holder.items()):
        print(f"    {node:<20} {v.get('requests', 0):>3} req  "
              f"{v.get('total', 0):>7} tok  "
              f"{v.get('cached_tokens', 0):>6} cached")
    print(f"\n--- projected to {args.target} repos ---")
    print(f"  requests {p['requests']:>6}  = {p['pct_of_rpd']}% of the 1,000/day limit")
    print(f"  tokens   {p['tokens']:>6}  = {p['pct_of_tpd_documented']}% of the "
          f"documented 200K/day cap")
    print(f"  wall clock at the 8K/min floor: {p['minutes_at_tpm_floor']} min")
    print(f"\n  fits in one day: {result['fits_in_one_day']}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
