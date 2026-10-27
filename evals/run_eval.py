"""Run the quality harness over a set of repositories and report the numbers.

Usage:
    python evals/run_eval.py                 # the default golden set
    python evals/run_eval.py repo1 repo2     # specific repos
    python evals/run_eval.py --no-geval      # faithfulness only (much cheaper)

Results go to evals/results/ and, when LANGSMITH_TRACING is on, to the
LangSmith project named by LANGSMITH_PROJECT.

Budget warning: G-Eval costs 4 dimensions x n samples per repo. At 8,000
tokens per minute that is the most expensive thing in this project, which is
why --no-geval exists and why the golden set is small.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evals"))

from judge import (  # noqa: E402
    coverage,
    faithfulness,
    g_eval,
    number_precision,
    summary_text,
    weighted_overall,
)

from rba.config import Config  # noqa: E402
from rba.groq_client import GroqClient  # noqa: E402
from rba.nodes.summarize import summarize_one  # noqa: E402
from rba.nodes.verify import verify_claim  # noqa: E402
from rba.state import ClaimStatus, RepoFacts  # noqa: E402

log = logging.getLogger("eval")

# Chosen to span the hard cases: a dense research repo, two agent projects, a
# Java service, a repo whose benchmark shows the author's approach losing, and
# a nearly-empty one that the agent should decline to dress up.
GOLDEN_SET = [
    "lecturelens-research",
    "AiAgnoSample",
    "AiAgentDemo",
    "spring-gemini-chat",
    "chronosModelImprovement",
    "AiTerminal",
]

# Thresholds. Faithfulness is blocking; the rest are tracked.
THRESHOLDS = {"faithfulness": 1.0, "number_precision": 1.0, "quality_mean": 4.0}


def load_facts(path: Path, names: list[str]) -> dict[str, RepoFacts]:
    raw = json.loads(path.read_text())
    facts = raw.get("facts", raw)
    return {k: RepoFacts(**v) for k, v in facts.items() if not names or k in names}


async def evaluate_repo(client: GroqClient, facts: RepoFacts,
                        run_geval: bool, n: int) -> dict:
    summary, evidence = await summarize_one(client, facts)
    summary.claims = [verify_claim(c, evidence) for c in summary.claims]
    text = summary_text(summary)

    result: dict = {
        "repo": facts.repo,
        "claims_total": len(summary.claims),
        "claims_verified": sum(1 for c in summary.claims
                               if c.status is ClaimStatus.VERIFIED),
        "claims_rejected": sum(1 for c in summary.claims
                               if c.status is ClaimStatus.REJECTED),
        "claims_needs_measure": sum(1 for c in summary.claims
                                    if c.status is ClaimStatus.NEEDS_MEASURE),
        "number_precision": round(number_precision(summary, evidence), 4),
        "coverage": round(coverage(summary), 4),
        "declined": bool(summary.not_claimable_reason),
    }
    if not text.strip():
        result |= {"faithfulness": 1.0, "quality_mean": 0.0, "note": "declined"}
        return result

    faith = await faithfulness(client, text, evidence)
    result |= {
        "faithfulness": round(faith.score, 4),
        "atomic_claims": faith.total,
        "unsupported": faith.unsupported,
    }
    if run_geval:
        quality = await g_eval(client, text, evidence, n=n)
        result |= {"quality": {k: round(v, 2) for k, v in quality.dimensions.items()},
                   "quality_mean": round(quality.mean, 3)}
    result["overall"] = weighted_overall(
        result["faithfulness"], result.get("quality_mean", 0.0) or 0.0,
        result["number_precision"])
    return result


def push_to_langsmith(results: list[dict], project: str) -> str | None:
    """Record the run as a LangSmith dataset plus per-example feedback."""
    if os.environ.get("LANGSMITH_TRACING", "").lower() not in ("1", "true", "yes"):
        return None
    try:
        from langsmith import Client
    except ImportError:
        return None
    try:
        client = Client()
        name = f"{project}-summarization"
        datasets = list(client.list_datasets(dataset_name=name))
        dataset = datasets[0] if datasets else client.create_dataset(
            dataset_name=name,
            description="Evidence-grounded repo summarization: faithfulness and G-Eval.")
        for r in results:
            client.create_example(
                dataset_id=dataset.id,
                inputs={"repo": r["repo"]},
                outputs={k: v for k, v in r.items() if k != "repo"},
            )
        return str(dataset.id)
    except Exception as exc:  # tracing must never fail the eval
        log.warning("LangSmith push failed: %s", exc)
        return None


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("repos", nargs="*", help="Repos to evaluate (default: golden set)")
    ap.add_argument("--facts", default="/tmp/m2_state.json",
                    help="Fact bundle produced by a prior run")
    ap.add_argument("--no-geval", action="store_true", help="Skip G-Eval (cheaper)")
    ap.add_argument("--n", type=int, default=3, help="G-Eval samples per dimension")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    cfg = Config.load()
    names = args.repos or GOLDEN_SET
    facts_path = Path(args.facts)
    if not facts_path.exists():
        print(f"No fact bundle at {facts_path}. Run `rba run <user>` first.")
        return 2
    facts = load_facts(facts_path, names)
    if not facts:
        print(f"None of {names} found in {facts_path}.")
        return 2

    started = time.time()
    results = []
    async with GroqClient(cfg) as client:
        for name, f in facts.items():
            log.info("evaluating %s", name)
            try:
                results.append(await evaluate_repo(client, f, not args.no_geval, args.n))
            except Exception as exc:
                log.warning("eval failed for %s: %s", name, exc)
                results.append({"repo": name, "error": str(exc)[:200]})
        usage = client.usage_report()

    ok = [r for r in results if "error" not in r]
    agg = {}
    for metric in ("faithfulness", "number_precision", "quality_mean", "overall"):
        vals = [r[metric] for r in ok if r.get(metric) is not None]
        if vals:
            agg[metric] = round(sum(vals) / len(vals), 4)
    agg["repos"] = len(ok)
    agg["elapsed_s"] = round(time.time() - started, 1)
    agg["tokens"] = sum(v["total"] for v in usage.values())

    out_dir = ROOT / "evals" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "latest.json"
    out.write_text(json.dumps({"aggregate": agg, "results": results}, indent=1))

    print(f"\n{'repo':<28}{'faith':>7}{'numPrec':>9}{'qual':>7}{'verified':>10}{'rej':>5}")
    for r in results:
        if "error" in r:
            print(f"{r['repo'][:27]:<28}  ERROR {r['error'][:40]}")
            continue
        print(f"{r['repo'][:27]:<28}{r.get('faithfulness', 0):>7.2f}"
              f"{r.get('number_precision', 0):>9.2f}{r.get('quality_mean', 0):>7.2f}"
              f"{r.get('claims_verified', 0):>6}/{r.get('claims_total', 0):<3}"
              f"{r.get('claims_rejected', 0):>5}")
    print(f"\naggregate: {agg}")

    failures = []
    for metric, threshold in THRESHOLDS.items():
        if metric in agg and agg[metric] < threshold:
            failures.append(f"{metric} {agg[metric]} < {threshold}")
    for r in ok:
        for claim in r.get("unsupported", []):
            print(f"  UNSUPPORTED [{r['repo']}]: {claim[:100]}")

    ds = push_to_langsmith(results, os.environ.get("LANGSMITH_PROJECT", "rba"))
    if ds:
        print(f"LangSmith dataset: {ds}")
    print(f"results: {out}")

    if failures:
        print("\nBELOW THRESHOLD: " + "; ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
