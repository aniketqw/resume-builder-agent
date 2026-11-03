# Resume bullets for this project

Two versions of the same entry, tuned for different roles. Every number below
is traceable to a file in this repository or to a recorded eval run — the same
rule `nodes/verify.py` applies to the claims this tool generates about *your*
repositories. The evidence trail is at the bottom; if a figure ever stops being
true, the source is named so it can be corrected rather than quietly carried.

---

## Version A — AI / LLM engineer roles

**Resume Builder Agent — LLM Agents** · Oct 2026 – Present
*Tech: Python, LangGraph, LangChain, Groq, GitHub MCP, FastAPI, ONNX, Docker, pytest*

- Built a **12-node LangGraph agent** with SQLite checkpointing and a
  human-review interrupt that reads a GitHub account through the **read-only
  GitHub MCP server** and turns each repository into evidence-linked claims.
- Wrote **claim verification as plain Python, not a prompt**: every number and
  quote in a generated claim must appear in that repository's evidence, or the
  claim is rejected or demoted to a measurement task — covered by 28 tests, and
  measured at **0.92 faithfulness and 1.00 number-precision across 39 claims
  from 6 repositories**.
- Scored resume-against-posting with a **three-channel fused model** (dense ONNX
  embeddings 0.5, BM25 0.2, skills 0.3), choosing convex combination over
  Reciprocal Rank Fusion on published nDCG evidence; a channel that cannot be
  computed has its weight redistributed rather than being scored as zero.
- **Rejected a 4,253× faster embedding approximation** that got the sign wrong
  on 6 of 20 bullets — it would tell a user a line helps when it hurts — and
  kept the exact leave-one-out path, then made it affordable by batching:
  **4.9 s → 0.37 s, bit-identical output**.
- Held generation to **one request and ~3,000 tokens per repository** behind a
  SQLite cache keyed on repo + HEAD SHA + prompt version; a warm re-run of 10
  repositories measured **0 requests and 0 tokens**.

---

## Version B — Backend / platform roles

**Resume Builder Agent — Evidence-grounded resume service** · Oct 2026 – Present
*Tech: Python, FastAPI, Docker, ONNX Runtime, SQLite, LangGraph, pytest*

- Built a **FastAPI service and a dependency-free browser frontend** (14 ES
  modules, 6 stylesheets, no bundler and no `package.json`) over a Python
  scoring engine, with SSE-streamed progress for multi-minute jobs and
  optimistic concurrency on the shared working document.
- Made per-control scoring viable: one screen needs **~145 document scores at
  290 ms each**. Exploited the embedding model's 512-token cut to dedupe
  candidates sharing a prefix — **16 variants collapse to 4 forward passes** —
  taking exact leave-one-out attribution from **4.9 s to 0.37 s**, asserted
  equal to the unbatched path **to float equality** in tests.
- Shipped a **multi-stage Docker image**: non-root, embedding model baked in so
  the container scores with no network access, and a healthcheck that asserts
  the scoring model actually loaded rather than mere process liveness.
- Ran **ONNX Runtime instead of PyTorch** to hold the service near **150 MB
  resident against 400–700 MB**, and made thread counts configurable
  (**297 ms → 83 ms** per embedding at 4 threads) — deliberately single-worker,
  since a per-worker rate limiter would multiply the provider budget by the
  worker count.
- Built a **per-model limiter that reserves an estimated spend, then reconciles
  against the provider's `x-ratelimit-*` headers** across TPM/RPM/RPD with a
  daily-budget stop; **540 offline tests**, none requiring a credential.

---

## Evidence trail

| Claim | Source |
|---|---|
| 12-node graph, SQLite checkpointing | `v1/src/rba/graph.py:136-147`, `make_checkpointer` `:168` |
| human-review interrupt | `v1/src/rba/graph.py:97` — dynamic `interrupt()` from `langgraph.types` |
| read-only MCP | `v1/src/rba/github_tools.py:259` (`--read-only`), hosted `…/mcp/x/repos/readonly` |
| verification is code, not a prompt | `v1/src/rba/nodes/verify.py:1` and `verify_claim` `:169` |
| 28 tests | `pytest v1/tests/test_verify.py --collect-only` → 28 (21 `def test_`, parametrised) |
| 0.92 faithfulness, 1.00 number-precision | `v1/evals/results/latest.json` → `aggregate` |
| 39 claims / 36 verified / 2 rejected / 1 needs-measure, 6 repos | same file, `results[]` summed |
| channel weights 0.5 / 0.2 / 0.3 | `v1/src/rba/ats/fusion.py:28` `CHANNEL_WEIGHTS` |
| convex combination over RRF | `fusion.py:3-10`, citing Bruch et al., arXiv 2210.11934 |
| weight redistributed, not zeroed | `fusion.py` `convex_fuse` docstring |
| 4,253× faster, sign wrong 6 of 20 | `v1/src/rba/resume/attribution.py:13-28` |
| 4.9 s exact leave-one-out | `attribution.py:13` |
| 4.9 s → 0.37 s, bit-identical | commit `b312690`; `v2/tests/test_batch.py` asserts float equality |
| ~145 scores at 290 ms | `v2/src/rba/ats/batch.py:3-5` |
| 16 variants → 4 forward passes | `v2/src/rba/ats/batch.py:12-13` |
| 1 request / 3,058 tokens per repo | `v1/evals/budget.md:19` |
| warm re-run: 0 requests, 0 tokens | `v1/evals/budget.md:91` (108 s, vs 352 s cold) |
| ~150 MB vs 400–700 MB resident | `v2/src/rba/ats/dense.py:12-14` |
| 297 ms → 83 ms per embedding | measured; `v2/src/rba/ats/dense.py:100` |
| non-root, model pre-baked, healthcheck | `v2/Dockerfile` — `USER app`, `models` stage, `HEALTHCHECK` |
| 14 JS + 6 CSS, no bundler | `v2/web/`; no tracked `package.json` |
| SSE progress | `v2/src/rba_api/jobs.py`, `routes/github.py` `StreamingResponse` |
| single-worker rationale | `v2/src/rba_api/app.py:1-8` |
| limiter: reserve then reconcile | `v1/src/rba/limiter.py` — `reserve()` / `reconcile()` on `x-ratelimit-*` |
| 540 offline tests | `cd v2 && pytest --collect-only -q` → 540 (v1 collects 495) |

### Two numbers deliberately *not* claimed

- **"199 offline tests"**, from the previous version of these bullets, matches
  nothing in either tree. v1 collects 495 and v2 collects 540.
- **The combined 1,035.** v2 vendors the engine, so the two suites overlap
  heavily and adding them would be double-counting — the exact inflation this
  project exists to prevent. Cite one, or state both separately.

### One claim retired

The earlier bullet led with a *"deterministic 100-point ATS scorer."* It exists
(`v1/src/rba/scoring.py` `WEIGHTS` sums to exactly 100), but
`v1/src/rba/resume/diagnostics.py:1-16` disowns it as the headline:

> Those weights were invented. Nobody measured that a missing phone number is
> worth exactly six points, and presenting a number built from invented weights
> as "your ATS score" is the same category of error as inventing an achievement.
> So the weights are not used here. The headline number stays the three-channel
> fused score.

Leading with it claimed the weaker scorer and omitted the better one. The rubric
survives as diagnostics — *which keyword is missing, is this too long* — which is
worth a clause, not a headline.
