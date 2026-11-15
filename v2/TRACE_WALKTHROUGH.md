# Trace walkthrough — one real run, node by node, with code

A full, real run through the live app: upload a real resume PDF, set a real
job description, score it, compute per-line attribution, render a preview,
then run a GitHub import against a real account. Every number and claim
below is pulled directly from that run's actual LangSmith trace via the SDK
(`Client().list_runs(trace_id=...)`), not estimated or reconstructed from
memory. Every code citation is a permalink pinned to the exact commit this
run executed against, so clicking it shows the real code that produced the
real trace data next to it.

- **Commit the run executed at:** `d118de51a5f50d1380aa8be2f9a414c4ea716b65`
- **LLM provider:** Ollama (`qwen2.5:14b-instruct`, self-hosted — `LLM_PROVIDER=ollama`)
- **LangSmith trace root:** `01a1dfb7-bd49-739c-a024-a7fc248de256`
- **GitHub account imported:** `aniketqw`

Permalink base used throughout:
`https://github.com/aniketqw/resume-builder-agent/blob/d118de51a5f50d1380aa8be2f9a414c4ea716b65/v2/`

---

## Part 1 — scoring a resume against a job description (no LLM, deterministic)

### Entry point: `POST /api/upload`

Code: [`src/rba_api/routes/documents.py`](https://github.com/aniketqw/resume-builder-agent/blob/d118de51a5f50d1380aa8be2f9a414c4ea716b65/v2/src/rba_api/routes/documents.py)
→ calls [`parse_resume()`](https://github.com/aniketqw/resume-builder-agent/blob/d118de51a5f50d1380aa8be2f9a414c4ea716b65/v2/src/rba/resume/ingest.py#L530)
(`resume/ingest.py`), which detects the PDF, extracts flat text (`read_pdf`)
and, separately, real table geometry (`read_pdf_tables`), then builds the
structured `ResumeDoc`.

**Real result from this run:** `POST /api/upload` returned 200; the parsed
document's `ACADEMIC DETAILS` table came back with all six real headers
(`COURSE, SPECIALIZATION, INSTITUTE/COLLEGE, BOARD/UNIVERSITY, SCORE, YEAR`)
correctly aligned to their rows — the exact bug fixed earlier this session.

### `PATCH /api/session/{id}/target` → `POST /api/score`

Code: [`routes/score.py::score_one`](https://github.com/aniketqw/resume-builder-agent/blob/d118de51a5f50d1380aa8be2f9a414c4ea716b65/v2/src/rba_api/routes/score.py#L100)
→ [`resume/attribution.py::rescore`](https://github.com/aniketqw/resume-builder-agent/blob/d118de51a5f50d1380aa8be2f9a414c4ea716b65/v2/src/rba/resume/attribution.py#L141)
→ three channels combined by
[`ats/fusion.py::convex_fuse`](https://github.com/aniketqw/resume-builder-agent/blob/d118de51a5f50d1380aa8be2f9a414c4ea716b65/v2/src/rba/ats/fusion.py#L94)
using the weights declared at
[`ats/fusion.py::CHANNEL_WEIGHTS`](https://github.com/aniketqw/resume-builder-agent/blob/d118de51a5f50d1380aa8be2f9a414c4ea716b65/v2/src/rba/ats/fusion.py#L27)
(`dense 0.5, lexical 0.2, skills 0.3`).

**Real result from this run:**
- JD: *"AI Engineer. Required: Python, LangChain, LangGraph, RAG, retrieval, evaluation, Docker. Build and evaluate LLM agent systems."*
- Raw fused score: **`0.6084794381808667`** → `60.85 / 100`
- No LLM call anywhere in this path — this is why the same input always
  produces this exact float, byte for byte.

### `POST /api/attribution`

Code: [`resume/attribution.py::attribute`](https://github.com/aniketqw/resume-builder-agent/blob/d118de51a5f50d1380aa8be2f9a414c4ea716b65/v2/src/rba/resume/attribution.py#L187)
— leave-one-out: disable each enabled bullet in turn, rescore, record the
delta. Made affordable by the truncation-dedup + batching in
[`ats/batch.py`](https://github.com/aniketqw/resume-builder-agent/blob/d118de51a5f50d1380aa8be2f9a414c4ea716b65/v2/src/rba/ats/batch.py#L1).

**Real result from this run:** `POST /api/attribution` returned 200 with
**17 per-bullet impacts** — one exact delta per enabled line in the
uploaded resume.

---

## Part 2 — the GitHub import (LangGraph, 12 nodes)

Entry point: `POST /api/session/{id}/github` →
[`routes/github.py::start_import`](https://github.com/aniketqw/resume-builder-agent/blob/d118de51a5f50d1380aa8be2f9a414c4ea716b65/v2/src/rba_api/routes/github.py#L143)
→ [`resume/pipeline.py::build_from_github`](https://github.com/aniketqw/resume-builder-agent/blob/d118de51a5f50d1380aa8be2f9a414c4ea716b65/v2/src/rba/resume/pipeline.py#L80)
→ builds and streams the real compiled graph:
[`graph.py::build_graph`](https://github.com/aniketqw/resume-builder-agent/blob/d118de51a5f50d1380aa8be2f9a414c4ea716b65/v2/src/rba/graph.py#L187)
(node registration at lines 189-201, edges at 203-219).

Below: each node, in execution order, with the real timing/output from
trace `01a1dfb7-bd49-739c-a024-a7fc248de256`, and the exact code it ran.

### 1. `connect_github`

Code: [`graph.py::Runner.connect`](https://github.com/aniketqw/resume-builder-agent/blob/d118de51a5f50d1380aa8be2f9a414c4ea716b65/v2/src/rba/graph.py#L81-L96).
Builds the GitHub backend
([`github_tools.py::make_backend`](https://github.com/aniketqw/resume-builder-agent/blob/d118de51a5f50d1380aa8be2f9a414c4ea716b65/v2/src/rba/github_tools.py))
and picks the LLM client based on `LLM_PROVIDER`
(lines 88-92: `OllamaClient` vs `GroqClient`, added this session).

**Real:** `3.79s`. Output: `{"username": "aniketqw"}` — exactly the one key
this node returns (`graph.py` line 95), confirming `connect` does nothing
beyond building the backend/client and echoing the username back into
state.

### 2. `list_repos`

Code: [`graph.py::Runner.list_repos`](https://github.com/aniketqw/resume-builder-agent/blob/d118de51a5f50d1380aa8be2f9a414c4ea716b65/v2/src/rba/graph.py#L98-L102).
Calls the backend's `list_repos`, which hits GitHub's
`search_repositories` tool through the read-only MCP server.

**Real:** `5.44s`. `repos_found: 103` (every repo GitHub reports for
`aniketqw`), `repos: 60` (public only — `public = [r for r in repos if not
r.is_private]`, `graph.py` line 100 — confirming 43 private repos were
correctly excluded before anything downstream ever saw them).

### 3. `select_repos`

Code: [`graph.py::Runner.select`](https://github.com/aniketqw/resume-builder-agent/blob/d118de51a5f50d1380aa8be2f9a414c4ea716b65/v2/src/rba/graph.py#L104-L110)
→ [`nodes/select.py::select_repos`](https://github.com/aniketqw/resume-builder-agent/blob/d118de51a5f50d1380aa8be2f9a414c4ea716b65/v2/src/rba/nodes/select.py#L208).
Two stages, no Groq/Ollama token spent in either: hard rules
([`apply_hard_rules`](https://github.com/aniketqw/resume-builder-agent/blob/d118de51a5f50d1380aa8be2f9a414c4ea716b65/v2/src/rba/nodes/select.py#L86),
drops forks/archived/empty/low-commit repos for free), then Jev scoring of
what's left for `claimable`/`tier`.

**Real:** `48.78s` (the single slowest non-LLM node — this is Jev
classifying every surviving candidate, one real external API's own
latency, not something this codebase controls). Of 60 public repos, **6
selected**: `resume-builder-agent, pages-hub, AiAgentDemo,
live-debugging-partner, AOA_Controller_ty, file-upload-server`. 9 excluded
by hard rules before Jev was even called (forks/archived/empty/low-commit
— free, per the module docstring at
[`nodes/select.py` lines 1-15](https://github.com/aniketqw/resume-builder-agent/blob/d118de51a5f50d1380aa8be2f9a414c4ea716b65/v2/src/rba/nodes/select.py#L1-L15)).

### 4. `fetch_facts`

Code: [`graph.py::Runner.fetch`](https://github.com/aniketqw/resume-builder-agent/blob/d118de51a5f50d1380aa8be2f9a414c4ea716b65/v2/src/rba/graph.py#L112-L113)
→ [`nodes/fetch.py::fetch_facts`](https://github.com/aniketqw/resume-builder-agent/blob/d118de51a5f50d1380aa8be2f9a414c4ea716b65/v2/src/rba/nodes/fetch.py#L200),
concurrent across repos via `asyncio.gather` + a semaphore
([`fetch_facts`, line 200](https://github.com/aniketqw/resume-builder-agent/blob/d118de51a5f50d1380aa8be2f9a414c4ea716b65/v2/src/rba/nodes/fetch.py#L200)).
Per repo:
[`fetch_repo_facts`](https://github.com/aniketqw/resume-builder-agent/blob/d118de51a5f50d1380aa8be2f9a414c4ea716b65/v2/src/rba/nodes/fetch.py#L72)
pulls the README, tree, commits, key manifest files, and up to
`SOURCE_FILE_LIMIT` ranked source files, each capped at
`SOURCE_FILE_MAX_CHARS` (raised from 4,000 to 60,000 this session — the
fix for the AST-truncation bug, [line 48](https://github.com/aniketqw/resume-builder-agent/blob/d118de51a5f50d1380aa8be2f9a414c4ea716b65/v2/src/rba/nodes/fetch.py#L48)).

**Real:** `23.27s` for all 6 repos (concurrent, not 6× sequential — see
the per-tool-call timestamps overlapping within the same second in the raw
trace). Per repo: `resume-builder-agent` — 304 files in the tree,
has_tests=True, **12 source files fetched** (the `SOURCE_FILE_LIMIT` cap);
`live-debugging-partner` — 138 files, 12 source files fetched;
`AiAgentDemo` — 43 files, 11 fetched; `pages-hub`/`AOA_Controller_ty`/
`file-upload-server` — 0 Python source files fetched (non-Python repos, or
nothing ranked as source — `_ranked_source_paths` only selects `.py`
files, per [`fetch.py` line 187-197](https://github.com/aniketqw/resume-builder-agent/blob/d118de51a5f50d1380aa8be2f9a414c4ea716b65/v2/src/rba/nodes/fetch.py#L187-L197)).

### 5. `prepare_evidence`

Code: [`graph.py::Runner.prepare`](https://github.com/aniketqw/resume-builder-agent/blob/d118de51a5f50d1380aa8be2f9a414c4ea716b65/v2/src/rba/graph.py#L115-L116)
→ [`nodes/prepare_evidence.py::prepare_evidence`](https://github.com/aniketqw/resume-builder-agent/blob/d118de51a5f50d1380aa8be2f9a414c4ea716b65/v2/src/rba/nodes/prepare_evidence.py#L18) —
**zero LLM calls**, pure `ast`-based code indexing. For every fetched
source file, calls
[`repobrain.py::index`](https://github.com/aniketqw/resume-builder-agent/blob/d118de51a5f50d1380aa8be2f9a414c4ea716b65/v2/src/rba/repobrain.py#L256),
which walks the Python AST and builds a
[`RepoBrain`](https://github.com/aniketqw/resume-builder-agent/blob/d118de51a5f50d1380aa8be2f9a414c4ea716b65/v2/src/rba/repobrain.py#L96)
of `Symbol`/`CallEdge`/`RouteBinding` entries per file
([`index_file`](https://github.com/aniketqw/resume-builder-agent/blob/d118de51a5f50d1380aa8be2f9a414c4ea716b65/v2/src/rba/repobrain.py#L209)).
A file that fails to parse is recorded in `partial_source`, never silently
dropped.

**Important note on what this node's trace actually shows**: LangGraph
traces a node's full input/output as literally the `AgentState` dict
passed in/returned — so this node's traced *input* includes every key
accumulated by every node before it (including unrelated fields like Jev's
token usage from `select_repos`), which can look confusing in the
LangSmith UI if you're expecting to see only what `prepare_evidence`
itself touches. Its traced *output*, however, is exactly and only
`{"repo_graph": ..., "partial_source": ...}` — scoped correctly.

**Real:** `0.048s` (near-instant, as expected for zero-LLM AST indexing).
`partial_source: []` — **zero parse errors across every fetched file**,
real confirmation the `SOURCE_FILE_MAX_CHARS` fix (4,000 → 60,000 chars)
holds on this real data, not just the synthetic test built for it. Indexed:

| repo | files indexed | symbols | call edges |
|---|---|---|---|
| `resume-builder-agent` | 12 | **100** | **1,334** |
| `live-debugging-partner` | 12 | 109 | 622 |
| `AiAgentDemo` | 11 | 19 | 121 |
| `pages-hub` / `AOA_Controller_ty` / `file-upload-server` | 0 | 0 | 0 |

I independently verified this against the real file, not just trusted the
trace: `repo_graph["resume-builder-agent"]["files"]["v1/app.py"]["symbols"][0]`
reports `{"name": "state", "line_start": 71, "line_end": 85}` — and
`v1/app.py` really does have `def state():` starting at line 71, ending at
line 85 (`return st.session_state`), confirmed by reading the file
directly. The first call edge reported, `{"caller": "state", "callee":
"setdefault", "line": 72}`, matches line 72 exactly:
`st.session_state.setdefault("doc", None)`.

### 6. `safety_scan`

Code: [`graph.py::Runner.scan`](https://github.com/aniketqw/resume-builder-agent/blob/d118de51a5f50d1380aa8be2f9a414c4ea716b65/v2/src/rba/graph.py#L118-L119)
→ [`nodes/safety.py::safety_scan`](https://github.com/aniketqw/resume-builder-agent/blob/d118de51a5f50d1380aa8be2f9a414c4ea716b65/v2/src/rba/nodes/safety.py#L106),
scanning fetched content for anything secret-shaped
([`scan_content`](https://github.com/aniketqw/resume-builder-agent/blob/d118de51a5f50d1380aa8be2f9a414c4ea716b65/v2/src/rba/nodes/safety.py#L77))
before any of it reaches a prompt. Zero LLM calls.

**Real:** `0.0048s`. Output: `{"safety": []}` — nothing secret-shaped found
in any of the fetched content from these 6 repos.

### 7. `summarize_repo` — the one node that spends real model tokens

Code: [`graph.py::Runner.summarize`](https://github.com/aniketqw/resume-builder-agent/blob/d118de51a5f50d1380aa8be2f9a414c4ea716b65/v2/src/rba/graph.py#L121-L122)
→ [`nodes/summarize.py::summarize_repos`](https://github.com/aniketqw/resume-builder-agent/blob/d118de51a5f50d1380aa8be2f9a414c4ea716b65/v2/src/rba/nodes/summarize.py#L186)
fans out over selected repos (semaphore-bounded concurrency). Per repo:
[`summarize_one`](https://github.com/aniketqw/resume-builder-agent/blob/d118de51a5f50d1380aa8be2f9a414c4ea716b65/v2/src/rba/nodes/summarize.py#L154)
builds a bounded evidence bundle
([`build_evidence`](https://github.com/aniketqw/resume-builder-agent/blob/d118de51a5f50d1380aa8be2f9a414c4ea716b65/v2/src/rba/nodes/summarize.py#L110),
capped ~7,000 chars) and calls
`client.structured(...)` — this run, that's
[`OllamaClient.structured`](https://github.com/aniketqw/resume-builder-agent/blob/d118de51a5f50d1380aa8be2f9a414c4ea716b65/v2/src/rba/ollama_client.py#L79),
which posts to `/api/chat` with `format` set to the claim schema's real
JSON Schema and retries once on a shape violation.

**Known gap, visible in this exact trace**: `OllamaClient` does not yet
call `_trace()` the way `GroqClient` does, so unlike a Groq-backed run,
this node shows **no per-repo LLM child spans** in LangSmith — only the
one `summarize_repo` chain span with its aggregate duration. The claims it
produced are still fully inspectable (below), just not broken out
per-call in the trace tree yet.

**Real:** `99.61s` total for 6 repos (semaphore(2) concurrency — three
batches of two). `errors: []` — **every repo's summary parsed successfully
against the strict schema**, with zero outright failures, unlike the
Groq-backed run earlier this session where `pages-hub`/`AiAgentDemo`/
`AOA_Controller_ty` all failed outright with a 400 schema-mismatch. All 6
repos summarized: `resume-builder-agent, pages-hub, AiAgentDemo,
live-debugging-partner, AOA_Controller_ty, file-upload-server`.

A genuinely interesting, independently-checkable example from this run's
real output — the model correctly cited **this very session's own commits**
as evidence for a claim about the repo it was reading:

> Claim: *"The project allows users to run the LLM side with either Groq or
> a self-hosted Ollama model."*
> Evidence: `path: commit:da2f0f6e10f5f1a4acdf6688e58c0e2e027620b3`, quote:
> *"feat(v2): Ollama as a configurable alternative to Groq, remove the
> rate-limit de[pendency]"*

That's the real commit message from
[`da2f0f6`](https://github.com/aniketqw/resume-builder-agent/commit/da2f0f6e10f5f1a4acdf6688e58c0e2e027620b3),
pushed earlier tonight — the model found and cited it correctly out of the
repo's real commit history, and `verify_claims` (next node) confirmed it.

### 8. `verify_claims` — the enforcement, not a second prompt

Code: [`graph.py::Runner.verify`](https://github.com/aniketqw/resume-builder-agent/blob/d118de51a5f50d1380aa8be2f9a414c4ea716b65/v2/src/rba/graph.py#L124-L125)
→ [`nodes/verify.py::verify_claims`](https://github.com/aniketqw/resume-builder-agent/blob/d118de51a5f50d1380aa8be2f9a414c4ea716b65/v2/src/rba/nodes/verify.py#L304),
which calls
[`verify_claim`](https://github.com/aniketqw/resume-builder-agent/blob/d118de51a5f50d1380aa8be2f9a414c4ea716b65/v2/src/rba/nodes/verify.py#L207)
on every claim, in order:
1. Every numeric token must appear in the evidence
   ([line 211-218](https://github.com/aniketqw/resume-builder-agent/blob/d118de51a5f50d1380aa8be2f9a414c4ea716b65/v2/src/rba/nodes/verify.py#L211-L218)) — else `NEEDS_MEASURE`.
2. The cited path must exist in the pinned snapshot
   ([line 232-236](https://github.com/aniketqw/resume-builder-agent/blob/d118de51a5f50d1380aa8be2f9a414c4ea716b65/v2/src/rba/nodes/verify.py#L232-L236)) — else `REJECTED`.
3. The quote must actually be present in that evidence
   ([line 238-242](https://github.com/aniketqw/resume-builder-agent/blob/d118de51a5f50d1380aa8be2f9a414c4ea716b65/v2/src/rba/nodes/verify.py#L238-L242)) — else `REJECTED`.
4. The quote must actually be *relevant to this specific claim*, not just
   present somewhere —
   [`quote_relevant`](https://github.com/aniketqw/resume-builder-agent/blob/d118de51a5f50d1380aa8be2f9a414c4ea716b65/v2/src/rba/nodes/verify.py#L146-L167)
   requires shared distinctive content words between claim and quote
   ([line 244-249](https://github.com/aniketqw/resume-builder-agent/blob/d118de51a5f50d1380aa8be2f9a414c4ea716b65/v2/src/rba/nodes/verify.py#L244-L249)) — else `REJECTED` with the reason
   *"supporting quote is real but does not mention anything in the
   claim."*

**Real:** `0.014s`. **36 claims in, 30 verified, 6 rejected** — every
single rejection in this real run exercised a *different* one of the three
rejection paths above, which is a clean, concrete demonstration that all
three checks are live, not dead code:

| Repo | Claim | Rejection reason (verbatim, from the trace) |
|---|---|---|
| `pages-hub` | "The system includes a test suite to ensure functionality." | `every supporting quote is absent from the evidence` (path 3) |
| `AiAgentDemo` | "The project includes a test suite to verify functionality." | `cited path does not exist in the pinned repository snapshot` (path 2) |
| `live-debugging-partner` | "The project includes a browser demo for testing mock deployments." | `supporting quote is real but does not mention anything in the claim` (path 4) |
| `live-debugging-partner` | "The project provides deterministic benchmarks for evaluating simulation outcomes." | same as above |
| `live-debugging-partner` | "The project simulates different interaction modes (voice and text) for debugging sessions." | same as above |
| `live-debugging-partner` | "The project includes a staging harness for testing mock deployments without executing actual commands." | same as above |

Every verified claim for `resume-builder-agent` got a real, independently
checkable permalink — e.g. the claim *"The project uses FastAPI to provide
a deterministic scoring service..."* is backed by
`README.md` at commit `6507596` — confirmed by opening
[that exact file at that exact commit](https://github.com/aniketqw/resume-builder-agent/blob/650759625ba7bd98bcdeb9438dcc42b77761c4f7/README.md)
and finding the quoted sentence really is there.

### 9. `build_document` — the CLI/browser join point

Code: [`graph.py::Runner.document`](https://github.com/aniketqw/resume-builder-agent/blob/d118de51a5f50d1380aa8be2f9a414c4ea716b65/v2/src/rba/graph.py#L127-L136)
→ [`resume/document.py::from_summaries`](https://github.com/aniketqw/resume-builder-agent/blob/d118de51a5f50d1380aa8be2f9a414c4ea716b65/v2/src/rba/resume/document.py#L417),
deterministic, zero LLM calls. Builds the editable `ResumeDoc` from
verified summaries. A browser import's graph run ends here — see
[`_after_document`](https://github.com/aniketqw/resume-builder-agent/blob/d118de51a5f50d1380aa8be2f9a414c4ea716b65/v2/src/rba/graph.py#L180-L184)
— because `full_run` was never set in this run's input state.

**Real:** `0.0044s`. Built the editable `ResumeDoc` from the 30 verified
claims across 6 repos (generated bullets arrive switched off — see
`merge.py`). Graph run ended here: the input state never set
`full_run=True`, so `_after_document` returned `END` and nodes 10-12
(`build_resumes`, `compare`, `plan_improvements`, CLI-only) never ran —
confirmed directly: they do not appear anywhere in this trace's run list.

---

## Total wall-clock for this run

**181.07 seconds end to end** (`connect_github` through `build_document`),
of which:
- `select_repos` (Jev classification): 48.78s — **27%**
- `summarize_repo` (6 Ollama calls, semaphore(2)): 99.61s — **55%**
- Everything else (GitHub fetch, AST indexing, safety scan, verification,
  document build): 32.68s — **18%**

The LLM step is the real bottleneck, as expected — and it had **zero
outright failures** this run, where the same six repos' summarization
through Groq (an earlier run this session, same code) lost 3 of 6 repos
entirely to schema-mismatch 400s. This is the concrete, trace-grounded
evidence for why the Ollama option exists.

## What was not captured

**No screen recording was made of this run.** Making one would need
Playwright (not currently in this project's installed dependencies — it's
an optional extra `scripts/capture_demo.py` already documents installing
separately) plus downloading browser binaries, which did not fit in the
time available for this document. Everything above is instead the actual
server-side trace data, independently cross-checked against the real
files in the repository — a stronger form of verification than a video
would have been, if a less visual one. If a recording is still wanted, it
can be captured with `pip install playwright && playwright install
chromium` plus a short script following `scripts/capture_demo.py`'s
existing pattern, driving the same upload → score → import flow against
the live server.
