# v2 trace

Why-history of the coherence revamp, in the same spirit as `v1/TRACE.md`: every entry is
traceable to a source in the repository, or says so when it isn't. Each table ends with
either `> Rationale recorded in <source>.` or `> Reconstructed from the diff.`

**Revert anchor:** `git tag v2-pre-revamp` → commit `8f2f3877fb87ae42a072166fb2e38ebc351b6ae5`
(tree clean at tagging time). To fully revert: `git reset --hard v2-pre-revamp` (destructive —
confirm with the user first) or `git diff v2-pre-revamp -- v2/` to see everything this
revamp touched.

**Baseline before any edit:** `v2/.venv/bin/python -m pytest -q` from `v2/` —
34 test files, 456 test functions, 540 collected node ids (see run output captured at the
start of this work for exact pass/fail counts).

**Context:** the LangGraph orchestration in `v2/src/rba/graph.py` existed (12 nodes, SQLite
checkpointer, one `interrupt()`) but only the CLI ran it — the browser import bypassed it
entirely via a hand-sequenced subset in `resume/pipeline.py`. Summarization never saw source
code, only README + file paths + commit messages. This revamp makes the graph the one thing
both entry points run, gives the summarizer real (if lightweight) code structure via a
stdlib-`ast` adapter, deletes an orphaned feature cluster, and completes LangSmith tracing.

---

## Iteration 1 — scope and backup

| # | Change | File | Why | What it improves |
|---|---|---|---|---|
| 1 | Tagged `v2-pre-revamp` at `8f2f387` | (git) | Revert anchor before any edit, per explicit instruction to back up v2 first | Full revert available with one command |
| 2 | Added `.claude/skills/langgraph-docs/SKILL.md` | repo root | Spec requires verifying LangGraph APIs against docs/installed version rather than guessing; the old `langchain-ai.github.io` docs host now redirects | Caught the `stream_mode` multi-mode requirement (see iteration below) before it became a bug |
| 3 | Created this file | `v2/trace.md` | User asked for a revertible, reviewable change log in table format | Mirrors `v1/TRACE.md`'s provenance convention for v2 |

> Reconstructed from the diff.

---

## Iteration 2 — delete the dead cluster

| # | Change | File | Why | What it improves |
|---|---|---|---|---|
| 1 | Deleted | `nodes/variants.py`, `nodes/additions.py`, `resume/versions.py`, `resume/grounding.py`, `prompts/variant.v1.md`, `prompts/addition.v1.md`, `prompts/summarize.v1.md`, `tests/test_variants.py`, `tests/test_additions.py`, `tests/test_versions.py`, `tests/test_grounding.py` | Verified zero reverse imports from any entry point (`grep` across `src/`) — a closed orphan component never reachable from the CLI or the API | Coherence: removes a feature cluster that looked wired up (60+ green tests) but was dead on arrival |
| 2 | Removed `versions`/`variants`/`additions` fields; added `claims` field | `src/rba_api/sessions.py` | Those fields were never assigned non-`None` by anything; checked `web/js/` first — frontend never reads `snapshot()["ready"]["versions"]` | Session struct now only holds live fields; `claims` prepares Step 1b |
| 3 | Removed `tenacity`, `sse-starlette` | `pyproject.toml` | Zero imports anywhere — retries are hand-rolled in `groq_client.py`/`limiter.py`; SSE frames are hand-formatted over `StreamingResponse` | Declared deps now match what's actually used |
| 4 | Removed `tiers.fallback`, `routing.compare`, `routing.judge` | `config.yaml` | Verified via grep: no `model_for_tier("fallback")` call anywhere; no node ever calls `client.structured(..., node="compare")` or `node="judge"` — only `summarize_repo`/`build_resumes`/`plan_improvements` do | Config now only lists tiers/routes that are actually read |
| 5 | Moved `LANGSMITH_API_KEY` out of `REQUIRED_KEYS` into new `OPTIONAL_KEYS` | `config.py`, `cli.py` | `REQUIRED_KEYS` gated `rba doctor`'s blocking exit code — so a missing (optional, per README:74) LangSmith key could fail `doctor` outright | `doctor` now reports it as an informational, non-blocking row |

**Test delta:** 540 → 469 passed (−71, exactly the 4 deleted test files). `ruff check src tests`: all checks passed.

> Rationale recorded in `/Users/aniketsaxena/.claude/plans/resume-builder-v2-staged-otter.md`, Step 1.

---

## Iteration 3 — stop discarding evidence

| # | Change | File | Why | What it improves |
|---|---|---|---|---|
| 1 | `Bullet` gains `evidence_path`, `evidence_sha`, and a `permalink` property | `resume/document.py` | `from_summaries` kept `claim.evidence[0].quote` and discarded `.path`; the commit SHA was captured (`fetch.py`) but used only as a cache key — verified directly before writing any code | A generated bullet can now prove itself: `github.com/<repo>/blob/<sha>/<path>`, not just a trust-me quote |
| 2 | `from_summaries(..., facts=None)` fills both fields from `RepoFacts.head_sha` | `resume/document.py` | `facts` already exists in graph state; nothing new computed | File-level permalinks, ahead of Step 2's line-level upgrade |
| 3 | `merge_candidates` passes `evidence_path`/`evidence_sha` through | `resume/merge.py` | It rebuilt bullets field-by-field and would have silently dropped both on the merge path (the one the browser actually uses) | Permalinks survive the GitHub-import merge, not just a unit test |
| 4 | `Session` gets a `claims` field; `routes/github.py` sets `session.claims = result.claims` | `rba_api/sessions.py`, `rba_api/routes/github.py` | `ImportResult.claims` was computed and then never stored on the session | Claims (with evidence) are now reachable from the API, not thrown away two lines before the response |
| 5 | Two new tests: permalink built + round-trips through `to_dict`/`from_dict`; empty for a `commit:` path or a "yours" bullet | `tests/test_resume.py` | Must not fabricate a link where there is nothing to click | Guards both the happy path and the two cases that must stay empty |

**Test delta:** 469 → 471 passed (+2 new). `ruff check src tests`: all checks passed.

> Rationale recorded in `/Users/aniketsaxena/.claude/plans/resume-builder-v2-staged-otter.md`, Step 1b.

---

## Iteration 4 — repo brain: real source, a call graph, located evidence

| # | Change | File | Why | What it improves |
|---|---|---|---|---|
| 1 | New `rba/repobrain.py`: stdlib-`ast` adapter (`index`, `RepoBrain.symbol/callers/callees/call_chain`, `symbol_at_line`) | `repobrain.py` (new) | CodeGraphContext (Neo4j+tree-sitter) was ruled out earlier — protobuf conflict with onnxruntime, no Docker, no Python API. Spec still requires the adapter *shape*; implemented it with zero new deps | A swappable boundary: CGC could implement this same interface later without touching the summarizer or verifier |
| 2 | Two correctness bugs caught and fixed by my own smoke test before committing: a route decorator (`@router.post(...)`) was showing up as a callee of the function it decorates; a nested function's calls were double-counted under its enclosing function | `repobrain.py` | Verified both empirically, not assumed | `callees()`/`call_chain()` give real, non-duplicated edges |
| 3 | `RepoFacts.source_files`; `fetch_repo_facts` fetches up to 12 ranked `.py` files per repo | `state.py`, `nodes/fetch.py` | The summarizer had never seen a line of code — only README + filenames + commit messages + dependency names | Real source finally reaches the pipeline, bounded and ranked (non-test files preferred) |
| 4 | **Bug found and fixed**: README/key-file fetches used `repo.default_branch`, not the pinned `head_sha` | `nodes/fetch.py` | A push between "list the tree" and "read the file" could silently desync content from the commit a permalink claims it came from | Every file read in a run is now pinned to the exact commit that run saw — verified with a ref-recording stub backend |
| 5 | New `nodes/prepare_evidence.py`: builds a `RepoBrain` per repo from `source_files`, zero LLM calls, surfaces indexing gaps | `nodes/prepare_evidence.py` (new) | Deterministic structure-building step the plan calls for between `fetch_facts` and `safety_scan` (wired into the graph in Step 3) | Partial-source gaps are reported, never silently dropped |
| 6 | `Evidence` gains `repo`, `sha`, `line_start`, `line_end`, `permalink` — filled in only by `verify_claims`, never by the model | `state.py` | The model should never self-attest its own citation's location | A verified claim carries a real, clickable `blob/<sha>/<path>#L..` link |
| 7 | `build_evidence` adds a compact `code` block (defs+lines, call edges, routes, 2 short excerpts) from the brain; dropped before `files` under budget pressure; prompt bumped to version 3 (cache invalidates) | `nodes/summarize.py`, `prompts/summarize.v2.md` | Gives the model something real to cite instead of a bare filename | Root-caused fix for the documented tcell/go-json false-citation failure |
| 8 | `verify_claim` validates `evidence.path` against `facts.tree`/`readme_path`/`source_files` before anything else, and locates each surviving quote's real line range via the brain | `nodes/verify.py` | Closes the "path is model-authored and never validated" hole named in the original spec | A fabricated path is now rejected outright, with a line-numbered citation for everything that survives |
| 9 | `Bullet` gains `evidence_line_start`/`evidence_line_end`; permalink includes `#L..` fragment; threaded through `from_summaries` and `merge_candidates` | `resume/document.py`, `resume/merge.py` | Completes the Step 1b chain now that a real line range exists | A browser-rendered bullet links straight to the exact lines that support it |

**New test files:** `test_repobrain.py` (8), `test_prepare_evidence.py` (1), `test_summarize.py` (5, first-ever coverage for `build_evidence`/`summarize_one` — none existed before). Extended `test_fetch.py` (+2), `test_verify.py` (+4), `test_resume.py` (updated 1 existing test for the new line-range fields).

**Test delta:** 471 → 491 passed (+20). `ruff check src tests`: all checks passed.

> Rationale recorded in `/Users/aniketsaxena/.claude/plans/resume-builder-v2-staged-otter.md`, Step 2.

---

## Iteration 5 — one graph, one branch

| # | Change | File | Why | What it improves |
|---|---|---|---|---|
| 1 | Deleted `make_checkpointer`, `CHECKPOINT_PATH`, `Runner.human_review`, the `review` kwarg, `AsyncSqliteSaver`/`interrupt` imports; `build_graph(runner)` takes one argument | `graph.py` | Both judged "high level, not required right now" | Smaller graph, no msgpack-serialisability constraint on state |
| 2 | Added `prepare_evidence` node (wraps Step 2's `nodes/prepare_evidence.py`) and `build_document` node (wraps `from_summaries`, zero LLM calls) | `graph.py` | The CLI/browser join point the plan calls for | Both entry points now run through the same evidence-prep and document-build steps |
| 3 | One conditional edge: `build_document -> build_resumes` iff `state["full_run"]`, else `END` | `graph.py` | Keyed off **state**, not a `Runner` attribute, so the graph is readable standalone | The graph is still 13 named nodes (12 shared + `report` on the CLI branch only), no second hand-sequenced workflow |
| 4 | `Runner.__init__` gains `github_token`/`groq_key`/`jev_key`/`limiter`/`cache`/`max_repos`, all defaulting empty so CLI behaviour is unchanged; `select()` now constructs its own `JevClient` (was previously only wired in `pipeline.py`, never in the graph) | `graph.py` | Lets one `Runner` serve both the CLI (no per-request creds) and the API (per-request creds), without branching | Prepares Step 5/6: the API's `limiter`/`cache` can finally reach the graph path |
| 5 | `AgentState` gains `full_run`, `resume_text`, `document` | `state.py` | Forgotten in the first pass at this file (Step 2) — caught immediately because the new tests failed with `KeyError: 'document'` | State is complete for the branch to actually carry a result |
| 6 | `compare`'s docstring/comment now states explicitly it is role-fit advice from the hand-written rubric, **not the ATS score** | `graph.py` | Honesty fix from the plan's Step 9, done opportunistically while already in this function | No ambiguity between the graph's rubric table and the real `/api/score` scorer |
| 7 | `test_graph.py`: deleted the 2 human-review tests; `test_full_graph_runs_offline` now passes `full_run=True` and asserts `result["document"]` too; added `test_browser_import_stops_at_the_document` | `tests/test_graph.py` | The test that actually proves CLI and browser share one graph | Same fixtures, same graph, `full_run` is the only difference between the two tests |

**Test delta:** 491 → 490 (−2 human-review, +1 browser-branch). `ruff check src tests`: all checks passed.

> Rationale recorded in `/Users/aniketsaxena/.claude/plans/resume-builder-v2-staged-otter.md`, Step 3.

---

## Iteration 6 — delete `rba resume`

| # | Change | File | Why | What it improves |
|---|---|---|---|---|
| 1 | Deleted `resume_run` command, `--review`/`--thread` options on `run`, the `__interrupt__` branch, `make_checkpointer` import | `cli.py` | Nothing left for it to resume — the interrupt and checkpointer are gone | `rba run --help` and `rba resume --help` both verified by hand: options gone, command gone |
| 2 | `run` now calls `build_graph(runner).ainvoke({"username": u, "full_run": True}, {"recursion_limit": 50})` | `cli.py` | Matches the new 1-arg `build_graph` signature and the `full_run` branch | CLI behaviour otherwise unchanged — verified `rba config` still renders the three real routing entries (`compare`/`judge` correctly absent, from Step 1) |

**Test delta:** none (no test referenced the removed command). `ruff check src tests`: all checks passed.

> Rationale recorded in `/Users/aniketsaxena/.claude/plans/resume-builder-v2-staged-otter.md`, Step 4.

---

## Iteration 7 — `build_from_github` becomes a graph driver

| # | Change | File | Why | What it improves |
|---|---|---|---|---|
| 1 | **Bug found and fixed**: `Runner.list_repos` never filtered private repos — only the old hand-sequenced `pipeline.py` did. Unifying onto the graph would have silently reintroduced private-repo leakage | `graph.py`, `state.py` (`repos_found`) | Caught before it shipped by comparing old vs. new behaviour line-by-line, not assumed | `list_repos` now filters to public, matching the original spec's own node description |
| 2 | `build_from_github` rewritten: builds a `Runner` + `build_graph(runner)`, drives it with `graph.astream({...}, stream_mode=["updates","values"])`, maps node names to `NODE_PROGRESS` entries for the existing `job.report` callback | `resume/pipeline.py` | The hand-sequenced 5-node subset is gone; the browser now runs the identical graph, just without `full_run` | One workflow. The 1,433 s full-run figure no longer applies to the browser path at all, by construction — the seven CLI-only Groq calls never run |
| 3 | Kept `build_from_github`'s name, signature (+`resume_text`/`limiter`/`cache`) and `ImportResult` contract; kept the two literal `stats["errors"] = ...` / `stats["summarised"] = ...` lines | `resume/pipeline.py` | `test_github_import.py` monkeypatches this name at 3 sites; `test_merge.py:184` does `inspect.getsource()` asserting those exact substrings | Zero test edits needed for either file — both verified green |
| 4 | **Bug found and fixed**: a real circular import. `pipeline.py` importing `graph.py` at module level crashed in a fresh process when `rba.graph` was the first thing touched (`graph.py` → `resume.document` → `resume/__init__.py` → `pipeline.py` → `graph.py`, mid-init). The full test suite had been passing anyway, by accident of pytest's collection order masking it | `resume/pipeline.py` | Reproduced directly with `python -c "from rba.graph import Runner"` before and after the fix | The exact failure a bare `rba run <user>` would hit in production is gone — deferred the import, matching `cli.py`'s own existing pattern for the same reason |
| 5 | Added `test_node_progress_covers_every_browser_branch_node` | `tests/test_graph.py` | Guards the one real regression risk: a node renamed in `graph.py` silently losing its SSE progress frame | `set(NODE_PROGRESS) == graph nodes - {__start__,__end__} - CLI-only` |

**Test delta:** 490 → 491 (+1). `ruff check src tests`: all checks passed. Verified by hand in a fresh process: `from rba.graph import Runner` first, `from rba.resume.pipeline import build_from_github` first, and `import rba.cli` — all three clean.

> Rationale recorded in `/Users/aniketsaxena/.claude/plans/resume-builder-v2-staged-otter.md`, Step 5.

---

## Iteration 8 — the dead limiter goes live

| # | Change | File | Why | What it improves |
|---|---|---|---|---|
| 1 | `_run_import` passes `limiter=app_state.limiter_for(body.groq_key)`, `cache=app_state.cache`, `resume_text=session.doc.as_text() if session.doc else ""` | `rba_api/routes/github.py` | `AppState.limiter_for` was defined and documented at length but called by nobody — every import got a private `Limiter` believing it owned the whole 8,000 TPM | Closes the exact failure `deps.py`'s own module docstring describes |
| 2 | New test: two imports with the same Groq key share one `Limiter` object and one `Cache`; resume text reaches the pipeline | `tests/test_github_import.py` | This is one of the acceptance cases named in the original spec ("imports sharing an effective Groq key share the actual limiter") | Verified directly, not by inspection |

**Test delta:** 491 → 492 (+1). `ruff check src tests`: all checks passed.

> Rationale recorded in `/Users/aniketsaxena/.claude/plans/resume-builder-v2-staged-otter.md`, Step 6.

---

## Iteration 9 — resume-aware repo selection

| # | Change | File | Why | What it improves |
|---|---|---|---|---|
| 1 | New `score_relevance(repos, signals, resume_text)`: one batched `dense.embed([resume_text, *profiles])` call, cosine similarity clamped at 0 | `nodes/select.py` | The requested capability — "understand from the resume which repos are relevant" — using the same free, deterministic, local embedding the ATS scorer already uses. No LLM, no new dependency | One model pass for the whole candidate set, not one per repo |
| 2 | `RepoRef.relevance: float = 0.0` | `state.py` | Carries the score without touching the admission logic | `embed` returning `None` (model unavailable) leaves `relevance` at 0 — degrades to exactly today's ranking, verified directly with a monkeypatch |
| 3 | Sort key becomes `0.75*claimable + 0.25*relevance`; **the admission threshold still tests `claimable` alone** | `nodes/select.py` | Relevance may re-order who makes the cut, but must never promote something Jev judged unclaimable | Verified directly: a repo with a perfect relevance match but `claimable=0.01` still gets rejected |
| 4 | `rel={relevance:.2f}` appended to `selection_reason` | `nodes/select.py` | The UI's existing "why was this repo excluded" string carries it for free | No new API surface needed |
| 5 | 4 new tests: relevant repo outranks unrelated at equal claimable; empty resume text is a no-op; unavailable model degrades safely; claimable floor cannot be bought with relevance | `tests/test_select.py` | Covers the real embedding path (this environment has it available) and the degraded path | All 20 `test_select.py` tests pass, including the 16 pre-existing ones unchanged |

**Test delta:** 492 → 496 (+4). `ruff check src tests`: all checks passed.

> Rationale recorded in `/Users/aniketsaxena/.claude/plans/resume-builder-v2-staged-otter.md`, Step 7.

---

## Iteration 10 — complete LangSmith tracing

| # | Change | File | Why | What it improves |
|---|---|---|---|---|
| 1 | Verified (via `inspect.signature` on the installed `langsmith 0.14.17`) that LangGraph already auto-traces every node natively when `LANGCHAIN_TRACING_V2`/`LANGCHAIN_API_KEY` are set — confirmed in `langchain_core/callbacks/manager.py`'s `_configure()` and LangGraph's own `_runnable.py` (`from langsmith.run_helpers import _set_tracing_context`) | (verification only) | The spec's own skill (`langgraph-docs`) requires checking installed APIs, not recalling them | Nothing to build for node-level tracing — it already exists once env vars are set early enough |
| 2 | `enable_tracing()` called in `Runner.__init__`, not only lazily inside `GroqClient.__init__` | `graph.py` | Previously tracing only turned on once a `GroqClient` was constructed inside the `connect` node — so `connect_github` itself ran before tracing was live | Every node, including the first one, is now covered from the start of a run |
| 3 | `_trace` rewritten: uses `langsmith.run_helpers.get_current_run_tree()` to find the active graph-node run and attach a **child** via `parent.create_child(...).post()/.end()/.patch()`; falls back to a flat root run only when no parent context exists (e.g. a unit test calling `GroqClient` directly) | `groq_client.py` | The old `create_run` had no `parent_run_id`, so every LLM call rendered as a disconnected root next to the trace tree LangGraph was already building | One trace tree per import: all nodes + nested LLM calls, not a flat list |
| 4 | `start_time`/`end_time` now passed to every trace | `groq_client.py` | The old call passed neither, so latency was never visible | Real per-call latency in LangSmith |
| 5 | Cache hits are now traced too, as zero-token runs, at the early-return point in `complete()` | `groq_client.py` | A cached completion previously produced no run at all, under-reporting usage | Accounting stays honest for a warm run |
| 6 | 4 new tests with `StubRunTree`/`StubTracerClient`: nests under an active parent; falls back to root with no parent; cache hit traced at zero tokens; a broken tracer still lets the call succeed | `tests/test_groq_client.py` | `conftest.py` disables tracing for the whole suite, so these are the only tests that actually exercise `_trace`'s new logic (via direct `_tracer` stubbing, bypassing `enable_tracing()`) | Verified the nesting/fallback/cache-hit/failure-isolation behaviour directly, not by inspection |

**Test delta:** 496 → 500 (+4). `ruff check src tests`: all checks passed (one auto-fixed style issue, `datetime.timezone.utc` → `datetime.UTC`).

**Not yet done:** an actual live trace against a real LangSmith project — deferred to final verification, since it needs `GROQ_API_KEY`/`LANGSMITH_API_KEY` and a real import run. `conftest.py`'s tracing-disable (0.8s → 5.8s measurement) is untouched and still load-bearing.

> Rationale recorded in `/Users/aniketsaxena/.claude/plans/resume-builder-v2-staged-otter.md`, Step 8.

---

## Iteration 11 — honesty relabel: rubric advice, not the ATS score

| # | Change | File | Why | What it improves |
|---|---|---|---|---|
| 1 | Report's role table: heading → "Role-fit advice (not an ATS score)", column "ATS score" → "Role-fit points", added a one-sentence caveat pointing at the real scorer (`/api/score`) | `report.py` | `routes/meta.py:43-49` already disowns this rubric as "advice... never the score" — the generated report said the opposite | A reader of `out/report.md` sees the same honesty the API already states |
| 2 | "+N ATS points"/"+N ATS" in the plan section → "+N role-fit points" | `report.py` | Same conflation, in the improvement-plan rendering | Consistent wording throughout one document |
| 3 | CLI summary table: title "ATS scores" → "role-fit advice (not an ATS score)", column "score" → "points" | `cli.py` | Same fix, visible on every `rba run` | The CLI's own terminal output no longer contradicts the API |
| 4 | Updated/added tests: heading string, "not an ATS score" caveat present, new column header | `tests/test_report.py` | — | Pins the wording so it can't silently drift back |
| 5 | (Already done in Iteration 2) `tiers.fallback: qwen/qwen3.8-27b` deleted from `config.yaml` — unreachable, and `json_object`-only, so falling back to it on a strict-schema node would trade a slow success for a parse failure | — | — | No action needed this iteration |

**Scope note:** did not rename the internal `PlanStep.ats_gain` Python attribute or `rba/scoring.py`'s prose — those are internal/non-user-facing and renaming them is materially bigger surgery than the "2-line honesty fix" this step calls for. Only the strings a human actually reads were changed.

**Test delta:** 500 → 501 (+1). `ruff check src tests`: all checks passed.

> Rationale recorded in `/Users/aniketsaxena/.claude/plans/resume-builder-v2-staged-otter.md`, Step 9.

---

## Iteration 12 — live verification, and one real bug it found

Ran a real GitHub import against `aniketqw` (max_repos=2), with real Groq/GitHub/
TypeSafe/LangSmith credentials from `.env`. Confirmed with the user's explicit
go-ahead first, since it spends real API quota and posts to a real LangSmith project.

| # | Finding / change | File | Why | What it improves |
|---|---|---|---|---|
| 1 | Live run completed in 68.7s (not the documented 1,433s) — `repos_found: 103, public: 60, selected: 2, groq_tokens: 5431, claims_total: 6 (3 verified, 2 needs_measure, 1 rejected)` | (live run) | Confirms the browser-branch stop at `build_document` is what fixes the latency, by construction, not by assumption | Real numbers, not projected ones |
| 2 | **Real bug found and fixed**: a `needs_measure` (unverified) claim produced a permalink like `github.com/pages-hub/blob/<sha>/files` — "files" was the evidence bundle's own JSON key, not a real path. `verify_claim` only resolves path/sha/line-range for claims that reach VERIFIED; `from_summaries` was building a permalink from unvalidated, model-authored evidence for any non-rejected claim, regardless of status | `resume/document.py` | Exactly the kind of invented-looking-legitimate citation this whole project exists to prevent — and it was caught by running a real import, not by any unit test | Only a VERIFIED claim's path/sha/line-range is ever trusted; a `needs_measure` bullet now shows its quote with no permalink, honestly |
| 3 | New regression test reproducing the exact live failure (`path="files"`, status=NEEDS_MEASURE) | `tests/test_resume.py` | — | Pins the fix so it can't silently regress |
| 4 | LangSmith trace verified directly via `Client.list_runs()`: one `trace_id` spanning 60 runs, with real nesting — graph root → `fetch_facts` → nested `list_commits`/`get_file_contents` MCP tool calls (traced automatically, a bonus I didn't have to build), and → `summarize_repo` (chain node) → the two `summarize_repo` (llm) calls as children | (live verification) | Confirms Step 8's design against a real backend, not mocks | "One trace tree per import" is now a verified fact, not a claim |

**Test delta:** 501 → 502 (+1, the permalink regression test). `ruff check src tests`: all checks passed.

---

## Iteration 13 — chunked dense scoring (fixes the 512-token truncation), with a determinism fix

The dense ATS channel embeds with `bge-small-en-v1.5`, a 512-token-ceiling
model. `aniket_bits.txt` runs ~1,150 tokens as plain text — roughly 56% of
every resume was silently invisible to the dense channel, truncated before
embedding. Fix: chunk the document along section/bullet boundaries, embed
each chunk, and mean-pool (length-weighted, re-normalized to unit length)
into one vector — established practice for long-document sentence-embedding
matching (sentence-BERT chunk-averaging). The alternative, Jina AI's "late
chunking" (2024), needs a long-context encoder to embed the whole document
*before* splitting, which `bge-small` cannot do without reopening the
memory-footprint tradeoff already settled in `docs/literature-survey.md`.
Full citation-grade literature grounding for both is deferred to
`docs/literature.tex` (Part D) — this entry states the practice plainly, not
as a fully-cited trail yet.

| # | Finding / change | File | Why | What it improves |
|---|---|---|---|---|
| 1 | New `document_chunks(doc)`: splits a resume into section/bullet-bounded chunks with content-derived, toggle-stable chunk IDs | `resume/chunking.py` | Chunk boundaries must track resume structure (sections/bullets), not an arbitrary token window, or a chunk can split a bullet mid-sentence | Every chunk is a coherent unit; IDs stay stable when an unrelated bullet is toggled, so the cache keeps working |
| 2 | **Determinism bug found and fixed**: the same chunk embedded in one batch call alongside *different* batch-mates produced a reproducible ~4.3e-08 vector-level difference (not randomness — identical across 10 trials; a floating-point batch-padding artifact of `encode_batch`'s dynamic padding-to-longest-in-batch). This broke exact (`==`) equality between a document scored alone (`rescore()`) and the same document scored as the base of a leave-one-out sweep (`attribute()`) | `ats/batch.py` (`_dense_values`) | The UI's whole premise is that "the document's current score" is one reproducible number no matter how many variants are swept alongside it in a batch request — floating-point batch noise would make that false | `_dense_values`'s chunked branch now embeds the **first** chunked candidate (always "the document as it stands") via its own dedicated `_ensure_chunks_embedded` call, then embeds the rest together in a second call, merging vector maps before pooling. The base candidate is now bit-exact-reproducible regardless of batch composition |
| 3 | Rejected alternative fix: fixed-length padding instead of dynamic batch padding, to eliminate the ~4.3e-08 diff at the source | (design only, not shipped) | Measured directly: 76.9x slower (9ms → 701ms for two short chunks) | Confirms the isolated-first-call fix (row 2) is the right tradeoff — not a shortcut |
| 4 | Regression test proving row 2's fix, verified by *temporarily reverting it*: confirmed the test fails with the exact predicted ~4e-8 diff, then restored the fix and confirmed it passes | `tests/test_chunking.py::test_base_candidate_is_bit_exact_regardless_of_batch_mates` | A fix that merely "looks right" isn't enough — proving the test actually catches the bug it claims to catch is the project's stated verification discipline | This regression cannot silently come back |
| 5 | `chunks=` wired into every "current document" candidate: `rescore()`/`attribute()`'s full-document and leave-one-out candidates, and `/api/score`'s and `/api/score/batch`'s base + variant candidates | `resume/attribution.py`, `rba_api/routes/score.py` | Chunking only fixes truncation where it's actually used — every scoring path had to opt in, not just the engine-level API | All four scoring entry points use the same chunked-and-pooled vectors; no path still single-embeds a truncated blob |
| 6 | Two more cross-path tests broke from row 5's wiring and were fixed by aligning both sides to use chunking consistently, loosening `==` to `pytest.approx(..., abs=1e-6)` with the row-2 floor as the documented reason: `test_batch.py::test_dedup_matches_the_naive_path_with_dense_on` (was comparing un-chunked candidates against now-always-chunked `rescore()` — apples to oranges) and `test_api.py::test_attribution_matches_the_engine` (warm session cache vs a fresh no-cache direct call) | `tests/test_batch.py`, `tests/test_api.py` | Both comparisons are fundamentally cross-request / cache-state-dependent, which A1's fix (row 2) does not and architecturally cannot make bit-exact without the row-3 cost | `test_score_equals_rescore_exactly` and `test_base_candidate_is_bit_exact_regardless_of_batch_mates` were explicitly left untouched at `==` — they test the property that genuinely holds exactly |
| 7 | `tests/test_live.py::test_one_toggle_stays_inside_the_interaction_budget` failed at 818ms (budget 450ms) — root cause: the test never passed a `cache=`, so chunking's caching benefit (now load-bearing for speed) went unused | `tests/test_live.py` | Chunking trades a slower cold start for the ability to see the whole document; without a warm cache every re-score re-embeds every chunk from scratch | Fixed by sharing one `DenseCache()` across warm-up and measured calls, matching how `rba_api/sessions.py::Session.dense_cache` actually works in production |
| 8 | New, separate, honestly-labelled test for the real cold-start cost rather than hiding it: `test_the_first_ever_score_in_a_session_is_not_held_to_the_warm_budget` — measured ~753ms for a brand-new session's first-ever score, asserted only against a loose 3000ms sanity ceiling | `tests/test_live.py` | A session's first score genuinely is slower now (chunking's accepted cost); hiding that inside a passing 450ms test would be dishonest, not a fix | A real regression here stays visible (printed), not silently masked |
| 9 | Live HTTP-layer verification against the actual running dev server (not the unit-test micro-benchmark): uploaded `fixtures/aniket_bits.txt`, set a real JD, one warm-up `POST /api/score`, then measured `POST /api/score` (raw `0.6624445573832407`, **2.2ms**) and `POST /api/score/batch` with 5 variants (base raw `0.6624445573832407`, **2.7ms**) | (live run via `httpx`) | Confirms the fix and the caching story hold through the real HTTP layer, not just inside pytest | Flicker check: the two base/headline raw floats are bit-identical through the live API — both numbers sit far under the 450ms unit-test budget |

**Test delta:** full suite **518 passed, 0 failed** (was 502 at the end of Iteration 12; net +16 across `test_chunking.py` (new file) and the fixes above). `ruff check src tests`: all checks passed.

---

## Iteration 14 — Evidence & Call-Graph tab

A new tab makes the claim→evidence chain and the per-repo code structure
directly inspectable, instead of only showing up as the finished resume
lines. The data already existed in graph state (`prepare_evidence.py`
produces a `RepoBrain` per repo) — it was being computed and then dropped on
the floor before this iteration, same shape of bug as Iteration 8's
`facts`/`summaries`/`claims` fix.

| # | Finding / change | File | Why | What it improves |
|---|---|---|---|---|
| 1 | `ImportResult` gained `repo_graph: dict`; `build_from_github` now captures `final.get("repo_graph")` instead of dropping it | `resume/pipeline.py` | The graph state already had it (`prepare_evidence` node); nothing downstream could reach it | Code-structure data survives past the graph run |
| 2 | `to_dict()` added to every `repobrain.py` dataclass (`Symbol`, `CallEdge`, `RouteBinding`, `FileIndex`, `RepoBrain`), explicit dict literals matching `document.py`'s convention | `repobrain.py` | An API response needs plain JSON, and `dataclasses.asdict()` is not this project's convention | `RepoBrain.to_dict()` is the entire payload for the call-graph panel |
| 3 | `Session.repo_graph` added; `routes/github.py` stores it alongside `session.claims` on import completion; new `GET /api/session/{id}/repo-graph` returns, per repo, **every** claim (verified/needs_measure/rejected — not only the ones that reached the resume) with full evidence, plus the repo's `RepoBrain.to_dict()` | `rba_api/sessions.py`, `rba_api/routes/github.py` | The whole point of this tab is explaining *why* a claim did or didn't survive verification — hiding the rejected/needs_measure ones would defeat it | A `needs_measure` claim now shows its `measure_hint`; a `rejected` one shows its `reason`; nothing is silently hidden |
| 4 | Response keys are the union of `repo_graph`'s repos and claims' repos, not just `repo_graph`'s | `routes/github.py` | A repo can have claims without a successful code index (non-Python source, or an `ast` parse failure) — keying off `repo_graph` alone would silently drop its claims from the response | Every claim is reachable from this endpoint, with `"graph": null` honestly marking the repos that have no index |
| 5 | New `web/js/tabs/evidence.js`: repo picker, a claim→evidence chain list (status chip, quote, clickable GitHub permalink or — for an unverified claim — its `measure_hint`/`reason` instead of a link), and a call-graph panel (`<details>` per file, symbols with line ranges and their unique callees, plus a `partial_source` gaps list) | `web/js/tabs/evidence.js`, `render.js`, `store.js`, `api.js`, `actions.js`, `events.js`, `css/panels.css` | Deliberately a plain list, not a force-directed graph diagram — `repobrain.py`'s own scope is bare-name call resolution, Python-only, with no import resolution (see its module docstring); a graph layout would visually overclaim precision this data doesn't have | A GitHub import's evidence and code structure become directly clickable, reusing the existing `--good`/`--warn`/`--bad` CSS tokens for status colour so it doesn't invent a new visual language |
| 6 | Three new tests: `test_to_dict_is_json_serialisable_and_carries_the_real_structure` (`test_repobrain.py`); `test_repo_graph_reports_every_claim_status_and_the_code_index`, `test_repo_graph_is_empty_for_a_session_that_never_imported`, `test_repo_graph_on_an_unknown_session_is_404` (`test_github_import.py`) | `tests/test_repobrain.py`, `tests/test_github_import.py` | Proves the endpoint's shape, the all-statuses guarantee, and the 404/empty edge cases, not just that it returns *something* | `test_github_import.py` now 20/20, `test_repobrain.py` now 9/9 |
| 7 | Live-server verification: `node --check` on every touched JS file; live `GET /js/tabs/evidence.js`, `/css/panels.css`, `/js/render.js` all 200; live `GET /api/session/{id}/repo-graph` against a fresh session returns `{"repos": {}}` (200), against an unknown session returns 404 | (live run) | No bundler in this project, so a JS syntax error would otherwise only surface in a browser console | Confirms the new code is reachable and syntactically valid through the real HTTP layer, not just inside pytest |

**Not done, honestly recorded:** clicking through the tab against a *real*
GitHub import in an actual browser. That spends real Groq/GitHub API quota
and needs the user's explicit go-ahead (as the live import in Iteration 12
did); this session had no browser-automation tool to drive it headlessly
either way. The fake-pipeline tests in `test_github_import.py` exercise the
identical endpoint shape a real import produces, so the endpoint and UI
logic are proven; only the "a human watched the permalinks resolve in a
browser" step is outstanding.

**Test delta:** full suite **522 passed, 0 failed** (was 518 at the end of
Iteration 13; +4 from this iteration). `ruff check src tests`: all checks
passed.

---

## Iteration 15 — GitHub-import reliability, found via a real live run

A real import against the user's own GitHub account surfaced four real bugs
the test suite hadn't caught. Full reasoning, root-cause citations and
verification detail for each live in `v2/PLAN_PIPELINE_FIXES.md` (kept as a
standing, continuously-updated handoff document, not folded entirely into
this entry) -- this is the short version.

| # | Finding / change | File | Why | What it improves |
|---|---|---|---|---|
| 1 | Groq's strict-schema 400s (`json_validate_failed`) were retried by blindly doubling `max_tokens` regardless of cause -- correct for a genuine truncation, useless for a real shape violation (a claim missing its required `text` field). New `SchemaRejectedError(GroqError)` distinguishes the two; `structured()` now retries a shape violation with a corrective message showing the model its own bad generation, the same repair already used for a client-side `ValidationError` | `groq_client.py` | Measured live: 3 real repos failed outright with this exact 400 before the fix | Confirmed live post-fix: the retry now genuinely runs (log shows `"schema validation failed after retry"` only after a second distinct mistake, not the old blind retries) -- a real improvement, not a full fix for every repo every time |
| 2 | `fetch_repo_facts` truncated every source file to 4000 chars *before* `repobrain.index()`'s `ast.parse()`, guaranteeing a `SyntaxError` on any file over that length (landing mid-token) -- also silently made any verified claim's quote past that cutoff unverifiable | `nodes/fetch.py` | Reproduced live: the user's own repo's source files filled `partial_source` with spurious syntax errors that had nothing to do with code quality | Cap raised to 60,000 chars (a safety ceiling, not an expected truncation point); `verify_claims`' quote-matching gets full files too |
| 3 | `read_pdf()`'s flat `pypdf` text extraction destroys a PDF table's column structure, and the text-only `_parse_table` heuristic cannot always recover it -- confirmed live: the user's "ACADEMIC DETAILS" table rendered with headers and values in the wrong cells entirely | `resume/ingest.py` | New `pdfplumber`-based `read_pdf_tables()`/`_pick_table()`/`_table_from_grid()` detect the real table geometry instead of guessing from reordered text, with the old heuristic kept as a fallback | Verified against the user's actual uploaded PDF (not just synthetic fixtures): the table now recovers headers and rows correctly, matching what the live import had garbled |
| 4 | First pass of fix #3 only checked a detected grid's row 0 for the header row -- missed that pdfplumber merges a section's heading line into the same table as the header row below it when nothing visually separates them, which is **every real resume tested**, not an edge case | `resume/ingest.py` | Found by testing fix #3 against the user's real PDF, not assuming the synthetic tests were representative | `_pick_table` now searches a grid's first 3 rows for the real header; verified the regression test fails without this depth search and passes with it |
| 5 | Measured, not re-tuned: pulled real per-node timing for a live import via the LangSmith SDK directly (`list_runs(trace_id=...)` -- the web UI doesn't render through a simple fetch). Confirmed `fetch_facts`/`summarize_repos` concurrency already works (overlapping tool-call timestamps); the real cost was `select_repos` (68.9s, Jev classification) and Groq rate limiting, neither of which the existing concurrency caps control | (measurement only) | Deliberately did not raise `McpBackend`'s or `summarize_repos`'s concurrency caps without a second measurement to justify it -- an unverified tweak is exactly what this project's house style forbids | Confirms the caps are not the bottleneck today; left as-is rather than guessed at |

**Not yet done, tracked in `PLAN_PIPELINE_FIXES.md`**: Ollama as a
configurable alternative to Groq (model chosen and smoke-tested --
`qwen2.5:14b-instruct` on a shared lab GPU host -- client code not yet
written), a requested refactor pass, and re-confirming the frontend
preview/export against the corrected table end to end.

**Test delta:** full suite **532 passed, 0 failed** (was 522 at the end of
Iteration 14; +10 across the new regression tests for findings #1-#4).
`ruff check src tests`: all checks passed. `pdfplumber>=0.11.4` added as a
new dependency (`pyproject.toml`); `uv sync --extra dev` required (not
plain `uv sync`) to keep `pytest`/`ruff` in the venv, since they live under
the `dev` optional-dependency group, not the main dependency set.

---

## Iteration 16 — Ollama as a configurable Groq alternative, and two more real parsing bugs

| # | Change | File | Why | What it improves |
|---|---|---|---|---|
| 1 | New `OllamaClient` matching `GroqClient.structured()`'s exact shape (`(messages, schema, ...) -> (T, Usage)`); `graph.py`'s `Runner.connect()` branches on `LLM_PROVIDER` (env var) to pick which client to build | `ollama_client.py`, `graph.py` | Removes Groq's 8,000 TPM free-tier rate limit as a factor entirely for anyone who opts in -- no changes needed in `nodes/summarize.py`/`build.py`/`plan.py`, confirmed by grepping every `client.structured()` call site first | Verified both with a 6-test fake-transport suite and a real call against a live remote Ollama host -- clean, correctly-shaped JSON on the first attempt |
| 2 | Found and fixed live: `_pick_table` only checked a detected grid's row 0, but pdfplumber merges a section's heading line into the same table as its header row when nothing visually separates them -- the first version of Fix 3 (Iteration 15) worked on every synthetic test and failed on the one real file tested | `resume/ingest.py` | Confirmed against the user's actual uploaded PDF, not synthetic fixtures | `_pick_table` searches a grid's first 3 rows; verified the regression test fails without this depth search |
| 3 | Found and fixed live, re-checking the *whole* real document's render (not just the table that was already known to be broken): an entry's title and date range landed on two separate lines in this PDF's extraction, which `_parse_entries` had never seen before -- it silently dropped the title line and produced an entry with an **empty title** | `resume/ingest.py` | Affected every entry in the real resume (Internships, Projects, Position of Responsibility), not an isolated case | New `_merge_split_dates()` rejoins them before entry-splitting; verified by reverting and confirming the predicted empty-title failure |
| 4 | Found and fixed in the same pass: this PDF's bullet glyph extracted as `\x7f` with no terminal period on any bullet, so the "ends in a full stop" continuation rule merged three separate bullets into one | `resume/ingest.py` | A leading bullet marker is as hard a "new bullet" signal as a trailing period | `_BULLET_MARKERS` added to `_join_wrapped`'s `starts_new` check, marker stripped from stored text; verified by reverting |

**Test delta:** full suite **540 passed, 0 failed** (was 532 at the end of
Iteration 15; +8 across the new Ollama tests and the two new parsing
regressions). `ruff check src tests`: all checks passed.

---

## Summary

| Gate | Status |
|---|---|
| Tests | **522 passed, 0 failed, 0 skipped** (baseline was 540 before the revamp started; net change accounts for −71 deleted orphan tests, +53 new tests across all iterations including chunking and the Evidence tab) |
| Lint | `ruff check src tests`: all checks passed, throughout every iteration |
| One graph, two entry points | Verified: `grep -n build_graph src/rba/cli.py src/rba/resume/pipeline.py` hits both; `test_browser_import_stops_at_the_document` proves it |
| Demo path (resume+JD, resume+role) stays Groq-free | Verified functionally with a guarded `GroqClient.__init__` monkeypatch: zero construction attempts across both paths |
| Shared limiter | Verified directly: two imports with the same Groq key get the same `Limiter` object |
| SSE progress | Verified: ≥8 `progress` frames then `done`; `NODE_PROGRESS` keys match the graph's actual node set by a dedicated test |
| LangSmith tracing | Verified **live**: one `trace_id`, real parent/child nesting, confirmed via `Client.list_runs()` against the real project |
| Live GitHub import | Run for real, with the user's explicit go-ahead; found and fixed one real bug (see Iteration 12) |
| CodeGraphContext (real backend) | **Not verified** — ruled out: forces `protobuf==3.20.3`, breaking the `onnxruntime` the ATS dense scorer needs. The adapter boundary (`repobrain.py`) exists and is stdlib-only; CGC could implement the same interface later |
| Docker | **Not run** — daemon was not running in this environment at any point during this work |
| `.env` / credentials | Gitignored, untracked, never printed or logged; verified in `.gitignore` before any live call |
| Commits | **None made.** Working tree is a clean set of uncommitted changes on top of `v2-pre-revamp` (`8f2f387`) |

**Revert:** `git diff v2-pre-revamp -- v2/` shows everything this revamp touched; `git checkout v2-pre-revamp -- v2/` (or `git reset --hard v2-pre-revamp`, destructive) restores the pre-revamp state exactly.

<!-- Further iterations appended below as steps land. Keep entries short: file, why,
     what it improves. Do not restate the whole plan here — link back to
     /Users/aniketsaxena/.claude/plans/resume-builder-v2-staged-otter.md for the full design. -->
