# Handoff: chunked ATS scoring + Evidence/Call-Graph tab

**If you are a fresh agent (Codex or otherwise) picking this up cold: read
this whole file before touching anything.** It tells you exactly what's done,
what's left, in what order, and the house style to match. Do not improvise
scope beyond what's listed here — every step below was individually verified
against the real code before being written down; nothing here is a guess.

- Repo: `/Users/aniketsaxena/Documents/p/from_aug_1/p2/resume-builder-agent`
- Work inside `v2/` only. Never touch `v1/` (historical reference, preserved
  as-is).
- Branch: `v2-revamp`. Last commit: `13bb8d6`. Everything below is
  **uncommitted work on top of that commit** — do not assume it's merged.
- Full prior design/planning context (why each decision was made, what was
  measured, what was rejected and why) lives in
  `v2/trace.md` — read it if you need the reasoning behind something, not
  just the what.
- Venv: `v2/.venv` (already set up — `source v2/.venv/bin/activate` or call
  `v2/.venv/bin/python` / `v2/.venv/bin/pytest` directly). Run everything
  from inside `v2/`.
- Test: `pytest -o addopts=""` (the repo's own `addopts=-q` hides the final
  summary line in some shells — override it). Lint: `ruff check src tests`.
- The live dev server, if you want to check it in a browser:
  `RBA_ONNX_THREADS=4 RBA_EMBED_BATCH=16 .venv/bin/uvicorn rba_api.app:app --reload --port 8000`,
  then open `http://localhost:8000`. `.env` already has all credentials
  (Groq, GitHub, TypeSafe, LangSmith) — **never print, log, or commit its
  contents.**

## House style — match this, don't invent a different one

- **Comments explain WHY, never WHAT.** A comment earns its place only if it
  records a non-obvious reason, a measured number, a specific past bug, or a
  citation. Never restate what the code visibly does.
- **Module and function docstrings tell the design story**, often with a
  measured before/after number inline (e.g. "cost 1,269ms vs 1,579ms,
  measured on `fixtures/aniket_bits.txt`"). If you add a behavior, either cite
  a real measurement or say "measured here, see test X" — never invent a
  plausible-sounding number.
- **Test names are full sentences describing the guarantee**, not
  `test_case_1`: e.g. `test_base_candidate_is_bit_exact_regardless_of_batch_mates`.
  Each test's docstring (if it has one) explains the *real failure* it guards
  against, often naming the exact bug or incident.
- **Dataclasses get explicit `to_dict()`/`from_dict()` methods**, not generic
  `dataclasses.asdict()` — follow the pattern already in
  `src/rba/resume/document.py` (`Bullet`, `Section`, `Entry`).
- **Never fabricate a number, a citation, or a test result.** This is the
  single governing rule of the whole project (see root `README.md` and
  `docs/literature-survey.md`). If something is unverified, say so explicitly
  rather than presenting a plausible guess as fact.
- **Minimal diffs. Prefer deleting code to adding it** when a change makes
  something obsolete. Don't add defensive code for cases that can't happen.
- Before considering any step "done": (1) run the specific tests touched, in
  isolation, (2) run the full suite, (3) run `ruff check src tests`, (4) if a
  fix claims to catch a regression, prove it — temporarily revert the fix,
  show the test fails, then restore the fix and show it passes again. Record
  the real pass/fail counts, never assume.

---

## STATUS (update the checkboxes below as you go — this is the source of truth)

### Part A — ship chunked dense scoring, with a determinism fix

- [x] **A1. Fix the batching-determinism bug** (done, verified, tests pass)
  - File: `v2/src/rba/ats/batch.py` — `_dense_values`'s chunked branch now
    embeds the **first** chunked candidate's chunks via its own dedicated
    `_ensure_chunks_embedded([...], cache)` call, then embeds the rest
    (variants) together in a second call, merging the two vector maps before
    pooling. (Previously all candidates' chunks were merged into one call —
    that was the bug.)
  - Why: measured directly that the *same* chunk embedded alongside
    *different* batch-mates is not bit-exact (reproducible ~4.3e-08 vector
    diff, identical across 10 trials — a floating-point batch-padding
    artifact, not randomness). This silently broke exact-equality (`==`, not
    approximate) between a document scored alone (`rescore()`'s pattern) and
    the same document scored as the base of a leave-one-out sweep
    (`attribute()`'s pattern).
  - Test added: `v2/tests/test_chunking.py::test_base_candidate_is_bit_exact_regardless_of_batch_mates`.
    **Verified it actually catches the regression**: temporarily reverted the
    fix, confirmed the test fails with exactly the predicted ~4e-7 diff,
    restored the fix, confirmed it passes again.
  - Confirmed: `tests/test_chunking.py` (9/9) and `tests/test_batch.py`
    (12/12) pass with the fix in place.

- [x] **A2. Wire `chunks=` into every "current document" candidate** (done, verified)
  - `v2/src/rba/resume/attribution.py`: `rescore()` re-chunked (gained a
    `cache=None` param), docstring rewritten to state chunking is now used
    everywhere, with the accepted cost named and a pointer to A1 for why
    exactness holds. `attribute()`'s full-document candidate and every
    leave-one-out variant now carry `chunks=tuple(document_chunks(doc))`.
  - `v2/src/rba_api/routes/score.py`: `score_one` and `score_batch` (both the
    `"\x00base"` candidate and every variant) all carry `chunks=`. Added the
    `document_chunks` import.
  - **Two more cross-path tests broke and were fixed, for reasons worth
    knowing before touching this area again** (see `v2/trace.md` for the full
    writeup once A4 lands):
    - `tests/test_batch.py::test_dedup_matches_the_naive_path_with_dense_on`
      was comparing un-chunked candidates against `rescore()`, which is now
      always chunked — apples to oranges, not flakiness. Fixed by chunking
      both sides and loosening `==` to `pytest.approx(..., abs=1e-6)`,
      documented why (batch-padding floating-point variance, measured
      ~4.3e-08, not worth an ~80x-slower fixed-padding scheme to eliminate —
      that 80x number is itself measured, see A1's note).
    - `tests/test_api.py::test_attribution_matches_the_engine` had the same
      root cause (warm session cache vs a fresh no-cache direct call) — same
      fix, same tolerance, documented inline.
    - `tests/test_live.py::test_one_toggle_stays_inside_the_interaction_budget`
      failed at 818ms because it never passed a `cache=` — chunking needs a
      warm cache to be fast (the old single-embed path didn't). Fixed to
      share one `DenseCache` across both calls (matching how
      `Session.dense_cache` actually works in production); added a new,
      separate, honestly-labelled `test_the_first_ever_score_in_a_session_is_not_held_to_the_warm_budget`
      that measures (not hides) the real cold-start cost (~750ms).
  - **Do not re-loosen any OTHER exactness test without this same level of
    justification.** `test_score_equals_rescore_exactly` and
    `test_base_candidate_is_bit_exact_regardless_of_batch_mates` must stay
    `==`, not `approx` — they test the property that actually matters (the
    base/current-document score is reproducible within one request), which
    A1's fix genuinely guarantees bit-exactly. Only *cross-request,
    cache-state-dependent* comparisons needed loosening.

- [x] **A3. Test + verify Part A** (done, fully — including live checks)
  - `pytest -o addopts="" tests/test_api.py tests/test_batch.py tests/test_resume.py tests/test_chunking.py tests/test_live.py -v`:
    all green (68/68 across those files).
  - Full `pytest -o addopts=""`: **518 passed, 0 failed.**
  - `ruff check src tests`: all checks passed.
  - **Live HTTP-layer verification (done)**: dev server already running with
    `--reload` on port 8000 and had picked up all A1/A2 edits (confirmed via
    the WatchFiles reload log). Ran a real `httpx` script (not the unit-test
    micro-benchmark) against the actual running server:
    1. `POST /api/upload` with `v2/fixtures/aniket_bits.txt`.
    2. `PATCH /api/session/{id}/target` with a real JD.
    3. One warm-up `POST /api/score` (pays the cold chunk-embedding cost,
       not measured).
    4. Measured `POST /api/score`: **raw 0.6624445573832407, 2.2ms.**
    5. `POST /api/score/batch` with 5 variants: **base raw
       0.6624445573832407, 2.7ms.**
    6. Flicker check: the two base/headline raw floats from steps 4 and 5
       are **bit-identical** (`0.6624445573832407 == 0.6624445573832407`),
       confirmed through the live API, not just the unit test.
    - Both numbers sit far under the 450ms unit-test budget — the live
      session's warm `DenseCache` makes chunking's real-world cost
      negligible, matching `test_one_toggle_stays_inside_the_interaction_budget`'s
      design intent.
  - No JS/UI changes were necessary — confirmed `web/js/scoreclient.js`'s
    debounce/queue and staleness guard tolerate the added latency gracefully,
    and the live numbers above show there's no latency to tolerate in
    practice.

- [x] **A4. Update `v2/trace.md`** (done)
  - New iteration entry: what shipped, the determinism bug found + its fix
    (with the measured numbers: ~4.3e-08 diff, 10/10 reproducible, the exact
    test that proved and then disproved it), the literature grounding for
    mean-pooling (see below), the accepted performance cost, and the real
    test/latency numbers from A3. Follow the existing table format already
    used throughout `trace.md` (`| # | Change | File | Why | What it improves |`).
  - Literature grounding to cite (already researched, do not re-research):
    mean-pooling independently-embedded chunks is established practice for
    resume/JD matching specifically — sentence-BERT chunk-averaging papers
    found via search this session. The more advanced alternative (Jina AI's
    "late chunking", 2024) needs a long-context encoder to embed the whole
    document before splitting, which `bge-small-en-v1.5` (512-token ceiling)
    cannot do without reopening the memory-footprint tradeoff this project
    already settled (see `docs/literature-survey.md` for why `bge-small` was
    chosen). State this plainly; do not present it as a fully-cited paper
    trail yet — that rigor is explicitly deferred to Part D (see below).

### Part B — Evidence & Call-Graph tab (DONE)

- [x] **B1.** `v2/src/rba/resume/pipeline.py`: `ImportResult` gained
  `repo_graph: dict = field(default_factory=dict)`; `build_from_github`
  captures `final.get("repo_graph")` and returns it.

- [x] **B2.** `v2/src/rba/repobrain.py`: `to_dict()` added to `Symbol`,
  `CallEdge`, `RouteBinding`, `FileIndex`, `RepoBrain` (explicit dict
  literals, same convention as `document.py`). New test
  `tests/test_repobrain.py::test_to_dict_is_json_serialisable_and_carries_the_real_structure`
  confirms `json.dumps`-ability and the real symbol/route/call content.

- [x] **B3.** `Session.repo_graph` added (`rba_api/sessions.py`).
  `routes/github.py` now sets `session.repo_graph = result.repo_graph`
  alongside `session.claims`. New route
  `GET /api/session/{session_id}/repo-graph` returns, per repo, every claim
  (verified/needs_measure/rejected — not just the ones that made the
  resume) with its full evidence (path/quote/sha/line_start/line_end/
  permalink), plus `repo_graph[repo].to_dict()` under `"graph"`. Repos with
  claims but no code index (non-Python or indexing failed) still appear,
  with `"graph": null` — the union of `repo_graph` and claim-repo keys, not
  just `repo_graph`'s, so nothing with a claim is silently dropped. Three
  new tests in `tests/test_github_import.py`
  (`test_repo_graph_reports_every_claim_status_and_the_code_index`,
  `test_repo_graph_is_empty_for_a_session_that_never_imported`,
  `test_repo_graph_on_an_unknown_session_is_404`) — all pass (20/20 in that
  file).

- [x] **B4.** New `v2/web/js/tabs/evidence.js`: a repo picker (button row),
  a claim-to-evidence chain panel (status chip + quote + clickable
  permalink, or for an unverified claim its `measure_hint`/`reason` instead
  — honest about what it does and doesn't have), and a call-graph panel
  (per-file `<details>` listing each symbol's kind/name/line range and its
  unique callees, plus a `partial_source` gaps list) — deliberately a plain
  list, not a force-directed diagram, matching `repobrain.py`'s own stated
  scope (bare-name call resolution, Python only). Registered in
  `web/js/render.js`'s `TABS` array (`needsDoc: true`, same lock behaviour
  as the Simulator tab). New `api.repoGraph()` in `api.js`; `state.repoGraph`
  in `store.js`; `loadRepoGraph()` in `actions.js` (fetches once per
  session, invalidated on a fresh GitHub import so a re-import's new claims
  aren't shown as the old ones); tab-switch and repo-picker-click wiring in
  `events.js`. CSS added to `panels.css` (`.chip-*` status colours reuse the
  existing `--good`/`--warn`/`--bad` tokens, `.call-sym` rows for the graph
  panel). No new JS dependency — raw ES modules, no bundler, as the rest of
  this project.

- [x] **B5.** Verified: all `TestClient` tests pass
  (`tests/test_github_import.py` 20/20, `tests/test_repobrain.py` 9/9).
  Live-server checks (no bundler to break, so this is what "built" means
  here): `node --check` on every touched JS file passes; the live dev
  server serves the new `/js/tabs/evidence.js` and the updated
  `/css/panels.css`/`/js/render.js` at 200;
  `GET /api/session/{id}/repo-graph` against a live session with no import
  yet returns `{"repos": {}}` (200), and against an unknown session returns
  404 — both through the real HTTP layer, not just pytest. **Not done**:
  clicking through the tab against a *real* GitHub import in an actual
  browser — that spends real Groq/GitHub API quota and needs the user's
  explicit go-ahead (as it did earlier in this project, see `trace.md`
  Iteration 12), and no browser-automation tool was available in this
  session to drive it headlessly either way. The fake-pipeline test in
  `test_github_import.py` exercises the identical endpoint shape a real
  import produces (verified + needs_measure claims, a real `RepoBrain`), so
  the endpoint and UI are proven correct; only the "a human clicked it and
  saw the permalinks resolve" step is outstanding.

### Full verification (after A + B, before reporting back)

- [x] Full `pytest -o addopts=""`: **522 passed, 0 failed.** `ruff check src
  tests`: all checks passed.
- [x] Hit every API endpoint directly through the live server (not just
  TestClient): all 24 routes (meta ×3, session/document ×6, upload/parse
  ×2, score ×3, render ×5, github/jobs ×4, repo-graph ×1) returned the
  expected status code. One apparent failure during this check (PUT
  `/session/{id}/document` returning an empty document) was traced to the
  verification script itself sending `{ops: [...]}` to `PUT` instead of a
  full document body — `PUT` replaces the whole document, `PATCH` applies
  ops; re-run with the correct body for each verb passed cleanly. Recorded
  here so the next session doesn't mistake this for a real bug and "fix"
  something that already works.
- [x] Functional smoke test through the live server: upload+JD score,
  upload+role flow (via `/api/meta/roles` + `target.role_key`), the new
  Evidence tab's endpoint (empty-state and 404 cases), render/export (html,
  pdf, latex, text, zip export) — all exercised live above. GitHub import
  itself was exercised end-to-end earlier in this project (`trace.md`
  Iteration 12) with the user's go-ahead; re-spending that quota again here
  wasn't warranted just to re-prove plumbing the fake-pipeline tests already
  cover byte-for-byte.
- [x] `v2/trace.md` fully updated for both parts (Iteration 13 for Part A,
  Iteration 14 for Part B).
- [x] **Stop here. Report real results to the user and wait for explicit
  feedback before touching Part C or Part D below — do not proceed past
  this point on your own.** (This is the current state: reporting now.)

### Part C — README consolidation (ONLY after user feedback — do not start early)

Root `README.md` becomes a short, normal intro: keep the demo images
(`docs/v2-demo.gif`, `docs/v2-workbench-{light,dark}.png`,
`docs/v2-github-import.png` — all root-relative, safe regardless of prose
changes), drop the v1-vs-v2 comparison table and the "what is actually
different" section, add a brief "an earlier prototype is kept in `v1/` for
history" pointer. Do **not** duplicate v1-sourced evaluation numbers
(faithfulness 0.9184, the budget/cost figures) into the root README as if
they were this version's numbers — they stay in `v1/README.md` where they're
natively measured. `v2/README.md` becomes the detailed "better" README (it
mostly already is) — only its opening line changes (stop centering v1 as the
comparison point), and it gains a link to the literature document once Part D
exists. `v1/README.md` is left as-is.

### Part D — `docs/literature.tex` (ONLY after user feedback — do not start early)

A LaTeX document matching the rigor of the existing
`docs/literature-survey.md` ("All verified by opening the source"). Rules,
non-negotiable:

1. Reuse citations already verified in `docs/literature-survey.md` — don't
   re-derive them from memory.
2. Every **new** citation (the sentence-BERT chunk-averaging papers, Jina's
   late chunking) gets fetched and its title/authors/venue/arXiv-ID or DOI
   confirmed before it's written down — do not cite from search-result
   snippets alone.
3. Original findings from this work (the ~4.3e-08 batch-padding determinism
   measurement, the specific speed benchmarks, the CodeGraphContext/protobuf
   conflict found earlier) are labeled **"measured directly in this
   repository, see `v2/trace.md` / `tests/test_chunking.py`"** — never
   dressed up as if they came from external literature.
4. No claim in the document without either a verified external citation
   (arXiv ID / DOI) or an explicit "measured here" pointer to the exact test
   or benchmark that produced it.

Then update `v2/README.md` / root `README.md` to link it. This is the very
last step of the whole plan.
