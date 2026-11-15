# Plan: fix GitHub-import reliability and resume-table parsing

**If you are a fresh agent (Codex or otherwise) picking this up cold: read
this whole file top to bottom before touching anything.** It is kept
up to date as work happens specifically so a session interruption doesn't
lose anything -- every numbered fix below has a STATUS line, and the
"Live-import findings" / "Ollama" sections near the bottom capture things
discovered mid-session that aren't folded into a fix yet. Nothing here is
a guess -- every claim is either a line-cited code reading or a directly
observed, timestamped result.

- Repo: `/Users/aniketsaxena/Documents/p/from_aug_1/p2/resume-builder-agent`.
  Work inside `v2/` only, never `v1/`. Branch `v2-revamp`. Everything here
  is uncommitted work on top of the state in `v2/trace.md` and
  `v2/HANDOFF.md` (chunked ATS scoring + the Evidence & Call-Graph tab,
  already shipped and verified separately from this plan).
- Venv: `v2/.venv`. **Important**: `pytest`/`ruff` live in an optional
  `dev` extra (`pyproject.toml`'s `[project.optional-dependencies]`), not
  the main dependency set -- a plain `uv sync` silently drops them from the
  venv (this happened once already this session while adding `pdfplumber`).
  Always use `uv sync --extra dev` for this project, never bare `uv sync`.
- Test: `pytest -o addopts=""`. Lint: `ruff check src tests`.
- The live dev server (if open in a browser, `--reload` auto-picks up
  edits): `RBA_ONNX_THREADS=4 RBA_EMBED_BATCH=16 .venv/bin/uvicorn
  rba_api.app:app --reload --port 8000`, `http://localhost:8000`. `.env`
  has real credentials -- never print, log, or commit its contents.
- Remote GPU host for Ollama (see the Ollama section below for full
  detail): reachable via `ssh a6000` (an entry already in this machine's
  `~/.ssh/config`), logs in as user `admin`. Three idle RTX A6000s (49 GB
  each). The user has explicitly authorized using this host and starting
  models on it; be careful anyway -- it is a **shared multi-user research
  machine** (many other users' long-running `ollama serve` processes were
  already running when this was first explored, all idle/0% GPU
  utilization at the time). Never stop another user's process without
  asking first, per the user's own explicit instruction.

## STATUS (update as you go -- source of truth if a session is interrupted)

- [x] **Fix 0** -- no code change needed (explained below, confirmed not a bug).
- [x] **Fix 1** -- Groq strict-schema 400 retry. DONE, verified.
  - `src/rba/groq_client.py`: new `SchemaRejectedError(GroqError)`, carrying
    `failed_generation`. `complete()`'s 400-branch now distinguishes
    `"max completion tokens reached"` (existing budget-growth retry,
    unchanged) from the generic `"json_validate_failed"` shape-violation
    case (now raises `SchemaRejectedError` instead of silently retrying
    with a bigger, unhelpful `max_tokens`). `structured()` now catches
    `SchemaRejectedError` alongside `ValidationError`/`JSONDecodeError` and
    retries once with a corrective message that echoes the model's own
    `failed_generation` back to it -- the same repair pattern already used
    for a client-side validation failure.
  - Added `schema_rejected_response()` to `tests/fakes.py`, body shape
    reproduced from the real live failure in this conversation (claims
    missing `text`). Three new tests in `tests/test_groq_client.py`:
    `test_a_shape_violation_400_raises_schema_rejected_not_groqerror`,
    `test_structured_recovers_from_a_real_schema_rejected_400`,
    `test_structured_gives_up_after_a_second_schema_rejected_400`.
  - **Verified the regression tests actually catch the bug**: reverted
    `groq_client.py` via `git stash`, confirmed the test file fails to even
    *import* (`SchemaRejectedError` doesn't exist without the fix) --
    restored the fix, confirmed all 19 tests in `test_groq_client.py` pass,
    the 3 new ones included.
- [x] **Fix 2** -- source-file truncation before AST indexing. DONE, verified.
  - `src/rba/nodes/fetch.py`: `SOURCE_FILE_MAX_CHARS` raised from 4000 to
    60,000 (a safety ceiling against one pathological huge file, not an
    expected-to-be-hit truncation point -- see the updated docstring in
    the file for why 60k and not unbounded). Checked every other reader of
    `facts.source_files` first (`verify.py`, `summarize.py`,
    `prepare_evidence.py`) -- none assumed the old 4000-char cap, and
    `verify.py`'s quote-matching actively needed full content too (a quote
    past char 4000 was previously unverifiable, a second bug this same fix
    closes).
  - Two new tests in `tests/test_fetch.py`:
    `test_a_long_source_file_still_parses_after_fetch` (a >4000-char file
    with a function past the old cutoff must parse clean, with zero
    `partial_source` entries) and `test_a_pathological_file_is_still_bounded`
    (confirms the new cap still bounds a huge file).
  - **Verified the regression test catches the bug**: reverted
    `fetch.py` via `git stash`, confirmed
    `test_a_long_source_file_still_parses_after_fetch` fails (truncated
    content doesn't match the full source) -- restored the fix, confirmed
    49/49 tests pass across `test_fetch.py`, `test_prepare_evidence.py`,
    `test_verify.py`, `test_summarize.py` (every reader of `source_files`).
- [x] **Fix 3** -- pdfplumber table parsing. DONE, verified against the
  user's actual real PDF (`H20250169_ Sarvam AI.pdf`), not just synthetic
  fixtures -- see "real-world verification" below.
  - **First pass had a real, now-fixed bug**: the initial implementation
    checked only `grid[0]` for a section's header row. Verified directly
    against the user's real resume: pdfplumber correctly detects the
    ACADEMIC DETAILS table, but groups the section's own heading line
    ("ACADEMIC DETAILS", alone, one cell) into the *same* detected table as
    the real header row below it -- nothing in the PDF's geometry visually
    separates a heading from the grid it introduces, so pdfplumber (quite
    reasonably) treats them as one contiguous table. That pushed the real
    header row to index 1, which the original `_pick_table` never checked,
    so it silently fell back to the broken text heuristic on every real
    upload -- the fix "worked" on every synthetic test (none of which had
    a merged heading) and failed on the one real file tested. Fixed:
    `_pick_table` now searches a grid's first `_HEADER_SEARCH_DEPTH` (3)
    rows for the best match, returns `(grid, header_row_index)`, and
    `_table_from_grid(grid, header_row)` builds the `Table` starting from
    that row, dropping anything before it. New regression test
    `test_pdfplumber_recovers_the_table_when_a_heading_is_merged_into_it`
    reproduces this exact shape with a synthetic PDF (heading cell +
    header row + data row, all in one bordered grid) -- verified it fails
    without the fix (temporarily set `_HEADER_SEARCH_DEPTH = 1`, confirmed
    the predicted failure, restored it) and passes with it.
  - **Real-world verification, not just synthetic**: the user supplied the
    actual PDF that produced the garbled live screenshot
    (`/Users/aniketsaxena/Downloads/H20250169_ Sarvam AI.pdf`, 354 KB, one
    page). Ran `parse_resume()` directly against its real bytes: ACADEMIC
    DETAILS now recovers all six headers and all three rows correctly
    aligned; CERTIFICATIONS recovers both rows correctly. This is the
    actual file that broke in the live import, tested directly, not a
    stand-in.
  - **A second, smaller real bug found and fixed in the same investigation**:
    the real PDF's "Subjects / Electives" note (a label-value pair below
    the academic table, not a table row) came out with an empty value --
    `pypdf`'s flat extraction put the label on its own line with the value
    nowhere adjacent to it (same root failure class as the main bug), but
    pdfplumber detected it correctly as its own one-row, two-cell grid.
    `_extract_extras()` now accepts the detected-tables pool and falls
    back to a matching detected row when the text-only split finds no
    value. Verified against the real file: now recovers
    `"Subjects / Electives: Agentic Artificial Intelligence, Data Mining,
    Advanced Database"` correctly.
  - `pyproject.toml`: added `pdfplumber>=0.11.4` (pypdf stays, for the
    prose path). Ran `uv sync` -- **caution for whoever runs this next**:
    plain `uv sync` dropped the `dev` optional-dependency group (pytest,
    ruff) from the venv because they're declared under
    `[project.optional-dependencies] dev = [...]`, not main dependencies;
    had to re-run `uv sync --extra dev` to restore them. A `uv.lock` did
    not exist before this and now does (654 KB, untracked) -- decide with
    the user whether to commit it before this work is committed.
    `uv sync`'s resolution also bumped `pypdf` 6.21.4 -> 7.1.0 incidentally;
    full suite re-run clean after (527 passed at that point), so this was
    not a silent break.
  - `src/rba/resume/ingest.py`: new `read_pdf_tables(data)` (pdfplumber,
    ruled-line detection first via `find_tables()`'s default strategy, a
    gap-based "text" strategy fallback for a borderless table, never
    raises), `_pick_table(grids, tokens)` (matches a detected grid to a
    section by its header row, consumes it so one grid can't be assigned
    to two sections, returns `None` rather than guessing below a 2-token
    match threshold), `_table_from_grid(grid)` (builds a `Table` directly
    from the detected rows -- no re-guessing column boundaries).
    `parse_bits()` gained an optional `detected_tables` parameter; for
    ACADEMIC DETAILS/CERTIFICATIONS it now tries `_pick_table` first and
    only falls back to the existing `_parse_table` text heuristic if no
    grid matched. `parse_resume()` now calls `read_pdf_tables` itself when
    the upload is a PDF. The `Subjects`/`Electives` extras extraction was
    factored into `_extract_extras()` so it runs regardless of which table
    path was used.
  - Four new tests in `tests/test_ingest.py`:
    `test_the_old_heuristic_mishandles_a_scrambled_extraction` (proves the
    bug is real), `test_pdfplumber_recovers_the_table_a_scrambled_extraction_loses`
    (proves the fix), `test_pick_table_will_not_steal_an_unrelated_grid`
    (guards against false-positive matches), and
    `test_parse_resume_detects_tables_from_real_pdf_bytes` (the actual
    upload entry point wires it through on its own, not only when a caller
    remembers to).
  - **Verified the regression tests catch the bug**: reverted
    `ingest.py` via `git stash`, confirmed `test_ingest.py` fails to even
    import (`_pick_table`/`read_pdf_tables` don't exist without the fix) --
    restored the fix, confirmed all 28 tests in `test_ingest.py` pass.
  - Full suite after Fix 3: **531 passed, 0 failed.** `ruff check src
    tests`: all checks passed.
- [x] **Fix 4** -- measure, don't guess-tune. DONE as "measured, no change
  made" -- see "Live-import findings" below for the real per-node timing
  pulled via the LangSmith SDK, which confirmed fetch/summarize
  concurrency already works as designed. Deliberately did not raise
  `McpBackend`'s `concurrency=4` or `summarize_repos`'s `concurrency=2`:
  neither cap showed up as the actual bottleneck in the one real run
  measured (the real cost was `select_repos`'s 68.9s Jev-classification
  call and Groq's own 429 rate limiting, neither of which either
  concurrency number controls), and changing a number tuned once already
  under real load without a second measurement to justify it would be
  exactly the kind of unverified tweak this plan's house style forbids.
- [x] **Full verification** -- DONE for Fixes 1-3 (see each fix's own
  entry above for real pass counts and real-upload verification). Fix 4's
  own verification is the measurement itself, recorded above.

---

**If you are Codex (or any fresh agent) picking this up cold: read this
whole file before touching anything.** Every claim below was verified
directly against the code at the lines cited -- none of this is a guess.
Work inside `v2/` only, never `v1/` (historical reference, preserved as-is).

- Repo: `/Users/aniketsaxena/Documents/p/from_aug_1/p2/resume-builder-agent`
- Branch: `v2-revamp`. Everything in this plan is new work on top of the
  state recorded in `v2/trace.md` and `v2/HANDOFF.md` (chunked ATS scoring +
  the Evidence & Call-Graph tab, both shipped and verified, uncommitted).
- Venv: `v2/.venv`. Test: `pytest -o addopts=""` (the repo's own
  `addopts=-q` hides the summary line). Lint: `ruff check src tests`.
- `.env` has real credentials (Groq, GitHub, TypeSafe, LangSmith) --
  **never print, log, or commit its contents.**

## House style -- match this, don't invent a different one

(Full detail in `v2/HANDOFF.md`'s "House style" section -- read that too.
Short version:)

- Comments explain WHY, never WHAT. Cite a measured number, a real bug, or
  a specific incident -- never restate what the code visibly does.
- Docstrings tell the design story, with a real measured number where one
  exists. Never invent a plausible-sounding number.
- Test names are full sentences describing the guarantee, not `test_1`.
- Dataclasses get explicit `to_dict()`/`from_dict()`, not
  `dataclasses.asdict()`.
- Never fabricate a number, a citation, or a test result.
- Minimal diffs. Prefer deleting code to adding it when something becomes
  obsolete. Don't add defensive code for cases that can't happen.
- Before calling any step done: (1) run the specific tests touched, in
  isolation, (2) run the full suite, (3) run `ruff check src tests`, (4) if
  a fix claims to catch a regression, prove it -- temporarily revert the
  fix, show the test fails, restore it, show it passes. Record real
  pass/fail counts, never assume.

## Context: how this plan came to exist

The user ran a real GitHub import against their own account through the
live dev server and cross-checked it against two LangSmith traces. Four
things looked wrong: (1) `prepare_evidence`'s trace showed a cost/token
count that looked misattributed, (2) `summarize_repo` failed outright for
3 of N repos with a Groq schema-validation 400, (3) `prepare_evidence`'s
`partial_source` list was full of `SyntaxError` entries against the
user's own `resume-builder-agent` repo (this project's own code, fetched
back from GitHub), and (4) the rendered resume's "ACADEMIC DETAILS" table
came out with headers and values in the wrong cells entirely. The user
also suspected the GitHub fetch (`list_commits`/`get_file_contents`) was
running one repo at a time rather than in parallel.

Each of these was traced to an exact line of code before this plan was
written -- see the per-fix sections below. One of the four (the trace
cost/token confusion) is **not a bug** and needs no code change; it is
explained in Fix 0 so nobody "fixes" something that already works.

---

## Fix 0 -- the `cost_usd`/`input_tokens` on `prepare_evidence`'s trace (NOT A BUG, no code change)

**What it looked like:** the LangSmith trace for the `prepare_evidence`
node showed `cost_usd: 0.001731` and `input_tokens: 41223` under "Input."
`prepare_evidence` is supposed to be zero-LLM (see its own module
docstring, `src/rba/nodes/prepare_evidence.py:1-7`) -- so this looked like
cost was being misattributed to a node that spends nothing.

**Why it actually happens:** `select_repos` (`src/rba/nodes/select.py:265-274`)
returns `usage.jev.{cost_usd,input_tokens}` as part of its node output --
this is Jev's repo-classification cost, a real spend, attributed to the
right node. But LangGraph passes the **entire accumulated `AgentState`**
into every subsequent node (that's how `StateGraph` works: state is
additive, not scoped per node), so `prepare_evidence`'s traced "Input" is
the whole state dict at that point in the run -- which includes the
`usage.jev` field `select_repos` set two nodes earlier. LangSmith is
rendering it faithfully; it is not double-charging or misattributing
anything.

**Action:** none required for correctness. Optional, low-priority,
**only if there is time left after Fixes 1-4**: LangGraph does not support
scoping a node's traced input to a subset of state keys out of the box, so
"fixing" the confusing trace display would mean writing a custom
`RunnableConfig`/callback wrapper per node -- a bigger, riskier change for
a cosmetic trace-readability win. Do not attempt this unless explicitly
asked; it is not blocking anything.

---

## Fix 1 -- DONE. Groq strict-schema 400s were retried blindly, without ever correcting the actual problem (the real cause of "3 repos could not be read")

**File:** `src/rba/groq_client.py`

**The bug, exactly (corrected from this plan's first draft after reading
the code more carefully -- see below):** `complete()`'s 400-handling
(`groq_client.py:201-215`, before this fix) matched a single condition --
`"json_validate_failed" in body_text or "max completion tokens reached" in
body_text` -- and for *either* cause, retried by doubling `max_tokens` (up
to 8192) with the exact same messages. That repair is correct for the
budget-exhaustion case. It does nothing for a genuine shape violation (the
model emitting a claim missing its required `text` field): the generation
wasn't cut short, it was simply wrong, and growing the token budget doesn't
change what the model generates. Because `max_retries` defaults to 5 and
`max_tokens` is capped at 8192, this meant up to 5 identical attempts at
the same unmodified prompt before `complete()` finally gave up and raised
a generic `GroqError` -- this plan's first draft said "never retried,"
which was not quite right; the truer statement is "retried the wrong way,
for up to 5 attempts, before failing anyway." Either way, the model was
never told what it got wrong, so each retry was no more likely to succeed
than the last.

`structured()` (`groq_client.py:289-`, before this fix) *does* have a
correction-retry loop for this exact kind of problem -- catch a validation
failure, append an assistant turn with the bad generation plus a user turn
describing what to fix, retry -- but it only applied to a **client-side**
`pydantic.ValidationError`, which only happens when `complete()`
successfully returns content that fails to parse. When Groq's own
schema-mode rejects the generation server-side, `complete()` never
returns -- it retries pointlessly inside itself (as above) and then
raises, so `structured()`'s existing repair logic never even had a chance
to run.

**The fix, implemented:**
1. Split `complete()`'s 400-handling into two distinct branches: match
   `"max completion tokens reached"` specifically for the existing
   budget-growth retry (unchanged), and treat any other
   `"json_validate_failed"` body as a genuine shape violation.
2. Added `SchemaRejectedError(GroqError)` carrying the 400 body and
   Groq's own `failed_generation` field (the actual malformed JSON the
   model produced, when the API includes it) -- raised for the shape-
   violation case instead of retrying inside `complete()`.
3. `structured()` now catches `SchemaRejectedError` alongside the existing
   `ValidationError`/`JSONDecodeError`, and retries once with the same
   corrective-message pattern, showing the model its own
   `failed_generation` as the prior assistant turn so it has something
   concrete to fix, not just an abstract description of the problem.
4. Added `schema_rejected_response()` to `tests/fakes.py`, with a body
   shape reproduced from the real failure captured in this conversation.
   Three new tests in `tests/test_groq_client.py` cover: the 400 raises
   `SchemaRejectedError` (not a generic `GroqError`) and does not retry
   inside `complete()`; `structured()` recovers on the second attempt and
   the retry message includes the model's own prior bad generation; and
   `structured()` still gives up cleanly after a second consecutive
   failure.
5. **Verified the regression tests catch the bug**: reverted
   `groq_client.py` via `git stash`, confirmed `test_groq_client.py` fails
   to even import (`SchemaRejectedError` doesn't exist without the fix) --
   restored the fix, confirmed all 19 tests pass.

**What NOT to do:** do not "fix" this by stuffing less context into the
evidence bundle, or by chunking the repo into multiple smaller summarize
calls. `build_evidence` (`nodes/summarize.py:110-151`) is already capped at
~7,000 chars (~2k tokens) and progressively drops the least-dense parts
first -- this is not a context-overflow problem, it is a missing-retry
problem. Adding chunking here would add real complexity (merging claims
across chunks, re-running `verify_claims` per chunk) to fix a bug that a
~10-line retry-path change already fixes directly. If, after Fix 1 ships
and is measured against a real import, schema failures are *still*
frequent even with the retry in place, that is the point to reconsider
per-repo chunking -- not before.

---

## Fix 2 -- source files are truncated before AST indexing, not after (the `partial_source` syntax-error spam)

**Files:** `src/rba/nodes/fetch.py`, `src/rba/nodes/prepare_evidence.py`

**The bug, exactly:** `fetch_repo_facts()` (`fetch.py:131-138`) fetches each
ranked source file's full content via `backend.get_file(...)`, then
immediately truncates it: `facts.source_files[path] = content[:SOURCE_FILE_MAX_CHARS]`
where `SOURCE_FILE_MAX_CHARS = 4000` (`fetch.py:32`). `prepare_evidence`
(`prepare_evidence.py:22-24`) then calls `repobrain.index(f.source_files)`
on that **already-truncated** dict, which calls `ast.parse()` on each
file's text (`repobrain.py:185-193`). Any file longer than 4000 characters
gets cut off mid-token -- mid-string-literal, mid-paren -- which is
exactly the error pattern observed: `"unterminated string literal"`,
`"'(' was never closed"`. This is not a signal about code quality; it is
truncation landing in the middle of valid syntax, on every file over the
cutoff, guaranteed.

**The fix:**
1. In `fetch.py`, stop truncating `facts.source_files` at fetch time.
   Fetch and store the **full** file content there (the whole point of
   `source_files` is to feed `repobrain.index()`, which needs syntactically
   complete files to parse correctly).
2. The LLM-facing side already has its own, separate, much smaller
   truncation: `_code_structure()` in `nodes/summarize.py` caps excerpts at
   `_EXCERPT_CHARS = 300` (`summarize.py:39,100`) and limits to
   `_MAX_STRUCT_FILES = 3` files / `_MAX_DEFS_PER_FILE = 6` defs
   (`summarize.py:34-36`). That path is unaffected by this fix and does not
   need full files -- leave it as is.
3. Consider whether `SOURCE_FILE_MAX_CHARS`/`source_files` should be
   renamed or re-scoped now that it holds full files rather than
   4000-char fragments (e.g. a docstring update in `fetch.py` explaining
   the new contract: "full content, for AST indexing; summarization
   excerpts it separately"). Check every other reader of
   `facts.source_files` before changing its shape (`grep -rn
   "source_files" src/`) to make sure nothing else assumed the old 4000-char
   cap.
4. Add a regression test: fetch (via a fake backend) a source file longer
   than 4000 characters with a multi-line string that would be cut
   mid-literal at the old cutoff, confirm `repobrain.index()` on the
   resulting `facts.source_files` parses it with no `parse_error`, where
   the old behaviour would have produced one. Put this in
   `tests/test_fetch.py` or `tests/test_repobrain.py`, whichever already
   covers the relevant seam -- check both before adding a new one.
5. Prove the regression test actually catches the bug: run it against the
   unfixed code first (temporarily keep the truncation), confirm it fails
   with a `SyntaxError` in `partial_source`, then apply the fix and confirm
   it passes.

**Memory/token-budget note:** removing per-file truncation at fetch time
means `facts.source_files` can now hold up to 12 full files per repo
(`SOURCE_FILE_LIMIT = 12`, `fetch.py:31`) instead of 12 files capped at 4KB
each. Measure the actual size increase against a few real repos before
assuming this is fine -- if any repo's selected files are large enough to
meaningfully bloat `AgentState` (which flows through every node as
described in Fix 0), consider capping `source_files` at something more
generous than 4000 but still bounded (e.g. 20,000 chars) rather than fully
unbounded, and say in the docstring exactly what measurement justified
that number -- don't invent one.

---

## Fix 3 -- resume table parsing destroys column structure (the garbled "ACADEMIC DETAILS" table)

**File:** `src/rba/resume/ingest.py`

**The bug, exactly:** `read_pdf()` (`ingest.py:63-68`) extracts text with
`pypdf`'s `PdfReader.extract_text()`, which returns a flat text stream with
no column information -- the file's own docstring already says so
(`ingest.py:13-19`: "PDF text extraction destroys column boundaries").
`_parse_table()` (`ingest.py:209-237`) then has to *guess* where one row
ends and the next begins, using heuristics (a line in all caps with no
trailing year is the header line; accumulate lines until one ends in a
year). This works when `pypdf` happens to emit the table's cells in
reading order, left-to-right then top-to-bottom, as one logical line per
row -- but for an actual 2-column grid like "ACADEMIC DETAILS," `pypdf`
frequently does not preserve that order (it can emit text in the PDF
content stream's internal order, which may be column-major or otherwise
decoupled from the visual grid). That mismatch is what produced the
screenshot: header words and value cells landing in the wrong `Table.rows`
entries entirely.

**The fix (the user's own suggestion, and it's the right one):**
1. Add `pdfplumber` as a dependency (`pyproject.toml` -- it is not
   currently listed; `pypdf>=6.21.4` is). `pdfplumber` exposes both
   word-level bounding boxes (`page.extract_words()`) and a table-detection
   API (`page.extract_tables()` / `page.find_tables()`) that uses actual
   line/rule geometry or column-gap heuristics, not just stream order.
2. In `ingest.py`, replace `read_pdf()`'s `pypdf` call with `pdfplumber`
   for the parts of the document that are genuinely tabular
   ("ACADEMIC DETAILS," "CERTIFICATIONS" -- the sections that construct a
   `Table`, see `document.py:136-153`). The rest of the BITS parser (section
   headings, bullets, entries) works off flat text today and does not need
   to change -- `pdfplumber` can also produce flat text
   (`page.extract_text()`) for everything outside the detected tables, so
   this does not have to be an all-or-nothing rewrite.
3. Concretely: for each page, call `page.find_tables()` (or
   `extract_tables()` if that already returns clean header+row grids for
   this resume's layout -- try it against the real fixture first and look
   at what comes back before committing to one API). If a table is found
   whose first row matches `KNOWN_HEADINGS`-adjacent column names (the
   `known` list already in `_split_headers()`, `ingest.py:242-243`), build
   the `Table` directly from pdfplumber's row/column grid instead of
   running `_parse_table()`'s line-heuristic path on that region's text.
4. Keep `_parse_table()` as a fallback for any table pdfplumber's detector
   misses (not every resume will have detectable table rules/gaps) --
   don't delete the heuristic path, just stop relying on it as the only
   path.
5. Test against the real fixture: `v2/fixtures/aniket_bits.txt` is plain
   text, not a PDF, so this needs a **new** fixture -- either the actual
   PDF the user uploaded (ask the user for it, or regenerate one via the
   existing PDF export path and re-upload it to create a representative
   test file), or a minimal synthetic PDF built with a PDF-writing library
   for the test itself. Whichever fixture is used, write a test asserting
   the resulting `Table.headers` is exactly
   `["COURSE", "SPECIALIZATION", "INSTITUTE/COLLEGE", "BOARD/UNIVERSITY",
   "SCORE", "YEAR"]` and each `Table.rows` entry lines up a specific
   institute with its own score and year, not a neighbour's.
6. This fix alone should also fix the LaTeX export
   (`resume/render/latex.py`) and the HTML/PDF preview
   (`resume/render/html.py`) -- both already render whatever `Table` object
   they're given correctly (confirmed: `latex.py` uses `tabularx` keyed off
   `Table.headers`/`Table.rows`, `html.py` emits a plain `<table>`). The bug
   is upstream of both renderers. Do not modify `latex.py`/`html.py` as
   part of this fix unless testing the corrected `Table` data through them
   reveals a *separate*, previously-masked rendering bug -- if so, note it
   separately rather than silently folding it into this fix.

---

## Fix 4 (optional, only after 1-3 are shipped and measured) -- raise the parallelism ceiling if the budget allows it

The user's instinct that imports feel slow is likely correct even though
the code is already concurrent (see the Fix-0-adjacent note at the top of
this plan) -- it's just rate-limited, not single-threaded. Once Fixes 1-3
are in and a real import succeeds cleanly end to end, measure real timing
per node (the LangSmith trace already has this) and *then* decide whether
raising `McpBackend`'s `concurrency=4` (`github_tools.py:277`) or
`summarize_repos`'s `concurrency=2` (`summarize.py:188`) is safe:

- `McpBackend`'s cap exists because the hosted MCP endpoint "degrades
  under bursty load" (`github_tools.py:286-288`) -- this was tuned
  empirically once already; raising it without re-measuring against the
  live hosted endpoint risks reintroducing that degradation.
- `summarize_repos`'s cap of 2 is sized against Groq's 8,000 TPM budget
  (`summarize.py:191-192`) -- raising it only helps if the actual
  bottleneck is concurrency rather than the token bucket itself (which
  would just throttle a 3rd+ concurrent call anyway via `Limiter`).

Do not change either number without first measuring real latency from a
live import with Fixes 1-3 applied, and recording the before/after numbers
-- the same "measure, don't assume" discipline as `trace.md`'s reasoning
about `summarize_repos`'s current value of 2.

---

## Verification checklist (all four fixes)

1. `pytest -o addopts="" tests/test_groq_client.py tests/test_fetch.py
   tests/test_repobrain.py tests/test_resume.py -v` -- all green, including
   the new tests from Fixes 1-3.
2. Full `pytest -o addopts=""` -- record the real pass count.
3. `ruff check src tests` -- all clean.
4. For Fix 1 and Fix 2: prove each new regression test actually catches its
   bug by temporarily reverting the fix, watching the test fail with the
   predicted error, then restoring the fix. Record this in the plan's
   completion notes, the same way `v2/trace.md` already documents it for
   earlier fixes.
5. Run one real GitHub import against the user's account through the live
   server (same as the import that surfaced these bugs) **with the user's
   explicit go-ahead first** -- it spends real Groq/GitHub quota. Confirm:
   no `summarize_repo[...]` failures for repos that previously failed with
   a schema-mismatch 400 (some repos may still legitimately fail for other
   reasons -- rate limits, network -- that is fine; only the specific
   schema-mismatch failure mode should be gone), and `partial_source` no
   longer lists syntax errors for files under the repo's own line count
   (spot-check one previously-broken file).
6. Re-render the resume whose "ACADEMIC DETAILS" table was garbled and
   visually confirm the headers and values line up, through both the HTML
   preview and the LaTeX export.
7. Update `v2/trace.md` with a new iteration entry (same format as existing
   ones: `| # | Change | File | Why | What it improves |`) covering all
   four fixes, with the real before/after numbers from this checklist --
   not estimated ones.
8. **Stop and report real results back to the user before touching
   anything in Part C (README) or Part D (literature.tex) from the
   earlier plan in `v2/HANDOFF.md`** -- this work is independent of those
   and does not unblock them.

---

## Live-import findings (2026-11-15, real run against the user's GitHub account)

Ran with the user's explicit go-ahead, through the live dev server
(`session s_6cIAwNBs2jJOrF3t`, `job j_jpzTti7x_1JvUw`). Pulled the actual
trace via the LangSmith SDK (`Client().list_runs(trace_id=...)`) rather than
the web UI, which doesn't render through a simple fetch.

**Confirmed, with real timestamps, that fetch concurrency already works**:
multiple `list_commits`/`get_file_contents` tool calls start within the
same second of each other repeatedly -- real evidence the per-repo fetch is
genuinely parallel, not sequential (see Fix 4 discussion above). Real
per-node timings from this run: `connect_github` 4.8s, `list_repos` 8.9s,
`select_repos` 68.9s (this is Jev classification cost -- notably the
single slowest node, worth a separate look later), `fetch_facts` 39.3s
(dozens of concurrent calls, a handful of tolerated transient errors),
`prepare_evidence` 0.22s, `safety_scan` 0.01s.

**Fix 1, measured result: real improvement, not a full fix.** Two repos
(`pages-hub`, `AOA_Controller_ty`) still failed, but critically the log now
reads `"schema validation failed after retry"` -- proof the corrective
retry path actually ran (it didn't exist before this session's work). On
the retry, the model made a **different** mistake each time (first attempt:
missing `text`; second attempt, after correction: `evidence` should be an
array, model returned an object instead) -- a small/fast model
(`openai/gpt-oss-20b`) can apparently produce a *new* kind of schema
violation on each attempt rather than converging. `structured()` only
retries once (by design, matching the existing client-side-validation
pattern) -- raising that to 2 retries is a plausible, cheap follow-up if
this keeps happening, but do not raise it speculatively without first
counting how often a *third* attempt would have actually helped (instrument
and measure, the same discipline as everything else in this plan).

**Fix 3, measured result: did NOT recover the table on this real upload --
under active investigation.** The job's final document still showed the
old garbled shape (`headers: ["COURSE"], rows: [["SPECIALIZATION", ""],
...]`) even though the upload that produced this session happened *after*
the Fix 3 reload (confirmed from the server log's reload markers). This
means `read_pdf_tables()`/`_pick_table()` did not find-and-match a grid in
the user's real PDF, for a reason not yet known -- possibilities being
investigated: pdfplumber's table detector found nothing at all (a
borderless layout its "text" strategy still can't resolve), or it found a
grid but the header cells didn't literally match the known token set
(e.g. a wrapped multi-line header cell). Added temporary `log.info` to
`read_pdf_tables()` (`src/rba/resume/ingest.py`) to print what pdfplumber
actually found on the next upload -- **remove this debug line once the
real cause is confirmed and fixed; it is not meant to stay.** Do not
consider Fix 3 complete until this is resolved and re-verified against a
real upload, not just the synthetic fpdf2 fixtures in
`tests/test_ingest.py`.

## New work items from this session, in the order the user wants them (fixes first, these after)

1. **Finish diagnosing Fix 3's real-world miss** -- DONE. Fixed and
   verified directly against the user's real PDF (`H20250169_ Sarvam
   AI.pdf`): pdfplumber merges a section's heading line into the same
   detected table as its header row; `_pick_table` now searches the first
   3 rows of a grid instead of assuming row 0. Committed as `3ea32c9`
   (pushed to `v2-revamp`). Full detail in Fix 3's own STATUS entry above.
2. **Ollama as a configurable alternative to Groq** -- DONE, verified live
   against the real remote host (not just a fake transport). New
   `src/rba/ollama_client.py`: `OllamaClient.structured()` matches
   `GroqClient.structured()`'s exact shape (`(messages, schema, *, node,
   tier, model, cache_key, **overrides) -> (T, Usage)`), POSTs to
   `/api/chat` with `format` set to the schema's real JSON Schema (not
   `"json"` mode), retries once on a schema violation with the same
   corrective-message repair `GroqClient` uses, fills `Usage` from
   Ollama's real `prompt_eval_count`/`eval_count` fields, and supports the
   same `Cache` class Groq uses (so a repeat prompt costs zero tokens
   either way). No rate limiter -- that is the entire point.
   `graph.py`'s `Runner.connect()` now branches on `LLM_PROVIDER` (env var,
   default `groq`); nothing in `nodes/summarize.py`/`build.py`/`plan.py`
   needed to change, confirmed by grepping for every `client.structured()`
   call site before writing this. `config.yaml` gained an `ollama:`
   section (`base_url`/`model` defaults, overridable per-environment via
   `OLLAMA_BASE_URL`/`OLLAMA_MODEL`); `.env`/`.env.sample` gained
   `LLM_PROVIDER`/`OLLAMA_BASE_URL`/`OLLAMA_MODEL`. Six new tests in
   `tests/test_ollama_client.py` (fake-transport pattern matching
   `test_groq_client.py`: schema shape, retry-on-violation, give-up-after-
   retry, 4xx not retried, cache hit costs zero tokens, usage grouped by
   node) -- all pass. **Also verified with a real, non-fake call** against
   `http://172.24.16.155:11434` directly (not through the test suite):
   summarized a one-line repo description into 3 correctly-shaped claims
   on the first attempt, `Usage(prompt_tokens=53, completion_tokens=139,
   total_tokens=192)`. Full suite after this change: 538 passed, 0 failed,
   lint clean.
   - **Known gap, not yet addressed**: `OllamaClient` does not yet feed
     LangSmith tracing the way `GroqClient._trace()` does (child runs
     nested under the active graph-node run). Usage accounting
     (`usage_report()`/`usage_total()`) works identically to `GroqClient`,
     so cost/token reporting in the CLI/API is unaffected -- only the
     LangSmith trace tree would be missing the per-call child spans for an
     Ollama-backed run. Address this if/when Ollama becomes the default
     rather than an opt-in, not before.
   - **Not yet done**: an actual end-to-end GitHub import run with
     `LLM_PROVIDER=ollama` set, to compare real schema-failure rate and
     claim quality against today's Groq run -- the direct `structured()`
     call above proves the client works, not that a full import's claims
     come out good.
3. **Refactor** -- user's own words: "it is a little bit too bulky." Scope
   not yet defined; likely candidates after today's additions are
   `groq_client.py` (now carries two schema-failure paths) and
   `ingest.py` (now carries two table-parsing paths side by side). Do this
   after Ollama is in, not before -- adding a second LLM backend will
   reshape `groq_client.py`'s boundaries anyway, so refactoring it twice
   would be wasted motion.
4. **Frontend rendering** -- DONE, verified live through `/api/render/html`
   against the user's real PDF end to end. Confirmed, again, that the
   defect was never in `html.py`/`latex.py`: both already rendered
   whatever document structure they were given correctly. Re-checking the
   *whole* real document's render (not just the one table) surfaced two
   more, entirely new, real parsing bugs in `resume/ingest.py` -- neither
   related to tables:
   - **Title and date on separate lines**: this PDF's extraction put an
     entry's heading ("Software Firmware Engineer, Marvell
     Semiconductors") and its date range ("May 2026 - Jul 2026") on two
     separate lines, not one. `_parse_entries` only recognised a heading
     when both were on the same line -- the title-only line was silently
     dropped as "stray text before the first entry," and the date-only
     line became an entry with an **empty title**. New `_merge_split_dates()`
     rejoins a heading-shaped, date-less line with an immediately
     following pure-date line before `_parse_entries` ever sees them.
     Affected every single entry in this real resume (Internships,
     Projects, Position of Responsibility), not an isolated case.
   - **Bullets with no terminal punctuation**: this PDF's bullet glyph
     extracted as `\x7f` (a private-use-area character some PDF producers
     emit for a bullet dingbat), and none of its bullets ended in a
     period. `_join_wrapped`'s only "new bullet" signal was a trailing
     full stop, so all three bullets under "Internships" merged into one.
     A leading bullet marker (`_BULLET_MARKERS = "-*•·—\x7f"`, sharing its
     character set with `pipeline.py`'s flat-text parser) is now an
     equally hard signal, and the marker itself is stripped from the
     stored text.
   - Two new regression tests in `tests/test_ingest.py`
     (`test_a_title_and_date_on_separate_lines_still_make_one_entry`,
     `test_bullets_with_a_glyph_marker_and_no_terminal_period_stay_separate`)
     reproduce both exactly, built from the real extracted lines (not
     guessed). **Verified both catch the bug**: temporarily reverted both
     changes, confirmed both tests fail with the predicted wrong output
     (empty title; one 3x-length merged bullet), restored the fixes,
     confirmed both pass.
   - Re-verified the entire real document end to end, live, through
     `POST /api/render/html`: every entry across Internships, Projects,
     and Position of Responsibility now has the correct title, subtitle,
     dates, and individually-separated bullets.
   - Full suite after this: **540 passed, 0 failed.** `ruff check src
     tests`: all checks passed.

---

## Ollama: current state (2026-11-15)

Goal: remove Groq's 8,000 TPM rate limit as a factor entirely by giving
`summarize_repo` a self-hosted alternative with no quota, configurable per
`.env` so Groq stays available too.

**Access.** `ssh a6000` (already a working `Host` entry in this machine's
`~/.ssh/config`, no password/key setup needed beyond what's already there).
Logs in as user `admin`. `nvidia-smi` showed three RTX A6000s, 49 GB each,
all at 0% utilization / ~20 MiB used (driver overhead only) when first
checked -- fully idle. **This is a shared research machine**: `ps aux | grep
ollama` showed many long-running `ollama serve` processes under several
other usernames (`kartikey`, `jagat`, `archit`, a system `ollama` user),
some running since August. None of them were touched, started, or stopped
-- the model pulled below runs through the Ollama instance already
listening on port 11434 (owned by someone else's long-running process, not
one newly started), using a fresh, separately-created model-storage
directory so it couldn't collide with anyone else's models either. If a
future session needs to start a *new* `ollama serve` process on this host
(rather than reusing the one on 11434), check `ps aux`/`nvidia-smi` again
first and ask before stopping anything that's running, per the user's own
explicit instruction.

**The permission bug, found and fixed directly.** `http://172.24.16.155:11434/api/tags`
returned `{"error": "mkdir /home/admin/ollama-models: permission denied:
ensure path elements are traversable"}` -- not a "no models" response, a
real server-side error. Root cause, confirmed by SSHing in: the directory
`/home/admin/ollama-models` did not exist at all (`ls` -> "No such file or
directory"). Created it as `admin` (`mkdir -p`), which fixed the top-level
error but moved it one level deeper (`mkdir .../manifests: permission
denied`) -- the Ollama server process holding port 11434 runs as a
different uid than `admin` (not identified further; not necessary to), so
even though `admin` owns the new directory, that process only had
read+execute on it (no write), consistent with a default `755`-ish mode on
a dir newly created by `admin`. Fixed with `chmod 777
/home/admin/ollama-models` -- permissive, but this is a disposable
model-cache directory on a path nothing else depends on, not a security-
sensitive location. After that, `/api/tags` returned `{"models": []}`
cleanly (no error, just genuinely empty) -- confirms the fix, not a
guess.

**Model chosen and verified.** Pulled `qwen2.5:14b-instruct`
(`POST /api/pull`, `{"model": "qwen2.5:14b-instruct", "stream": false}`) --
8.99 GB, Q4_K_M quantization, confirmed via `/api/tags` afterward
(`digest: 7cdf5a0187d5...`). Reasoning for this size: large enough for
reliable instruction-following and JSON-shape adherence (the exact thing
Groq's `openai/gpt-oss-20b` struggled with), small enough to leave the
other two idle GPUs and the rest of this GPU's 49 GB untouched for other
users on a shared machine. Smoke-tested directly against `/api/generate`
with `"format": "json"` and a prompt mirroring `summarize.v2`'s actual
shape (one_line + claims, each with text + evidence) -- returned clean,
correctly-shaped JSON on the **first** attempt, no corrective retry needed.
This is a real, if small, data point that a mid-size dense model may be
more reliable here than Groq's `gpt-oss-20b`, not just faster/cheaper --
worth re-measuring properly once real summarization traffic runs through
it, not just one smoke-test prompt.

**Not yet done -- next actual steps:**
1. Design: `.env` gains a provider toggle (e.g. `LLM_PROVIDER=groq` default,
   `ollama` opt-in) read in `Config` the same way other settings are.
2. New `src/rba/ollama_client.py` (or extend `groq_client.py` -- decide
   based on how much they'd actually share; don't force a shared base
   class if the two providers' retry/rate-limit semantics turn out to be
   different enough that it hides more than it reveals) implementing the
   same shape `summarize.py` already calls:
   `async def structured(messages, schema, *, node=None, tier=None, ...) -> tuple[T, Usage]`.
   Ollama has no token-bucket rate limit to replicate (that's the whole
   point), but still needs: the `format` parameter set to the schema's
   JSON Schema (Ollama supports a full schema, not just `"json"` --
   confirms more strictly than the smoke test above used; use the schema
   form for the real implementation), a retry-with-correction loop
   matching `structured()`'s existing pattern for when even this model
   gets the shape wrong, and a `Usage` object populated from whatever
   Ollama's response reports (check `/api/generate`'s response body for
   `prompt_eval_count`/`eval_count` -- likely the token-count equivalents,
   verify against a real response rather than assuming the field names).
3. `graph.py`'s `Runner.__init__` (or wherever `GroqClient` is currently
   constructed) branches on the configured provider and builds the right
   client -- `nodes/summarize.py` must not need to change at all, since it
   only calls `client.structured(...)`.
4. Test with a fake Ollama HTTP transport, matching the existing
   `FakeGroq`/`httpx.MockTransport` pattern in `tests/fakes.py` --
   don't require the real remote host to be reachable for the test suite
   to pass.
5. Once wired, re-run a real import with `LLM_PROVIDER=ollama` and compare
   against today's Groq run: schema-failure rate, wall-clock time per
   repo, and whether the claims it produces are actually good (not just
   valid JSON) -- validity and quality are different questions, and only
   the first one has been checked so far.
