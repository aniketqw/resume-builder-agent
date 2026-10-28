# TRACE — the engineering history of Resume Builder Agent

An account of how this project was built, commit by commit, with the reasoning
and the measurements behind each decision.

---

## How to read this document

This is not a changelog. A changelog records what changed; this records **why**,
**what else was considered**, and **what the evidence was**. It is written so
that three different readers get something out of it:

- someone evaluating the engineering can see the actual depth — what was tried,
  what was measured, what was thrown away and on what grounds;
- someone rebuilding the system can follow it as a construction order, because
  the sections are chronological and each one names the real files and the real
  constraints that forced the design;
- someone reading it once can still come away understanding the whole system,
  because the architecture is laid out before the commits start.

**Provenance convention.** This document was assembled after the fact by reading
the git history, the diffs, and the source. Every section therefore ends with
one of two markers:

- `> Rationale recorded in <source>.` — the reasoning was written down at the
  time, in the commit message body, a module docstring, `README.md`, or
  `evals/budget.md`. Those passages are quoted or closely paraphrased here.
- `> Reconstructed from the diff.` — no contemporaneous rationale exists for
  this point; the explanation is inference from the code and is labelled as
  such.

**Numbers.** Every figure in this document is traceable to a source in the
repository — a commit message, a docstring table, `evals/budget.json`,
`evals/budget.md`, or `README.md`. Where a number was not recorded, this
document says so rather than supplying one. That rule is not stylistic. The
system described here exists to stop a language model inventing a metric; a
trace that invents its own history would be the same failure one level up.

This history is unusually well documented at source. Of the 38 commits, 34 carry
a substantive message body explaining the reasoning, and several modules carry
module-level docstrings whose entire purpose is to record a rejected design so
nobody proposes it again. Those docstrings are the primary sources here.

---

## The governing constraint

From `README.md`:

> The governing rule: **the agent never invents an achievement or a number.** A
> claim that cannot be traced to repository evidence is not softened or hedged —
> it is dropped and replaced with a `[MEASURE: <command>]` task telling you how
> to go and measure it yourself.

Almost every architectural decision in this repository is downstream of that one
sentence. It is worth spelling out the consequences, because they are not
obvious and they are what makes the codebase shaped the way it is.

**1. Enforcement must be code, not a prompt.** "Only state supported facts" in a
system prompt is a request. `src/rba/nodes/verify.py` is the enforcement, and its
docstring says so in the first line: *"Claim verification. Code, not a prompt."*
Every numeric token in a generated claim must appear in the exact evidence text
the model was shown. If it does not, the claim is rejected. This is why the
evidence bundle is constructed in `build_evidence()` in
`src/rba/nodes/summarize.py` and preserved — the verifier needs byte-identical
access to what the model saw.

**2. The evidence set must be finite and model-free.** `src/rba/nodes/fetch.py`
opens with *"Deterministic fact gathering. No model touches anything in this
module."* If a model helped assemble the evidence, verifying against the
evidence would be circular.

**3. Every point of invention gets its own guard.** There are three places a
model can invent, and each has a separate check:

| Where | Guard | File |
|---|---|---|
| summarisation invents a fact | `verify_claim` | `src/rba/nodes/verify.py` |
| resume generation merges two claims into a new total | `enforce_provenance` | `src/rba/nodes/build.py` |
| skill extraction inflates a skill list | `source_verify` | `src/rba/ats/extract.py` |

The second is the subtle one. The generator only ever sees verified claims, so
it cannot reach the repositories — but it can still add 739 and 745 together.
`enforce_provenance` re-runs the number check at bullet level against the union
of the claims the bullet says it used.

**4. The rule applies to the project's own metrics.** This is where it bites
hardest and where most of the interesting work happened. An invented weight in a
scoring rubric is the same category of error as an invented achievement.
`src/rba/resume/diagnostics.py` says it plainly:

> The artifact this is modelled on shows seven bars that sum to 100 — keywords
> 44, role fit 18, numbers 12, sections 10, contact 6, length 6, verbs 4. Those
> weights were invented. Nobody measured that a missing phone number is worth
> exactly six points, and presenting a number built from invented weights as
> "your ATS score" is the same category of error as inventing an achievement.

That is why the project's own first scoring rubric (`src/rba/scoring.py`,
commit `01674c3`) was later superseded by the evidence-grounded `src/rba/ats/`
package (commit `bb3972f`) — the author applied his own rule to his own work and
the rubric failed it.

**5. Honest reporting of what is *not* guaranteed.** Commit `0c832f7` splits one
faithfulness number into two because averaging them hid which half was
guaranteed. Commit `b5b76b1` publishes run-to-run variance of up to 24 ATS
points rather than quietly keeping the better-looking run. Both are the rule
applied reflexively.

---

## Architecture overview

Two entry paths share one core.

### Path A — the CLI agent (`rba run <user>`)

A LangGraph state machine, wired in `src/rba/graph.py`, checkpointed to SQLite:

```
connect_github      MCP, read-only, hard-gated against write tools
list_repos          all repos for the account
select_repos        hard rules (free) then Jev scoring (fractions of a cent)
fetch_facts         deterministic: README, tree, languages, commits, tests, CI
safety_scan         flags .env / tokens / cookies committed in YOUR repos
summarize_repo      strict JSON, one repo per call, rate limited
verify_claims       pure Python; unsupported claims are dropped
score_roles         deterministic rubric
build_resumes       bullets drawn only from verified claims
compare             table built in Python; model writes only the rationale
plan_improvements   evidence-linked steps, ATS gain from scoring.py arithmetic
human_review        LangGraph interrupt()
report              Markdown + LaTeX + PDF
```

Two structural facts about that order, both recorded in the `src/rba/graph.py`
docstring:

- `fetch_facts` uses MCP tools; `summarize_repo`, `build_resumes` and
  `plan_improvements` use strict JSON. **Groq cannot combine tool use with
  structured outputs in one request**, so those are necessarily separate nodes.
  The node boundary is an API constraint, not a taste decision.
- The SQLite checkpointer is not optional. At 8,000 tokens per minute a full run
  stalls repeatedly, and `human_review` interrupts the graph outright.

### Path B — the Streamlit app (`app.py`)

The CLI path cannot run inside a web request: a measured full run took 1,433 s
including 642 s and 616 s of rate-limit backoff, and Streamlit Community Cloud
drops the websocket after roughly 10–15 minutes (`src/rba/resume/pipeline.py`
docstring). So `src/rba/resume/pipeline.py` runs the same nodes with a repo cap
small enough to finish inside the window, and the app's centre of gravity is the
*editable document* rather than the agent run:

```
source        upload a PDF  -> src/rba/resume/ingest.py   (deterministic, no LLM)
              or GitHub     -> src/rba/resume/pipeline.py (capped agent run)
merge         src/rba/resume/merge.py   generated lines arrive SWITCHED OFF
target        a job description, or a role archetype from src/rba/roles.yaml
score         src/rba/ats/                three channels, convex fusion
attribute     src/rba/resume/attribution.py  exact leave-one-out per bullet
advise        src/rba/resume/diagnostics.py  advice with no invented weights
render        src/rba/resume/render/      plain | html | pdf | latex
```

### The scoring core (`src/rba/ats/`)

Three independent channels, fused by convex combination:

| Channel | Mechanism | Module | Weight |
|---|---|---|---:|
| dense | BAAI/bge-small-en-v1.5 via ONNX Runtime, cosine | `dense.py` | 0.5 |
| lexical | BM25 against a 22-job background corpus | `lexical.py`, `corpus.py` | 0.2 |
| skills | weighted Jaccard over canonicalised skill ids | `skills.py`, `normalise.py` | 0.3 |

No LLM runs in the scoring path. Extraction (`extract.py`) is upstream and is
confined to parsing. The number can be recomputed by hand, which is the point —
and it is what makes exact leave-one-out attribution affordable at all.

### The document model (`src/rba/resume/document.py`)

`Bullet` / `Entry` / `Table` / `KeyValues` / `Header` / `Section` / `ResumeDoc`.
Three properties the docstring calls out as forced by the toggle requirement:

- **Stable, content-derived ids** (`_bullet_id` = `sha256(source + "\x1f" + text)[:12]`),
  so a bullet regenerated from the same evidence keeps its identity across runs.
- **Evidence travels with the bullet** — dropping provenance at the document
  boundary would make the resume unfalsifiable.
- **`enabled` is a property of the bullet, not a separate selection list** — one
  source of truth, so `as_text()` always reflects what the user sees.

`Bullet.origin` is `yours` | `repo` | `assumed`, which drives provenance
colouring in the UI.

---

# The commits

---

## 1e849d7 — chore: gitignore before anything else (.env must never be tracked)   2026-10-28

**Problem** A secret committed once is in the history forever. The window in
which that can happen is between `git init` and the first `git add`.

**Options considered** Add `.gitignore` alongside the first real code commit
(the normal practice); rely on `git rm --cached` afterwards.

**Decision** `.gitignore` containing `.env` is the **first commit in the
repository**, before any other file was ever staged. 27 lines, nothing else.
This is a one-line statement of the project's posture: the safety property is
established before there is anything to protect.

**Files** `.gitignore`

**Measured** None recorded — this commit predates any measurement.

**Caught** Pre-empts rather than catches. `README.md` later cites this as a
safety guarantee: *".gitignore containing .env is the **first commit** in this
repository's history, before any other file was ever staged."* `beff839` and
`401b381` are the enforcement that backs it up.

> Rationale recorded in the commit subject and `README.md` ("Safety"). The
> commit has no body.

---

## 6a6cf6c — M0: skeleton, config, limiter, groq+jev clients, docker, CI   2026-10-28

**Problem** Build the foundation on facts about the live APIs rather than on
documentation. Specifically: which models exist on the free tier, what the real
rate limits are, and which API combinations are illegal.

**Options considered** Read the pricing and model pages and encode what they
say. Hardcode model IDs in the client.

**Decision** Probe the live APIs and encode only what came back. The commit
message lists five findings, each of which became a structural decision:

- **Groq's free tier exposes only `gpt-oss-20b` / `gpt-oss-120b` / `safeguard`
  and `qwen3.8-27b`**; `llama-3.1-8b-instant` and `llama-3.3-70b-versatile` are
  gone for this tier. Consequence: model IDs live in `config.yaml`, never in
  code, with a header comment recording the verification date
  (`Verified against GET https://api.groq.com/openai/v1/models on 2026-10-28`),
  and `rba doctor` fails loudly if a configured model has disappeared.
- **8K TPM is the binding constraint, not 30 RPM.** Buckets are therefore
  per-model in `src/rba/limiter.py` — which means routing a step to a different
  tier also buys throughput, a second-order benefit of tier routing that is easy
  to miss.
- **The reasoning trap.** `openai/gpt-oss-*` models spend the completion budget
  on an internal `reasoning` field. With a small `max_tokens` they return
  `content: ""` and **HTTP 200** — a success that looks like a model failure.
  Guarded by forcing `reasoning_effort: "low"`, a floor on `max_tokens`, a
  dedicated `EmptyContentError`, and a regression test.
- **Structured outputs cannot coexist with tool use.** This is the constraint
  that dictates the graph's node boundaries (see `graph.py` above).
- **Jev `choice` questions require a `criteria` map**; a bare `options` list
  returns HTTP 422.

The limiter design is worth stating because it recurs: `reserve()` estimates
tokens and waits for window room → HTTP call → `reconcile()` replaces the
estimate with `usage.total_tokens` and overwrites the window from
`x-ratelimit-remaining-*`. **Headers are treated as more authoritative than
local accounting.** That decision is correct and also directly causes the bug
fixed in `fdbefc4`.

Jev (TypeSafe System One) is introduced as a second provider for a specific
economic reason, recorded in `src/rba/jev_client.py`: its *output tokens are
free* and input is $0.042/Mtok, and it draws on a **separate quota** — so cheap
routing decisions do not consume the scarce 8K Groq TPM budget.

**Files** `.dockerignore`, `.env.sample`, `.github/workflows/ci.yml`,
`Dockerfile`, `README.md`, `config.yaml`, `docker-compose.yml`,
`pyproject.toml`, `scripts/scan_secrets.sh`, `src/rba/{__init__,cache,cli,config,groq_client,jev_client,limiter,schema}.py`,
`tests/{conftest,fakes,test_groq_client,test_jev_client,test_limiter,test_schema}.py`
— 25 files, 2,052 insertions.

**Measured** Free-tier limits confirmed from `x-ratelimit-*` response headers:
`rpm: 30, rpd: 1000, tpm: 8000, tpd: 200000` default;
`openai/gpt-oss-safeguard-20b` at `rpm: 3, tpm: 2000`. `config.yaml` also
records a later empirical correction in a comment: *"at 1200 a live run hit
'max completion tokens reached' and lost a whole repo"*, hence
`max_tokens: 2500`.

**Caught** The reasoning trap and the structured-output/tool-use exclusion were
both found by probing, before they could cost a debugging session mid-pipeline.

`src/rba/schema.py` also lands here, solving a strict-mode problem that fails at
request time rather than author time: Groq strict mode requires *every* property
in `required` and `additionalProperties: false` on every object, and Pydantic's
`model_json_schema()` satisfies neither by default. `to_strict_schema()` fixes
both and inlines `$defs`; `lint_strict_schema()` exists so a unit test can keep
it honest. This module is where the bug in `fe93e24` will later hide.

> Rationale recorded in the commit message body and the `src/rba/limiter.py`,
> `src/rba/groq_client.py`, `src/rba/schema.py` and `src/rba/jev_client.py`
> module docstrings.

---

## 9a65c4e — M1: evidence-backed repo selection via Jev   2026-10-28

**Problem** Summarising every repository is unaffordable. Selection has to
happen before any Groq token is spent. The first attempt asked Jev to judge
repos from metadata alone — name, size, language — and it did not work.

**Options considered** (a) Score from metadata and accept the noise.
(b) Use a bigger/better model on the same metadata. (c) Give the model more
evidence.

**Decision** (c), and the reason is the most instructive point in the commit:
the metadata result was not a model failure. From the commit message:

> scoring 93 repos on name/size/language returned 0.13-0.35 for everything,
> which is the right answer to a question containing no evidence.

A flat band of uncertainty is the *correct* output when the input contains
nothing to discriminate on. Reaching for a stronger model would have been
treating a data problem as a capability problem. So `src/rba/signals.py` was
added: a cheap deterministic prefetch of README head, file-tree shape, commit
history, and test/CI detection — three REST calls per repo, zero tokens, and the
results are reused by the full fact fetch later.

`src/rba/signals.py` states the same thing: *"a 500 MB course-notes dump looks
exactly like a 500 MB real project. Asking a model to guess from that produces
uniform uncertainty, which is the correct answer to a question with no evidence
in it."*

**Files** `pyproject.toml`, `src/rba/github_tools.py` (new, 343 lines),
`src/rba/nodes/select.py`, `src/rba/signals.py`, `src/rba/state.py`,
`tests/test_select.py`, `tests/test_signals.py`

**Measured**, from the commit message:

| | score range over the same 93 repos |
|---|---|
| metadata only | 0.13 – 0.35 (no separation) |
| with signals attached | 0.09 – 0.81 (separates cleanly) |

Live run: **17.8 s, 245 GitHub REST calls, 74k Jev input tokens, $0.0031, and
zero Groq tokens spent.**

**Caught** No bug fixed; a design dead-end closed with a measurement rather than
an opinion.

> Rationale recorded in the commit message body and the `src/rba/signals.py` and
> `src/rba/nodes/select.py` module docstrings.

---

## 789a108 — M2: GitHub MCP backend, fact fetching, safety scan   2026-10-28

**Problem** Reach GitHub through the official MCP server without ever holding a
write capability, and gather facts across ~100 repositories without the session
dying.

**Options considered** The plain hosted MCP endpoint; the local Docker
container; REST only. For trees: walk with `get_file_contents`, or fall back to
REST.

**Decision** Four findings from enumerating the live server, each of which
changed the implementation:

1. **Endpoint choice is a security decision, settled by enumeration.**

   | Endpoint | Tools | Write tools |
   |---|---:|---|
   | `/mcp/` | 46 | `create_or_update_file`, `delete_file`, `push_files`, `create_repository`, `fork_repository`, `merge_pull_request` |
   | `/mcp/x/repos/readonly` | 13 | none |

   The read-only repos toolset is the default, and **the backend refuses to
   start if a write tool appears in the toolset**, regardless of what asked for
   it. `mode: docker` runs the container with `--read-only`, which is also a
   hard gate. This is defence in depth: the classic PAT `repo` scope grants
   write with no read-only variant, so the toolset gate is the only thing
   standing between a legitimate token and a destructive tool.

2. **There is no `list_repositories`.** Listing uses `search_repositories` with
   a `user:` qualifier, which **excludes forks** — hence MCP returns 98 where
   REST returns 113 — and does return all 41 private repos. Forks are excluded
   by rule anyway, so the behaviour is acceptable; documenting it prevents a
   future "where did 15 repos go" investigation.

3. **No `get_repository_tree` on the hosted endpoint** — it lives in the `git`
   toolset, which only the local container exposes. Walking a tree with
   `get_file_contents` costs **one call per directory: ~500 calls across 98
   repos, after which the endpoint returned empty trees and the session timed
   out.** Trees now come from a single REST call
   (`github.mcp.tree_via_rest`, on by default). Note the failure mode: not an
   error, but *empty trees* — a silent degradation that would have produced
   thin, honest-looking fact bundles.

4. **Sessions die partway through a long sweep**, after which every later call
   fails with the same error. The client now detects session errors and
   reconnects.

**Files** `config.yaml`, `src/rba/github_tools.py` (+345/−90),
`src/rba/nodes/fetch.py`, `src/rba/nodes/safety.py`, `src/rba/nodes/select.py`,
`tests/{test_fetch,test_github_tools,test_safety,test_select}.py`

**Measured** Live run: **25 repos selected, 24 with READMEs, 10 with tests,
223 s, 233 calls.** One real security finding: a committed `.env.local` in
`pehenava-backend`.

**Caught** A live run **rejected all 98 repos**. Emptiness was being judged from
`size_kb`, which the MCP `search_repositories` response omits — so every repo
read as size 0 and therefore empty. Fixed by deciding emptiness from file and
commit counts instead, and the split is now architectural:
`apply_hard_rules()` uses only metadata it can trust, and `drop_empty()` runs
*after* signals are gathered. The docstring of `apply_hard_rules` records the
rule: *"Emptiness is deliberately not judged here… Emptiness is a fact about
files and commits."*

`src/rba/nodes/safety.py` also lands here. It scans the *user's* repositories,
not this project, in two parts: path rules for files that should never be
committed, content rules for credentials pasted into otherwise-fine files. Only
bounded content already fetched is scanned — no repo is downloaded to search it
— and `_redact()` shows only enough of a match to locate it, never enough to use
it.

> Rationale recorded in the commit message body, `config.yaml` comments, and the
> `src/rba/github_tools.py` and `src/rba/nodes/safety.py` docstrings.

---

## 01674c3 — M3/M4: evidence-linked summarization, claim verification, ATS rubric   2026-10-28

**Problem** This is the heart of the project: turn a fact bundle into claims,
and then *prove* each claim or kill it. The first live run made clear that
naive versions of both the technology check and the number check were wrong in
ways unit tests had not caught.

**Options considered** For technology verification: scan claim prose for
"technology-shaped" tokens (the obvious approach), or check against a declared
field. For version masking: a regex, or procedural identification from context.

**Decision** Four fixes, all found by judging the first live run against the
real repositories. The pattern across all four is the same: **the naive
implementation was wrong in the direction of destroying good output**, which is
the failure mode a test suite built on synthetic fixtures will not surface.

- **Technology check.** Scanning claim prose for technology-shaped tokens
  **rejected 39 of 48 real claims** over words like `fallback.` and `scripts.` —
  a trailing full stop made them look like package names. Technologies are now
  checked against the summary's **declared `tech` list**, where the model has
  explicitly said "this is a technology name", so there is no guessing. And an
  unsupported entry is *dropped from the list* rather than invalidating the
  whole claim that happened to mention it. That second half matters: the
  granularity of the penalty should match the granularity of the evidence.
- **Impact regex** matched bare nouns, so *"MongoDB is used for **saving** and
  searching documents"* was flagged as an unmeasured performance claim. It now
  matches only comparatives and named metrics.
- **Version masking** used `\d+\.\d+`, which swallowed `0.81 accuracy` and
  `49.8%` — **precisely the numbers the module exists to police.** Versions are
  now identified procedurally from context (`_is_version`). This is the sharpest
  example in the history of a guard quietly disabling itself: a verifier that
  masks the metrics is a verifier that passes everything.
- **A summary exceeding `max_tokens` came back as a 400, not a truncated 200.**
  Those are now retried with a larger budget.

Prompt `summarize.v2.md` adds two behavioural instructions: report unfavourable
benchmark results as-is, and stop padding the `tech` list from transitive
dependencies.

`src/rba/scoring.py` lands here as a deterministic 100-point rubric (keywords
44, role_fit 18, numbers 12, sections 10, contact 6, length 6, verbs 4). Its
docstring gives the justification: *"The improvement planner quotes an 'ATS
gain' for every suggested step, and that number is only honest if the same
resume always scores the same."* Two sub-decisions inside it are worth noting
because they are the governing rule applied to a scoring function:

- `score_keywords` **saturates**: *"a resume that says 'agent' twenty times is
  not twice as relevant as one that says it twice, and rewarding repetition
  would push the generator toward keyword stuffing."*
- `score_numbers` is **capped at a realistic target**, not at "every bullet":
  *"Demanding a number in every bullet would be a licence to invent them."* And
  a `[MEASURE: ...]` placeholder scores nothing, so an unmeasured claim is never
  rewarded.

This rubric is later superseded (see `bb3972f`) — but on the grounds that its
*weights* were invented, not that its determinism was wrong. The determinism
survives into `diagnostics.py`.

**Files** `config.yaml`, `src/rba/groq_client.py`, `src/rba/nodes/summarize.py`,
`src/rba/nodes/verify.py`, `src/rba/prompts/{__init__,summarize.v1.md,summarize.v2.md}`,
`src/rba/roles.yaml`, `src/rba/scoring.py`, `src/rba/state.py`,
`tests/test_scoring.py`, `tests/test_verify.py`

**Measured** 39 of 48 real claims wrongly rejected by the prose-scanning
technology check. The specific failing tokens (`fallback.`, `scripts.`) and the
specific swallowed numbers (`0.81 accuracy`, `49.8%`) are recorded.

**Caught** All four above, plus the structural insight that the evidence bundle
handed to the model must be the *same text* the verifier checks against —
`build_evidence()` is documented as *"A compact JSON bundle. This exact text is
what the verifier checks against."*

> Rationale recorded in the commit message body and the `src/rba/nodes/verify.py`,
> `src/rba/nodes/summarize.py` and `src/rba/scoring.py` module docstrings.

---

## 401b381 — fix: build credential-shaped test fixtures at runtime   2026-10-28

**Problem** `scripts/scan_secrets.sh` flagged a realistic-looking GitHub token
literal in `tests/test_safety.py`. It was never a real credential.

**Options considered** Add an exception for the test file to the scanner. Use an
obviously-fake string that the scanner would not match.

**Decision** Neither. From the commit message:

> It was never a real credential, but a scanner that can tell the difference is
> a scanner with a hole in it.

Both alternatives weaken the scanner to accommodate the test. `fake_secret()` in
`tests/fakes.py` **constructs** credential-shaped strings at runtime, so no
credential-shaped literal exists in source at all. The scanner stays maximally
strict and the test still exercises the detector.

The second alternative deserves a note: using a string the scanner would not
match means the safety test no longer tests the scanner against anything the
scanner would catch, which is worse than useless.

**Files** `tests/fakes.py`, `tests/test_safety.py`

**Measured** None recorded.

**Caught** A latent weakening of the project's own secret scanner.

> Rationale recorded in the commit message body.

---

## beff839 — fix: secret scanner silently skipped the git history check   2026-10-28

**Problem** `scripts/scan_secrets.sh` reported "no commits yet" on a repository
with six commits. The history half of the scan had never run.

**Decision** The cause is a classic shell pipeline trap, recorded exactly:

> `git rev-list --all | head -1 | grep -q .` exits non-zero: head closes the
> pipe, rev-list dies on SIGPIPE, and the pipeline fails.

`head -1` closes the read end as soon as it has its line; `git rev-list` then
gets SIGPIPE and dies; the pipeline's exit status reflects that, and the guard
concluded there was no history to scan. Replaced with `git rev-list --count`,
which produces a number and exits cleanly.

**Files** `scripts/scan_secrets.sh` (4 insertions, 1 deletion)

**Measured** No count recorded beyond "a repo with six commits".

**Caught** A security control that had been silently inert since `6a6cf6c`.
Note the verification method, which is the right one for a detector:

> Verified by planting a credential-shaped string and confirming a non-zero
> exit.

A green scanner proves nothing unless you have also seen it go red. This is the
same discipline as `401b381`: the test must exercise the thing being tested.

> Rationale recorded in the commit message body.

---

## 6c6ecc9 — verify: a quote must support the claim, not merely exist   2026-10-28

**Problem** A live run produced the claim *"uses the tcell library for terminal
UI"*, attributed to the quote `github.com/goccy/go-json`. The quote is a real
line from that repository's `go.mod`. The containment check passed. The quote
supports nothing in the claim.

**Options considered** Require exact quote containment only (the status quo —
insufficient). Ask a model to judge relevance (reintroduces a model into the
enforcement layer, which the architecture forbids). Require shared content
words.

**Decision** `quote_relevant()` requires shared content words between the claim
and its quote. The docstring explains why this threshold rather than something
stricter:

> Requiring shared content words catches that without punishing honest
> paraphrase, since any real citation repeats at least one distinctive term.

The design pressure here is two-sided: too strict and honest paraphrase is
rejected (a false drop, which `fe93e24` will show is the costlier failure); too
loose and a real-but-irrelevant quote passes. Shared distinctive terms is the
weakest condition that still catches the observed failure.

`quote_supported()` already had the complementary accommodation: exact
containment first, then a token-overlap fallback, *"models reformat whitespace,
so a token-overlap fallback avoids rejecting an otherwise honest quote over a
line break."*

**Files** `src/rba/nodes/verify.py`, `src/rba/prompts/resume.v1.md` (new),
`tests/test_verify.py`

**Measured** Live result on 8 repositories: **48 of 51 claims verified, 3
rejections, all three true positives.** The best of the three, quoted from the
commit message:

> the model wrote 'losing 745 out of 1484', having computed 1484-739 itself. The
> evidence contains 739 and 1484 but never 745, so it was refused.

That single example is the whole thesis of the project in one line. The model
did arithmetic — reasonable, probably correct arithmetic — and produced a number
that exists nowhere in the evidence. No prompt instruction would have prevented
it; only a mechanical check over the evidence set does.

**Caught** The quote-relevance hole.

> Rationale recorded in the commit message body and the `quote_relevant`
> docstring in `src/rba/nodes/verify.py`.

---

## 9362620 — M4/M5: resumes, comparison, planner, report, graph, CLI   2026-10-28

**Problem** Assemble verified claims into role-targeted resumes, compare them,
plan improvements, and wire the whole thing into a resumable graph.

**Options considered** Let the model write the comparison numbers and the
claimed ATS gains (the straightforward approach); trust verification at the
claim level and not re-check the generated bullets.

**Decision** Five decisions, all variations on "the model proposes, Python
disposes":

- **`enforce_provenance` re-checks every generated bullet's numbers against the
  claims it says it used.** The reasoning, from the commit message: *"The
  generator is a second opportunity to invent: it can merge two claims into a
  total that appeared in neither."* Claim-level verification is necessary but
  not sufficient; composition is itself a point of invention.
- **The comparison table and every ATS gain come from `src/rba/scoring.py`, not
  from a model**, *"so a plan step and a score movement cannot disagree."*
  `attach_ats_gain` distributes the rubric's own `max_gain()` across the steps,
  which means the planner structurally cannot promise points the scorer would
  not award.
- **Plan steps naming a repository the candidate does not have are rewritten as
  new work** rather than presented as building on something imaginary. A
  hallucinated repo in an improvement plan is the same error class as a
  hallucinated metric.
- **SQLite checkpointer plus `interrupt()` for human review.** *"at 8K TPM a run
  stalls often enough that durable state is not optional."*
- **The report leads with security findings and lists refused claims
  explicitly.** `src/rba/report.py`: *"that refusal list is the most informative
  part."* Filtering is made visible rather than hidden — a system that silently
  drops claims is indistinguishable from one that had nothing to drop.

**Files** `config.yaml`, `src/rba/cli.py`, `src/rba/graph.py`,
`src/rba/nodes/build.py`, `src/rba/nodes/plan.py`, `src/rba/prompts/plan.v1.md`,
`src/rba/report.py`, `src/rba/state.py`, `tests/test_build.py`,
`tests/test_plan.py`, `tests/test_verify.py`

**Measured** Dry run on `aniketqw`: **32 calls, 67,500 projected tokens, ~8.4
minutes.** The `--dry-run` mode exists precisely so this projection can be
obtained without spending.

**Caught** No bug fixed; the commit *adds* the second-chance-to-invent guard
before it was needed, and `73256dc` will shortly verify it works.

> Rationale recorded in the commit message body and the `src/rba/nodes/build.py`,
> `src/rba/nodes/plan.py`, `src/rba/report.py` and `src/rba/graph.py` docstrings.

---

## 9008f8f — M6: eval harness (faithfulness + G-Eval), LangSmith tracing   2026-10-28

**Problem** "The summaries look good" is not a measurement. Quality needs to be
assessed with published metrics rather than vibes — and the assessment itself
must not be naively trusted, because LLM judges have known, documented biases.

**Options considered** A single holistic 1–5 quality rating from a judge model
(the common approach). One combined score. Same model for generation and
judging.

**Decision** Two metrics of **deliberately different kinds**, and the difference
is the design:

- **Faithfulness (RAGAS-style), blocking.** Decompose a summary into atomic
  claims; issue a **binary** supported/unsupported verdict per claim against the
  evidence; score = supported / total. Binary rather than holistic *"because LLM
  judges are measurably biased on 1-5 scales and far steadier on yes/no calls"*.
  Target is **1.0, not "high"** — the pipeline's whole promise is that nothing
  unsupported survives, so anything below 1.0 is a defect, not a grade.
- **G-Eval over the four SummEval dimensions** (coherence, consistency, fluency,
  relevance), scored 1–5 and **probability-weighted** rather than taken as a raw
  integer, *"raw integer ratings cluster and lose resolution, which is the
  problem G-Eval was designed around"*. Tracked, not enforced.

Three concrete mitigations for known judge bias, from `evals/judge.py`:
dimensions scored in **separate calls** (position effects), samples averaged
over **n ≥ 3** (variance), and the judge runs on a **different model tier than
the generator** (self-preference bias). `config.yaml` routes `judge: big` while
`summarize_repo: small`.

**Files** `evals/judge.py`, `evals/run_eval.py`, `pyproject.toml`,
`src/rba/cli.py`, `src/rba/config.py`, `src/rba/graph.py`,
`src/rba/groq_client.py`, `tests/conftest.py`, `tests/test_report.py`

**Measured** No scores reported in this commit; the harness is the deliverable.
`evals/run_eval.py` does record the cost shape: *"G-Eval costs 4 dimensions × n
samples per repo. At 8,000 tokens per minute that is the most expensive thing in
this project, which is why `--no-geval` exists and why the golden set is small."*

**Caught** A real async bug: *"the graph used `SqliteSaver`, which raises
`NotImplementedError` the moment `ainvoke` asks it for a checkpoint. Every node
does network I/O so the graph is async throughout; it needs `AsyncSqliteSaver`
and `aiosqlite`."* Note the failure timing — the sync saver works until the
first checkpoint, i.e. it fails exactly when it is needed. This is the same
shape as `73256dc`.

Tracing is explicitly scoped as a diagnostic: a failing LangSmith call is
swallowed (`enable_tracing` docstring: *"a misconfigured LangSmith must never
stop a run"*), and the test suite disables it so CI needs no network.

> Rationale recorded in the commit message body and the `evals/judge.py`,
> `evals/run_eval.py` docstrings.

---

## 73256dc — fix: RepoSignals must be serializable to survive a checkpoint   2026-10-28

**Problem** `RepoSignals` was a plain `__slots__` object. It lands in LangGraph
state, and the checkpointer serialises state with msgpack, so persisting raised
`Type is not msgpack serializable`.

**Options considered** Register a custom msgpack encoder. Keep signals out of
graph state and recompute them. Convert to a Pydantic model.

**Decision** Pydantic model. The timing argument is what makes this urgent
rather than cosmetic, from the commit message:

> raises 'Type is not msgpack serializable' the moment a run tries to persist --
> which is exactly when the checkpoint matters, since an 8K TPM run stalls
> constantly.

A serialisation bug in a checkpoint path is invisible in every fast run and
fatal in every slow one. Since every real run is slow here, it was fatal.

The class docstring preserves the reasoning so the type is not "simplified" back
later: *"A Pydantic model rather than a plain class because this lands in
LangGraph state, and the checkpointer serialises state with msgpack."*

**Files** `src/rba/nodes/fetch.py`, `src/rba/signals.py`, `tests/test_graph.py`
(new, 172 lines), `tests/test_select.py`, `tests/test_signals.py`

**Measured** None recorded.

**Caught** The serialisation bug, plus — more interestingly — the new offline
end-to-end graph test **confirms the provenance guard fires**: the fake
generator emits *"handling 9,999 requests per second"* from claims containing no
such number, and that bullet is dropped before it reaches the resume. This is
the `9362620` guard being demonstrated rather than assumed, with an adversarial
fake standing in for a model's worst behaviour. A test suite for this project
has to include a *lying generator*, or it only tests the happy path.

> Rationale recorded in the commit message body and the `RepoSignals` docstring
> in `src/rba/signals.py`.

---

## ceff6b8 — docs: README with measured numbers and what judging changed   2026-10-28

**Problem** The README described intent; the live runs had produced facts.

**Decision** Rewrite the README around measured numbers, and — unusually — add a
section titled "What judging actually changed" that lists the project's own
failures: the 39-of-48 rejection, the bare-noun impact regex, the version-mask
swallowing `0.81 accuracy`, and the `tcell`/`go-json` quote.

The editorial choice is the point. The natural README lists capabilities. This
one leads the quality section with the four ways the first implementation was
wrong, then states what the fixes achieved. The project's rule — never present
an unearned number — applied to its own documentation means the bugs are part of
the evidence that the numbers are real.

**Files** `README.md` (+81/−8), `tests/fakes.py`, `tests/test_verify.py`

**Measured** Consolidates the figures from `789a108`, `6c6ecc9` and `9362620`
into the README's free-tier table (98 repos, 25 selected, 151 claims,
139 verified, etc.).

**Caught** Nothing; the two test files change alongside for consistency with the
documented behaviour.

> Reconstructed from the diff. This commit has no message body; the editorial
> rationale above is inference from what the README section contains.

---

## fe93e24 — fix: two bugs that silently destroyed good output   2026-10-28

This is the most instructive commit in the repository. Both bugs produced output
that **looked fine** — valid resumes, no errors, no warnings — while being
badly wrong.

### Bug 1 — `_strictify` ate the model's own field names

**Problem** The improvement planner burned **nine minutes on retries and then
gave up**. Every generated plan step failed validation.

**Cause** `src/rba/schema.py`'s `_strictify` walks a JSON Schema stripping
`title`, `default` and `examples` as noisy JSON-Schema annotations — which is
correct, because Groq's strict mode rejects unknown keywords. But it stripped
them **everywhere**, including from inside a `properties` map, where `title` is
not an annotation but the *name of a field the model is supposed to emit*. The
plan schema therefore lost its `title` property, so every step the model
returned carried a field the schema did not declare, and strict validation
rejected it.

**Decision** `_strictify` is now **structure-aware**: annotations are stripped
from schema nodes, never from inside a `properties` map. The distinction is
positional — the same key means different things at different depths of a JSON
Schema — which is exactly the kind of thing a flat recursive key-deletion misses.

**Measured** *"One plan call now takes 3 seconds"*, against nine minutes of
failed retries.

### Bug 2 — citations parsed as invented numbers

**Problem** `enforce_provenance` **dropped 19 legitimate bullets in one live run
and 8 more in the next.**

**Cause** The generator appends claim-id citations to bullets:
`(AiAgnoSample#0, #1)` — and, unprompted, the full-width `【AOA_Controller_ty#0】`.
`enforce_provenance` extracts numeric tokens from a bullet and demands each one
appear in the source claims. `#0` parsed as the number **0**. Zero does not
appear in the claims, so the bullet was rejected as containing an invented
number.

This is the worst possible failure mode for this architecture: **the safety
mechanism itself was destroying correct output**, and the visible symptom was a
short resume rather than an error. A shorter resume looks like the verifier
working hard.

**Decision** Citations are stripped before the number check **and from the
output**, *"where an internal id never belonged anyway"*. Note the second half —
the fix removes the citations from the rendered bullet too, so the bug's
underlying cause (an internal identifier leaking into user-facing text) is
eliminated rather than merely tolerated.

**Measured** *"all five resumes now build with zero false drops, and the
balanced resume went from 32.9 to 61.2/100."*

**Files** `src/rba/nodes/build.py`, `src/rba/nodes/plan.py`,
`src/rba/prompts/resume.v1.md → resume.v2.md`, `src/rba/schema.py`,
`tests/test_build.py`, `tests/test_schema.py`

**Caught** Both bugs. The common lesson, worth stating because it recurs in
`fdbefc4` and `667895e`: **these were found by running the thing, not by
reasoning about it.** Neither produced an exception. One produced a slow
planner; one produced a shorter resume. Both are states a reviewer would read as
normal.

> Rationale recorded in the commit message body and restated in `README.md`
> ("Three later bugs were each caught by running the thing rather than reasoning
> about it").

---

## fdbefc4 — fix: the rate-limit header override never expired   2026-10-28

**Problem** *"A three-second judge call became one that never finished."*

**Cause** Commit `6a6cf6c` established the principle that response headers are
more authoritative than local accounting. That principle is right, but
incomplete: headers are authoritative **only for as long as they describe**.
`x-ratelimit-remaining-tokens` is a snapshot of the *current minute*, and the
bucket refills after `x-ratelimit-reset-tokens`. The limiter cached the value
with no expiry. So once any single response reported a low remaining figure, the
window stayed pinned at that figure **for the life of the process** — every
later call concluded it had no budget and slept a full minute first.

The `_Window` docstring now states the general principle:

> an override that never expires is worse than no override at all, because it
> pins the window at whatever the last response reported.

**Options considered** Drop the header override and rely on local accounting
(loses accuracy). Refresh the override on every call (does not help — the
problem is the stale value, and calls are what is being blocked). Give the
override an expiry.

**Decision** The override carries an expiry taken from the reset header and
**capped at the 60-second window it describes**, after which local accounting
resumes. The cap matters: a server could report a long reset, and trusting it
unboundedly recreates the bug in slow motion.

A second fix in the same commit: **judge call budgets are sized to their
payloads.** A two-field verdict was reserving the 2,500-token default meant for
summarization — so the faithfulness judge, which issues many tiny binary
verdicts, was reserving roughly 30% of the per-minute budget per verdict. The
two bugs compound: oversized reservations drain the window, the drained window
is captured by a header, and the un-expiring override freezes it there.

**Files** `evals/judge.py`, `src/rba/limiter.py`, `tests/test_limiter.py`

**Measured** The before/after is qualitative in the record: "a three-second
judge call became one that never finished", and the fix restores it. No
post-fix timing figure was recorded.

**Caught** A self-reinforcing livelock between two independently reasonable
decisions. Neither the header-authority rule nor a default token budget is wrong
alone.

> Rationale recorded in the commit message body and the `_Window` docstring in
> `src/rba/limiter.py`.

---

## 0c832f7 — evals: measure claim faithfulness separately from prose   2026-10-28

**Problem** The harness produced a single whole-summary faithfulness number.
That number averages two things with very different guarantees.

**Decision** Report them separately. The reasoning, from the commit message:

> The claims are what reach a resume and what verify_claims governs; the
> descriptive problem/approach fields are framing and are not verified.
> Averaging them into one number hides which half is actually guaranteed.

**Measured**, on the golden set:

| Metric | Result | Blocking? |
|---|---:|---|
| claim faithfulness — the text that reaches a resume | **1.00** | yes |
| number precision — numbers traceable to evidence | **1.00** | yes |
| whole-summary faithfulness — includes unverified prose | 0.83 | tracked |

The 0.17 gap is entirely unverified prose, and `evals/judge.py` names the
specific kind of thing that lives in it:

> "the original model does not account for distance between data points" is a
> fair reading of a repository's purpose and still nowhere in its README.

**The honest move here is reporting 0.83 at all.** A single averaged figure
would have been ~0.9 and unobjectionable. Splitting it produces one number that
is perfect and one that is visibly imperfect, and makes the imperfect one the
headline of a paragraph explaining exactly what is not guaranteed. `README.md`:
*"That gap is the honest part."*

`number_precision` is also notable as a design: a **model-free** proxy, checking
that every number in a summary occurs in the evidence. Its docstring says *"It
should be 1.0 by construction after verify_claims; if it is not, something
upstream regressed."* It is a cheap canary for the expensive guarantee, and it
does not depend on a judge being honest.

**Files** `README.md`, `evals/judge.py`, `evals/results/latest.json`,
`evals/run_eval.py`

**Caught** No bug; a measurement-design flaw that would have concealed future
regressions in the blocking half.

> Rationale recorded in the commit message body, `evals/judge.py` docstrings,
> and `README.md`.

---

## 98e0663 — fix: langchain[mcp] is a base dependency, not an extra   2026-10-28

**Problem** MCP is the *default* GitHub backend, but `langchain[mcp]` was
declared as an extra. A clean clone installed with `[dev]` alone failed two
tests on a missing module.

**Decision** Promote it to a base dependency. The principle: **a default path
cannot depend on an optional install.** If the config ships with `mode: remote`
and remote means MCP, then MCP is not optional.

**Files** `evals/results/latest.json`, `pyproject.toml`,
`tests/test_github_tools.py`

**Measured** None recorded (two failing tests).

**Caught** A packaging error that only a fresh environment surfaces. The
verification method is the notable part:

> Verified by cloning to a fresh directory and running the README's install path
> end to end.

Not "the tests pass on my machine" — the README's *documented* install path, in
an environment that has never seen the project. That is the only way to catch an
ambient dependency.

> Rationale recorded in the commit message body.

---

## b570e64 — Publish to a Hugging Face Space from GitHub Actions   2026-10-28

**Problem** Make a run of the agent publicly inspectable, without exposing
credentials or private repository contents.

**Options considered** A live Gradio or Docker Space (the obvious choice for an
agent demo); a static Space publishing a recorded run.

**Decision** A **static** Space, and the decision was settled by probe rather
than by discovery-at-the-end:

> A creation probe settled that in under a minute rather than at the end of the
> work: the Hub returns 402 with 'hosting Gradio and Docker Spaces on free
> cpu-basic requires a PRO subscription', and both free ZeroGPU slots are
> already taken.

Probing a platform constraint before building against it is a repeated habit in
this history (see also `6a6cf6c`, `789a108`, `4f4f6c1`). The cost of the probe
was a minute; the cost of discovering it after building a Gradio app would have
been the app.

Two deliberate divergences from the deployment this was modelled on
(`lecturelens-research`), both security decisions:

- **No `add_space_secret`.** The other deployment syncs a server-side
  `GROQ_API_KEY`; this one must not. *"The agent needs a GitHub token to read
  repositories and this account has private ones, so a server credential would
  expose them."* The reasoning chain is worth following: a server-side key makes
  the Space able to run the agent; running the agent requires a GitHub token;
  that token reads private repos; therefore a server credential on a public
  Space is a private-repo exposure. The fix is to not have the capability.
- **The published sample is regenerated from PUBLIC repositories only.** The
  normal report draws on 25 repos, **8 of them private**, and quotes their
  READMEs **29 times**. `scripts/check_sample.py` gates the build with a
  positive allowlist check *plus* a live cross-check against the 42 private
  repos. Positive allowlist plus negative cross-check — neither alone is
  sufficient, because an allowlist can be stale and a blocklist can be
  incomplete.

The deploy pipeline itself: allowlist bundler, sha256 manifest with selective
add/delete, credential scan, repo-id pin plus `whoami` assertion,
`HF_DEPLOY_COMMIT` → health poll, commit status.

`run.md` is added as a full audit trail of one run — per-stage tokens and
timings, every claim the verifier refused and why, dropped bullets, costs.

**Files** `.github/workflows/deploy-huggingface.yaml`, `.gitignore`,
`pyproject.toml`, `run.md`, `scripts/{build_sample,build_site,build_space,check_sample,check_space,deploy_space,run_trace}.py`,
`space/{README.md,index.html,style.css,assets/*}`, `tests/test_space_deploy.py`
— 18 files, 3,784 insertions.

**Measured** 25 repos in the normal report, 8 private, 29 README quotations;
42 private repos cross-checked.

**Caught** A bug that attacked the audit trail's honesty directly:

> 'state |= update' was overwriting the errors list, so the trace reported 'no
> bullets dropped' on a run that dropped five.

`|=` on a dict replaces a key's value wholesale; for an accumulating list that
is silent data loss. A run trace whose whole purpose is to disclose what was
dropped was reporting that nothing was dropped. Covered by a regression test.

Also added `--stage` / `--state` checkpointing: *"at 8K tokens/min a full run
outlives a CI step, and repeating the cached half to reach the uncached half is
waste."*

> Rationale recorded in the commit message body.

---

## 1b4cfee — fix: deploy tests must run without the agent package installed   2026-10-28

**Problem** The deploy workflow installs a minimal dependency set. The
`merge()` test imported `build_sample`, which imports the whole `rba` package,
which failed on `dotenv`.

**Decision** Move `merge()` into `scripts/run_trace.py`, which has no agent
dependencies. The alternative — installing the full package in the deploy job —
would make the deploy environment diverge from the thing it is meant to test.
The right fix is to make the deployment toolchain genuinely independent of the
agent, which is what it claims to be.

**Files** `scripts/build_sample.py`, `scripts/run_trace.py`,
`tests/test_space_deploy.py`

**Measured** *"Verified by running tests/test_space_deploy.py in a venv
containing exactly what the workflow installs: **31 passed**."*

Again the pattern from `98e0663`: reproduce the target environment exactly
rather than assert that it should work.

**Caught** A hidden coupling between the deploy toolchain and the agent package.

> Rationale recorded in the commit message body.

---

## 0767667 — fix: validate short_description length before publishing   2026-10-28

**Problem** The Hub caps `short_description` at 60 characters and **only
enforces it at publish time**, rejecting the whole commit after the bundle has
been built and uploaded.

**Decision** `validate_card` checks it locally. The principle, stated in the
commit message: *"so the failure arrives in a unit test instead of a CI deploy
step."*

This is shift-left applied to a remote validation rule. When a platform
validates late, the cheapest correct response is to replicate the rule locally
and fail early — the cost is a few lines and a duplicated constant, against a
failed deploy that has already consumed a build and an upload.

**Files** `scripts/build_space.py`, `space/README.md`,
`tests/test_space_deploy.py`

**Measured** The 60-character cap.

**Caught** A late-failing deploy.

> Rationale recorded in the commit message body.

---

## ab73a2f — ci: lint scripts/ too, and fix the import order it was hiding   2026-10-28

**Problem** CI linted `src`, `tests` and `evals` but not `scripts` — *"so the
whole deployment toolchain was unchecked."*

**Decision** Add `scripts/` to the linted paths. Adding it immediately surfaced
an import-order error in `evals/run_eval.py` introduced when `judge` became a
first-party module.

**Files** `.github/workflows/ci.yml`, `evals/run_eval.py`

**Measured** One violation found on enabling.

**Caught** The general lesson — an unlinted directory is not a clean directory,
it is an unmeasured one — recurs twice more in this history (`e2aaa2e` for
`app.py`, and the lint-discipline commits `520956a` / `91164c3`). By the end,
the lint surface covers `src`, `tests`, `evals`, `scripts` and `app.py`.

> Rationale recorded in the commit message body.

---

## b5b76b1 — docs: disclose the run-to-run variance we measured   2026-10-28

**Problem** Two independent runs over the same repositories, both at
`temperature: 0`, produced materially different ATS scores.

**Options considered** Report the better run. Report an average. Report the
variance.

**Decision** Report the variance, with the split that explains it:

| Half of the pipeline | Result |
|---|---|
| Verification (`verify_claims`) | **Identical.** Same 102/111 verdicts, the same 6 rejections, the same reasons. |
| Generation (`build_resumes`) | **Variable.** ATS scores moved by up to **24 points** (`balanced` 59.1 → 34.8). |

The interpretation is the valuable part, and it is not a rationalisation — it
falls directly out of the architecture:

> That split is the architecture working: the half deciding what may be SAID is
> deterministic Python and cannot drift; the half deciding HOW to say it is a
> language model, and Groq's gpt-oss models are not reproducible even at
> temperature 0.

Note the second clause: **`temperature: 0` is not a determinism guarantee.** It
makes sampling greedy; it does not make floating-point reduction order,
batching, or server-side routing deterministic. Treating `temperature: 0` as
reproducibility is a common and wrong assumption, and this measurement refutes
it on this provider.

The practical consequence is stated plainly rather than buried: *"treat a single
ATS score as indicative, not precise. What is stable is which claims are allowed
through — and that is the property the project actually promises."*

And the meta-point, from the commit message: *"Recording it rather than quietly
keeping the better-looking run."* 59.1 was available to report. 34.8 was equally
real.

**Files** `README.md`, `scripts/build_site.py`, `space/index.html`

**Measured** 102/111 verdicts identical, 6 identical rejections; ATS delta up to
24 points, `balanced` 59.1 → 34.8.

**Caught** An assumption about temperature-0 reproducibility, refuted by
measurement.

> Rationale recorded in the commit message body and `README.md`
> ("Reproducibility, measured").

---

## 0f027ef — deps: prepare for Streamlit Community Cloud   2026-10-28

**Problem** Three independent constraints of the target platform, each of which
would have broken the deployment.

**Decision**

1. **Dependency file priority.** *"Streamlit's dependency-file priority maps
   `pyproject.toml` to POETRY. Ours is hatchling with PEP-621 metadata and no
   lock file, so the deploy would have failed at install."* `uv.lock` outranks
   `pyproject.toml` in Streamlit's resolution order and matches the toolchain
   actually used, so the lock file is committed (3,367 lines).
2. **Memory.** Scoring dependencies added as `numpy`, `rank-bm25`,
   `onnxruntime`, `tokenizers` — **not** `sentence-transformers`.
   *"onnxruntime runs the embedding model without torch — sentence-transformers
   pulls ~1GB installed and Community Cloud only guarantees 690MB."* The
   platform's memory floor is the reason the entire dense channel is built on
   ONNX Runtime; this constraint is decided here, before the channel exists
   (`0b5e7c0`).
3. **Error disclosure.** `.streamlit/config.toml` sets
   `showErrorDetails = none`, because *"the app will be public while the repo
   stays private, and a traceback would leak source paths and possibly fragments
   of a user-supplied API key."*

That third decision has a cost that is paid later and is worth flagging now:
with tracebacks suppressed, **every failure in production is a blank page**.
That directly motivates `tests/test_app.py` driving the real app through
Streamlit's `AppTest` (`2023757`), `_diagnose()` mapping exceptions to
actionable sentences (`0da531f`), and the difficulty of diagnosing the OOM in
`667895e`.

**Files** `.gitignore`, `.streamlit/config.toml`, `pyproject.toml`, `run.md`,
`space/assets/*`, `uv.lock`

**Measured** ~1GB for sentence-transformers against Community Cloud's 690MB
guaranteed floor.

**Caught** Three deploy-time failures pre-empted by reading the platform's rules
rather than discovering them.

> Rationale recorded in the commit message body.

---

## bb3972f — ats: evidence-grounded scoring core (BM25 + skills + convex fusion)   2026-10-28

**Problem** The project's own rubric (`src/rba/scoring.py`) had invented weights.
From the commit message: *"The old roles.yaml said `langgraph: 3.0`,
`python: 1.0` because I decided so."* By the project's governing rule, that is
the same error as an invented achievement — presenting a number built from
invented weights as a score.

**Options considered**

- Keep the hand-weighted rubric and defend the weights.
- **Reciprocal Rank Fusion (k=60)** — the reflex choice for hybrid retrieval.
- Convex combination of normalised channel scores.
- Observed min-max normalisation vs theoretical bounds.

**Decision** A three-channel design following the **TalentCLEF 2026 second-place
system** (arXiv 2606.31692) — LLM parses documents to structured JSON, then rank
fusion across retrieval views — which scored **MAP 0.6632 against the winner's
0.7140, i.e. 93% of the best result, with no training.** Choosing the
second-place system is a deliberate trade: no training data exists here, and a
no-training design that reaches 93% is worth more than a trained one that cannot
be reproduced.

**Convex combination, not RRF.** Bruch et al., *An Analysis of Fusion Functions
for Hybrid Retrieval* (arXiv 2210.11934, TOIS), beat RRF with a simple weighted
sum on every dataset tested:

| Dataset | weighted sum nDCG@1000 | RRF |
|---|---:|---:|
| MS MARCO | 0.454 | 0.425 |
| Natural Questions | 0.542 | 0.514 |

plus similar gaps zero-shot on HotpotQA and FEVER. The decisive finding is the
second one: **RRF's tuned parameters fail to generalise out of domain** — which
is exactly this situation, since there is no in-domain data to tune on. Picking
the method whose failure mode is "needs tuning" when tuning is impossible would
be a mistake regardless of its headline score.

**Theoretical rather than observed min-max normalisation**, for two reasons from
the same paper and one from the use case: the choice of linear normalisation is
rank-equivalent; normalising by theoretical bounds carries one fewer
data-dependent statistic and so survives distribution shift better; and
*"observed min-max also has a nasty property for a single-document tool: with
one candidate, min == max and the normalised score is undefined."*

**BM25 replaces hand-assigned keyword weights.** `src/rba/ats/lexical.py`:
*"BM25's IDF term derives them instead: a term appearing in one job description
out of many scores high, a term appearing in all of them scores low. The
weighting comes from the corpus rather than from my opinion."* That is the
precise remedy for the invented-weights problem — not better weights, but
weights derived from data.

BM25 also keeps **literals** sharp — `AWS`, `CPA`, `Kubernetes`, version strings
— which a 384-dimension embedding blurs into "cloud" and "containers". The two
channels are complementary by construction, not redundant.

**A channel that cannot be computed returns `None` and has its weight
redistributed**, rather than scoring zero. `ChannelScores.available()`:
*"Those are genuinely unknown, not zero, and scoring them as zero would punish a
candidate for our missing inputs."* The commit credits the idea honestly:
redistribution is *"the one genuinely good piece of design in
`srbhr/Resume-Matcher`; its weights are asserted rather than fitted and are not
used here."*

The package `__init__` also states what it deliberately does **not** claim:
*"It does not claim to reproduce any real ATS. Greenhouse exposes
Strong/Good/Partial buckets and Workday A-D letter grades… No major ATS shows a
candidate a 0-100 number, so neither should we without saying whose number it
is."* Hence `ORDINAL_BANDS` (Strong ≥ 0.80, Good ≥ 0.60, Partial ≥ 0.40, Limited
otherwise) sitting beside the number.

The channel weights (`dense: 0.5, lexical: 0.2, skills: 0.3`) are labelled
honestly in the source — *"The form (convex combination) is evidence-backed;
these particular numbers are a judgement call and are reported as such in the
UI"* — with the one piece of support that exists: the dense:lexical ratio gives
dense a **71% share, inside the [0.6, 0.8] band Bruch et al. found consistently
effective** for that pair. This is the correct handling of a residual judgement
call: state it, bound it by what evidence exists, and print it on the page so a
reader can disagree.

**Files** `src/rba/ats/{__init__,fusion,lexical,skills}.py`, `tests/test_ats.py`

**Measured** Two bugs *"caught by smoke-testing before the tests existed"*:

- The tokenizer kept trailing punctuation, so `AWS.` never matched `AWS`. (Note
  this is the same failure class as the `fallback.` / `scripts.` bug in
  `01674c3` — trailing punctuation making a token unrecognisable — appearing in
  a completely different subsystem.)
- Normalising against a theoretical all-terms-present ceiling **scored a
  near-perfect resume at 0.12**, because the ceiling included filler like "we"
  and "engineer". The ceiling is now **the job description scored against
  itself**, which is interpretable and makes an identical document score 1.0.
  That second property is the test: a normalisation whose perfect case does not
  score 1.0 is not a normalisation.

**Caught** Both of the above, plus the design-level problem the whole commit
exists to fix.

> Rationale recorded in the commit message body and the `src/rba/ats/__init__.py`,
> `fusion.py` and `lexical.py` module docstrings.

---

## 4f4f6c1 — ats: curated alias normalisation, after ESCO failed an empirical test   2026-10-28

**Problem** "JS", "JavaScript" and "ECMAScript" must score as one skill, or a
resume is penalised for word choice. The question is what to normalise against.

**Options considered** **ESCO**, the EU occupational taxonomy — ~13,900 skills
against O*NET's 177, multilingual, officially maintained. This was the plan, and
it is the obvious "grounded in a standard" choice. Alternative: a curated alias
table, which is a smaller and less impressive claim.

**Decision** The plan was tested against the live ESCO API using the skills this
tool actually sees:

| query | ESCO result |
|---|---|
| python | Python (computer programming) |
| postgresql | PostgreSQL |
| javascript | JavaScript |
| kubernetes | **no results** |
| docker | **no results** |
| pytorch | **no results** |
| tensorflow | **no results** |
| aws | **no results** |
| terraform | **no results** |
| kafka | **no results** |
| react | *"react to emergency situations in a live performance environment"* |

ESCO is organised around *occupations*, so it has "use scripting programming"
but not "Kubernetes". Two further points make this fatal rather than
inconvenient:

- **Embedding nearest-neighbour cannot rescue it.** *"it is a coverage gap, not
  a matching problem."* No amount of better matching finds an entry that does
  not exist. This distinction is the one most likely to be got wrong — the
  instinct on seeing "no results" is to reach for fuzzy matching.
- **The `react` result shows the failure mode is not a miss but a confident
  wrong answer that would silently corrupt a score.** A miss is visible; a
  confident mismatch is not. `normalise.canonical()` therefore passes unknown
  skills through unchanged, with the docstring: *"Forcing it onto the nearest
  known name is how 'react' becomes 'react to emergency situations'."*

So: a curated alias table. The commit is explicit that this is a *retreat* to a
defensible position — *"A curated alias table is a smaller claim with better
accuracy"* — and lists the secondary wins: no 21MB embedding matrix, no network
call at scoring time, every mapping readable in the source. It also *"removes
the largest memory risk for the Streamlit free tier"*, which matters given the
690MB floor established in `0f027ef`.

**Files** `src/rba/ats/__init__.py`, `src/rba/ats/normalise.py` (new, 156
lines), `src/rba/ats/skills.py`, `tests/test_ats.py`

**Measured** The 11-query ESCO probe above, preserved verbatim in the
`src/rba/ats/normalise.py` docstring.

**Caught** A plausible, standards-based, more impressive design, rejected by an
eleven-query experiment that took minutes. The result is recorded in the module
docstring specifically so the idea is not re-proposed.

**A note on the skills channel itself.** `weighted_jaccard` is deliberately
**coverage rather than symmetric Jaccard**: *"A candidate is not penalised for
knowing things the job did not ask about — symmetric Jaccard would punish a
broad resume, which is the opposite of useful."* Required skills weigh more than
preferred, mirroring Greenhouse's Required (AND) / Preferred (OR) split and
Workday's basic/preferred grading. This is the one channel that can be explained
in full to a candidate who asks why they scored what they scored, which is why
it runs no model at all.

> Rationale recorded in the commit message body and the `src/rba/ats/normalise.py`
> and `skills.py` module docstrings.

---

## 0b5e7c0 — ats: ONNX dense channel and a BM25 background corpus   2026-10-28

**Problem** Two gaps: no semantic channel, and BM25 with no background corpus
is mathematically broken.

**Options considered** `all-MiniLM-L6-v2` via sentence-transformers (the default
choice in virtually every tutorial); `bge-small-en-v1.5` via ONNX Runtime.

**Decision — the model.** `BAAI/bge-small-en-v1.5` (MIT, 33M parameters, 384
dimensions). Two reasons it beats the reflex choice:

| | bge-small-en-v1.5 | all-MiniLM-L6-v2 |
|---|---:|---:|
| BEIR retrieval | **51.68** | 43.81 |
| `max_seq_length` | — | 256, **silently truncates** |

The docstring also corrects a widespread misreading: *"The 'MiniLM scores 56.3
on MTEB' figure quoted in most blog posts is the overall average across 56
tasks, not retrieval."* Silent truncation at 256 tokens is disqualifying for
documents as long as a resume — it is not a quality degradation, it is scoring a
different document than the one supplied.

**Decision — the runtime.** ONNX Runtime, not sentence-transformers.
*"sentence-transformers pulls PyTorch — roughly 1GB installed and 400-700MB
resident — and Streamlit Community Cloud only guarantees 690MB. ONNX Runtime
plus this model is about 150MB resident."* Measured peak RSS with the model
loaded: **270MB against Streamlit's 690MB guaranteed floor.**

**Decision — graceful degradation.** *"The channel degrades rather than fails:
if the model cannot be fetched (no network on first run, or a corrupted cache)
`embed()` returns `None`, the channel is dropped, and its weight is
redistributed across the others."* The whole-score architecture from `bb3972f`
is what makes this possible: a missing channel is a known unknown, not a zero.

**Measured** — two measurements that changed the code:

1. **Cosine mapping.** Mapping cosine as `(raw + 1) / 2` wasted half the scale:
   text embeddings are effectively never anti-correlated, so an irrelevant
   document scored **0.75** and a strong match **0.91** — a usable spread of
   0.16. Clamping the raw value gives **0.49 against 0.83**, roughly doubling
   the range. The textbook mapping is correct for the mathematical domain of
   cosine and wrong for the empirical distribution of text embeddings.
2. **BM25 needs a background corpus.** With only the resume and the job
   description to hand, IDF over two documents makes every term equally rare and
   the channel collapses. Measured on one pair: the same resume scored **0.180
   with no corpus and 0.476 against a 6-document background**, while an
   irrelevant resume stayed at **0.000** — so the corpus restores signal without
   inflating noise. A **22-job corpus** now ships with the package
   (`src/rba/ats/data/jd_corpus.json`).

   The corpus composition is itself a decision: it is *"deliberately spread
   across non-software roles too, so that 'Python' is rare in the corpus the way
   it is rare in the world, rather than universal the way it would be in an
   all-engineering sample."* An all-engineering background would drive the IDF
   of every relevant term towards zero — exactly the terms the channel exists to
   weight. And `corpus.py` is explicit about what the corpus is not: *"They are
   not training data, they carry no labels, and no part of the score is fitted
   to them."*

**End-to-end result with the semantic channel on:** strong **72.6**, partial
**52.0**, wrong field **23.2**, *"each with the missing required skills named"*.

**Files** `src/rba/ats/{__init__,corpus,dense,score}.py`,
`src/rba/ats/data/jd_corpus.json`, `tests/test_ats.py`

**Caught** Both measurement-driven corrections above.

`idf()` also carries a small but real guard: Robertson/Sparck-Jones IDF with the
usual +0.5 smoothing, **floored at zero**, because *"a term present in more than
half the corpus can otherwise go negative, which would let a very common word
subtract from a score."*

> Rationale recorded in the commit message body and the `src/rba/ats/dense.py`
> and `corpus.py` module docstrings.

---

## 2023757 — app: Streamlit scorer with the formula on the page   2026-10-28

**Problem** Ship something usable publicly. The full agent cannot be it.

**Options considered** Deploy the full agent as a web app. Deploy the scorer
only.

**Decision** Scope to the scorer. The constraint arithmetic, from the commit
message: *"A full run is 8-25 minutes against Groq's 8K-tokens-per-minute free
tier and Streamlit Community Cloud drops the websocket after roughly 10-15
minutes; scoring is instant and needs no key."* The two windows barely overlap,
so the agent-as-web-app is not a hard problem — it is an impossible one on this
infrastructure. (`6d9a463` later ships a *capped* GitHub import, which is the
version that fits.)

**The page states what the number is not.** From the commit message:

> no applicant tracking system shows a candidate a 0-100 score. Greenhouse
> reports Strong/Good/Partial, Workday reports A-D, and both are driven by
> employer-configured per-job criteria. The band beside the number exists to
> stay recognisable, and **the weights are printed on the page -- being able to
> disagree with the mix is the point.**

Printing the formula is the direct answer to the invented-weights problem. The
weights remain a judgement call; making them visible and contestable is what
separates a judgement call from an assertion.

**Testing.** `tests/test_app.py` drives the real `app.py` through Streamlit's
`AppTest`. The justification is specific and traces back to `0f027ef`:
*"Without it a broken app.py would fail only in production, where
showErrorDetails=none hides the traceback and viewers get a blank page."* Having
chosen to suppress error detail for security, the project owes itself a test
that exercises the real app end to end — otherwise the security decision has
removed the only feedback channel.

**Files** `app.py` (new, 163 lines), `tests/test_app.py`

**Measured** 8–25 minutes for a full run; 10–15 minute websocket window.

**Caught** Nothing; a scoping decision made on measured platform limits.

> Rationale recorded in the commit message body.

---

## 4ff6b0c — docs: link the Streamlit app from the README   2026-10-28

**Problem** A cold link to a hibernating Community Cloud app lands on a "Yes,
get this app back up!" interstitial, which reads as a broken deployment.

**Decision** Link it *with* the caveat. *"Community Cloud sleeps an app after
12h and the wake is manual, so a cold click lands on an interstitial -- saying
so beats letting someone conclude the app is broken."*

The README text: *"The app sleeps after 12 hours without traffic. A cold link
shows a 'Yes, get this app back up!' button — click it and give it about 30
seconds."*

**Files** `README.md` (9 insertions)

**Measured** 12-hour hibernation; ~30 seconds to wake.

**Caught** An avoidable bad first impression.

> Rationale recorded in the commit message body.

---

## ac4b5f6 — fix: declare markdown, which the HF page renderer silently depended on   2026-10-28

**Problem** `scripts/build_site.py` imports `markdown`, but the package was
installed ad hoc and never declared. The next `uv sync` dropped it,
`build_site.py` stopped running locally, and `space/index.html` drifted from its
sources.

**Decision** Declare the dependency and regenerate the page.

**Files** `pyproject.toml`, `space/index.html` (+313/−371), `uv.lock`

**Measured** None recorded.

**Caught** The undeclared dependency — but the notable part is **how**:

> which the deploy workflow's staleness check caught and failed on, exactly as
> intended.

The deployment pipeline from `b570e64` includes a check that generated artefacts
match their sources. That check caught a dependency bug three commits
downstream, in a different subsystem, by noticing that a generated file had
stopped being regenerated. A staleness check is normally justified as preventing
hand-edits to generated files; here it functioned as an integrity check on the
build environment. Worth noting that the project recorded the check working as
designed rather than just the fix.

> Rationale recorded in the commit message body.

---

## e2aaa2e — ci: lint app.py too, and fix the violations it was hiding   2026-10-28

**Problem** `app.py` sits at the repository root, outside `src/tests/evals/scripts`,
so it was never linted. It had unused `noqa` directives and RUF005
concatenations added with the scoring core.

**Decision** Add `app.py` to the linted paths. Same lesson as `ab73a2f`: the
lint surface must be enumerated deliberately, because a path-based lint config
silently excludes anything not listed, and root-level files are the easiest to
forget.

**Files** `.github/workflows/ci.yml`, `src/rba/ats/__init__.py`,
`src/rba/ats/score.py`, `tests/test_app.py`, `tests/test_ats.py`

**Measured** None recorded (a handful of violations).

**Caught** The violations, and the gap that hid them.

> Rationale recorded in the commit message body.

---

## 3656473 — ats: LLM extraction with source verification, and percentile calibration   2026-10-28

Two substantial pieces.

### Extraction — the model parses, and nothing else

**Problem** The skills channel needs structured skills, and resumes are prose.

**Options considered** Let the LLM extract skills directly (the obvious use).
Confine the LLM to parsing and verify every extracted skill against the source.

**Decision** The split follows published evidence rather than intuition:

| task | best reported | source |
|---|---:|---|
| zero-shot resume **parsing** | ~0.96 F1 | arXiv 2510.09722 |
| LLM **skill extraction** (strict-span F1), GPT-4 | 27.8 | arXiv 2402.03832 |
| same task, supervised model | 62.6 | arXiv 2402.03832 |

*"Parsing is nearly solved; deciding what counts as a skill is not."* So the
model parses resume prose into JSON Resume fields and **every extracted skill
carries the text it came from**; `source_verify()` drops any whose evidence is
absent from the document. Invented dates are cleared rather than kept.

The failure being defended against is named precisely, and it is not the one
people expect: *"the characteristic failure is not a miss but an **inflation**:
a model reading 'built a REST API' will happily also report Docker, CI/CD and
microservices, and **each invented skill raises the match score** against a job
description that asked for them."* The error is self-serving — it moves the
score in the flattering direction — which is exactly when a mechanical check is
most needed.

`source_verify` *"Returns the cleaned extraction and the list of rejected skill
names, which the caller shows rather than hides — a reader should be able to see
what the model tried to add."* Same principle as the report's refusal list.

### Calibration — a percentile, not a score out of 100

**Problem** *"A raw fused score of 0.63 is not interpretable. Neither is
'63/100' — it **looks** interpretable, which is worse, because it implies a
calibration nobody has done."*

**Options considered** Present the raw fused score ×100. Do real calibration
against hiring outcomes. Report a percentile against a stated reference set.

**Decision** Real calibration would mean *"candidates scoring 80 reach the next
stage about 80% of the time"*, measured against interview outcomes — which needs
labelled hiring data that *"does not exist publicly in English; the one public
person-job dataset with real labels is Chinese."* So: a percentile against a
stated reference set. *"That claim is true by construction, needs no outcome
labels, and makes the reference set the explicit assumption — which is the
honest place to put the uncertainty."*

Implementation is a short list of quantile breakpoints, so runtime scoring is a
binary search over a hundred floats.

**Measured — two things the reference set taught, both recorded in
`scripts/build_calibration.py`:**

1. **The all-pairs reference set was flattering nonsense.** Pairing every
   synthetic resume with every job description gave **462 of 484 pairs
   cross-family** (a nurse against a Kubernetes role). The bar therefore became
   "better than a random pairing", and *"a mediocre raw score of 0.6 came out at
   the 92nd percentile. Flattering and useless."* Rebuilt from **plausible**
   pairs only — three resume strengths per job family, each against its own
   family's job description — giving **strong 80th, medium 54th, weak 11th**.
   The general point: a percentile is only as meaningful as the comparison set,
   and an easy comparison set produces a number that is true and worthless.
2. **The reference set must record which channels produced it.** Built with the
   semantic channel on and then scored with it off, *"every result floored at
   the 0th percentile and the ordering collapsed."* Calibration now **refuses a
   mismatched channel set and says so** instead of silently returning garbage.
   This interacts directly with the graceful-degradation design from `0b5e7c0`:
   a channel that can vanish at runtime means a calibration curve has a
   precondition, and the precondition has to be checked.

**Known limitation, stated rather than hidden** (`build_calibration.py`): PJB
(arXiv 2603.17386) found *"performance heterogeneity across industry domains
exceeds the gains from any module upgrade, so one global distribution will
misrank some sectors. The reference set is small and synthetic; it fixes the
scale, not the ranking."*

**Files** `app.py`, `scripts/build_calibration.py`, `src/rba/ats/calibrate.py`,
`src/rba/ats/data/calibration.json`, `src/rba/ats/extract.py`,
`src/rba/ats/score.py`, `src/rba/prompts/extract.v1.md`, `tests/test_app.py`,
`tests/test_ats.py`, `tests/test_extract.py`

**Caught** Both calibration flaws, before publication.

> Rationale recorded in the commit message body and the `src/rba/ats/extract.py`,
> `calibrate.py` and `scripts/build_calibration.py` module docstrings.

---

## 520956a — test: rewrite a dict() call as a literal (C408)   2026-10-28

**Problem** CI failed on a lint rule that passed locally.

**Decision** Fix the violation, and fix the process: *"My local ruff passed on a
stale cache; CI, which has none, caught it. Running with `--no-cache` locally
from here."*

**Files** `tests/test_extract.py` (3 lines)

**Measured** None.

**Caught** A tooling-cache divergence between local and CI. The commit is
trivial; recording the *process* change is not. Together with `91164c3` it shows
a feedback loop being closed rather than a nuisance being swatted.

> Rationale recorded in the commit message body.

---

## 2d86702 — docs: point the badge at the deployed subdomain   2026-10-28

**Problem** The README badge pointed at a URL that does not serve the app.

**Decision** *"Deployed as aniketqw-resumebuilder, not resume-builder-agent."*
Point the badge at `https://aniketqw-resumebuilder.streamlit.app`.

**Files** `README.md` (2 lines)

**Measured** None.

**Caught** A dead link in the most prominent position in the README.

> Rationale recorded in the commit message body.

---

## 2f96654 — resume: editable document and exact per-bullet attribution   2026-10-28

The intellectual centre of the second half of the project.

**Problem** The product idea is a toggle beside every bullet showing what that
line is worth. That number has to be **what removing the line actually costs**,
or the feature is a guess dressed as a measurement.

**Options considered**

1. Exact leave-one-out: remove each bullet, re-embed, re-score.
2. A **mean-pool approximation**: embed each bullet once, mean-pool the kept
   ones, never re-embed.
3. An LLM-estimated impact per bullet.

**Decision** Exact leave-one-out — and the reason option 2 was rejected is the
single best example of empirical design discipline in this history.

The approximation is the obvious engineering answer and the author says so
plainly: *"I nearly shipped an approximation instead… I was ready to recommend
it until I measured it against ground truth."*

Measured on a 20-bullet resume (bge-small, CPU), from the
`src/rba/resume/attribution.py` docstring:

| approach | cost | accuracy |
|---|---|---|
| exact leave-one-out, all 20 deltas | 4.9 s | exact |
| re-score after one toggle | 153 ms | exact |
| mean-pool approximation | 1.15 ms | **sign wrong 6/20** |

The approximation is **4,253× faster**. Against ground truth it scored **Pearson
0.643** and agreed on **sign for only 14 of 20 bullets**.

Pearson 0.643 is the trap. On a correlation metric the approximation looks like
a usable, defensible speed/accuracy trade. But the user-facing question is not
"how large is this bullet's effect" — it is "**should I keep this line**", and
that is a question about the **sign**. An approximation that gets the sign wrong
on 30% of bullets *"would tell a user a line helps when it hurts"*, which is the
governing rule violated in the most direct possible way. The correct metric for
a decision support feature is the one that matches the decision, not the one
that is conventional.

And the economics make exactness affordable at all: *"We can afford that because
**scoring makes no LLM calls** — it is one ONNX embedding plus arithmetic."*
4.9 s behind a spinner *"is nothing next to the minutes of LLM work that
produced the bullets"*, and after that each toggle costs one embedding.

The rejection is recorded in the module docstring deliberately — *"with the
numbers recorded in the module docstring so nobody re-proposes it"* — which is
the right place for it. A rejected design with a measurement attached is a
permanent asset; the same rejection in a commit message alone would be
rediscovered.

**Why deltas do not sum to the total.** The docstring states it:
*"Deltas do not sum to the total, because bullets interact. They answer the
question asked and no other."* This is worth unpacking, because the
non-additivity is not a defect and a reader will otherwise assume it is:

- The fused score is **non-linear in the document's content**. BM25 saturates in
  term frequency — the `freq * (k1 + 1) / (freq + k1 * (1 - b + b * len/avgdl))`
  form means a second occurrence of a term adds less than the first, so removing
  a bullet that repeated a term already present elsewhere costs almost nothing,
  while removing the *only* carrier of a term costs a lot.
- BM25 also **normalises by document length** (the `b * len/avgdl` term), so
  removing any bullet shortens the document and raises every remaining term's
  contribution.
- The dense channel embeds the **whole document** and the cosine is over an
  L2-normalised vector, so a bullet's contribution depends on what else is in
  the vector.
- The skills channel is **set-based**: a skill named in two bullets survives the
  removal of either.

Consequently the deltas are leave-one-out marginals, not a decomposition. A
decomposition (Shapley values, say) would answer "what share of the total does
this line deserve", which is a fair-attribution question. Leave-one-out answers
"what happens if I remove this line", which is the question a toggle UI actually
poses. `rescore()` is documented as *"the authority on any particular
selection"* — if you want to know what a specific set of toggles scores, you
score that set.

**Grounded scoring buys something an LLM-guessed impact cannot.** From the
commit message: *"some bullets score NEGATIVE. Against a backend job
description, removing 'Ran incident response and wrote postmortems' raises the
score by 0.7, so the app can honestly tell someone to delete a line."* A model
asked to rate a bullet's value will essentially never say "this is hurting you";
an arithmetic re-score will, when it is true.

**Cost scales with text volume, not bullet count.** *"Cost scales with how much
text each embedding carries, not just with the bullet count"* — every variant
embeds the whole document, so an N-bullet resume costs N full-document
embeddings. This is the fact that causes the OOM fixed in `667895e`.

**Files** `out1/*` (a real run's output, committed as evidence),
`pyproject.toml`, `src/rba/resume/__init__.py`, `src/rba/resume/attribution.py`,
`src/rba/resume/document.py`, `tests/test_resume.py`

**Caught** A floating-point reporting artefact that made a correctness test look
broken:

> Attribution carries `raw_total` alongside the rounded `total` -- comparing a
> 1dp-rounded total against an unrounded rescore showed **0.051 of apparent
> error** and made the exactness test look broken. It is now exact to **4dp**,
> asserted against an independent rescore for every bullet.

The `Attribution` dataclass comment preserves it: *"comparing a rounded total
against an unrounded rescore introduces up to 0.05 of apparent error — enough to
make an exactness test look broken."* The important move is that the apparent
inexactness was *investigated* rather than accommodated with a looser tolerance.
The feature's entire claim is exactness; a test that tolerates 0.05 would have
retired the only check on it.

> Rationale recorded in the commit message body and the
> `src/rba/resume/attribution.py` and `document.py` module docstrings.

---

## 6d9a463 — app: GitHub import, resume upload, and per-bullet +/- scoring   2026-10-28

**Problem** *"The deployed app was still the old paste-a-resume-and-a-JD page --
none of the document model or attribution work was wired into it, so nothing new
was visible."*

**Decision** Ship the features now; match styling later. The self-criticism in
the commit message is the substance:

> I had been holding the UI back waiting for the artifact source; that was the
> wrong call, because it left the app unchanged.

The app becomes three steps: **source** (GitHub or upload) → **target** (job
description or role archetype) → an **editable resume** where every bullet
carries the exact score it contributes.

Two constraints shape the GitHub import, both already established:
`src/rba/resume/pipeline.py` runs the existing select/fetch/summarise/verify
nodes *with a repo cap*, because of the 8K TPM ceiling and the 10–15 minute
websocket window. And resume parsing is **deterministic — no LLM — so it never
invents content from an upload.**

**Files** `app.py` (+398/−185 net restructure), `src/rba/resume/__init__.py`,
`src/rba/resume/pipeline.py` (new), `tests/test_app.py`

**Measured** End-to-end verification on a pasted resume:

> **+12.5** and **+12.6** in green for the Python/Docker/Kubernetes lines and
> **−1.1** in red for 'Organised the office social committee', with a warning
> naming the lines that are lowering the score.

That negative number is the demonstration of the `2f96654` argument:
*"A scorer that guessed impacts could not say that honestly."*

**Caught** A process failure (holding a feature back for cosmetic reasons),
corrected explicitly.

> Rationale recorded in the commit message body and the
> `src/rba/resume/pipeline.py` docstring.

---

## 91164c3 — test: drop noqa directives that tests/* per-file-ignores already covers   2026-10-28

**Problem** Redundant `noqa` directives in `tests/test_app.py`, flagged by CI.

**Decision** Remove them. Process note: *"I ran the lint before the final edit
to test_app.py, so CI caught what I had not. Running CI's exact command before
pushing from now on."*

**Files** `tests/test_app.py` (2 lines)

**Measured** None.

**Caught** Not the lint error — the *habit*. This is the second half of the loop
started in `520956a`: first "disable the local cache", now "run CI's exact
command, not an approximation of it, and run it last". Two trivial commits that
together close a real feedback gap.

> Rationale recorded in the commit message body.

---

## 0da531f — fix: the GitHub token the UI collects was never passed anywhere   2026-10-28

**Problem** Users with correct credentials saw `Import failed: RuntimeError`.

**Cause** Two independent defects compounding:

1. `build_from_github` **accepted** `github_token` and `jev_key` and **used
   neither.** `make_backend` took no token at all, so every backend fell back to
   reading the environment. *"That works on a laptop with a .env and fails on
   Streamlit, where there is none."* A signature that accepts a credential and
   ignores it is worse than one that does not accept it — the caller has every
   reason to believe it was passed.
2. The error message *"printed only `type(exc).__name__`, which was true and
   useless."*

**Decision**

- `make_backend` now takes an explicit token and threads it through both MCP and
  REST. **Jev is made optional**: without a key, selection falls back to its
  heuristic rather than failing the whole import. The asymmetry is correct —
  GitHub access is load-bearing (there is nothing to import without it), Jev is
  an optimisation, and `heuristic_score()` exists precisely as the *"Fallback
  used only when Jev is unreachable. Intentionally crude."*
- `_diagnose()` maps the real failures — missing token, 401, 403, 404, 429 — to
  a sentence saying what to do, and **redacts anything token-shaped, since this
  app is public.** The redaction is not incidental: an error path that echoes
  input is a credential-disclosure path on a public app.

**Files** `app.py`, `src/rba/github_tools.py`, `src/rba/resume/pipeline.py`

**Measured** *"Verified end to end with the environment variables removed, so
the explicit token is the only possible source: **58 public repos, 3
summarised, 14 claims verified and 4 refused, in 95 s.**"*

Removing the environment variables is what makes the test valid. With a `.env`
present, a run proves nothing about whether the parameter is wired — the
fallback would have served it. This is the same discipline as `98e0663`
(fresh clone) and `1b4cfee` (exact deploy venv): reproduce the deficient
environment, not the comfortable one.

**Caught** A dead parameter that had been silently ineffective since
`6d9a463`, and an error message that destroyed the only diagnostic information
available on a platform where `showErrorDetails = none` (set in `0f027ef`).

> Rationale recorded in the commit message body and the `_diagnose` docstring in
> `app.py`.

---

## 667895e — fix: chunk embeddings so attribution cannot be OOM-killed   2026-10-29

**Problem** A GitHub import of more than a handful of repositories produced a
blank *"Error running app"* page on Streamlit Community Cloud, **with no
traceback** — *"because the process never gets far enough to raise one"*. The
OOM reaper had killed it.

**Cause** This is the `2f96654` cost model meeting the `0f027ef` memory floor.
Scoring N bullets embeds **N + 2 near-full-length copies of the resume** (one
per leave-one-out variant, plus the full document and the job description), and
`embed()` sent the whole batch through the transformer in a single call. The
ONNX model holds a `[batch, seq_len, hidden]` activation for every layer at
once, so peak memory is linear in batch size × sequence length — and here the
sequence length is a whole resume for every element.

**Measured peak RSS** on full-page resume texts
(`src/rba/ats/dense.py` comment and commit message):

| texts | unchunked peak RSS |
|---:|---:|
| 152 | 4,133 MB |
| 80 | 2,034 MB |
| 40 | 1,368 MB |
| 16 (chunk 16) | 574 MB |

Streamlit Community Cloud gives about **1 GB**. Even 40 texts exceeded it.

**Options considered** Cap the number of bullets scored (degrades the feature).
Reduce the model's sequence length (truncates the resume — exactly the
`all-MiniLM` failure rejected in `0b5e7c0`). Chunk the batch.

**Decision** `embed()` runs in **chunks of 4**. The docstring claims only what
was measured: *"the result is identical to a single large call, but peak memory
stays flat in the number of texts instead of growing with it."*

**Measured — the trade-off is essentially free:**

| | peak RSS | 152 texts |
|---|---|---|
| chunked (4) | **446 MB at any size** | **35.7 s** |
| batch 8 | — | 36.6 s |

Chunking is marginally *faster* here, which is not surprising once memory
pressure is the binding constraint, but it is recorded as a measurement rather
than asserted.

**The honest caveat about exactness**, which matters because exactness is the
feature's entire claim:

> Chunk boundaries change padding, so results are equal to float32 tolerance
> rather than bit-identical. Measured against an independent rescore on a real
> resume, the worst disagreement moves from **3.94e-05 to 4.80e-05 points on a
> 0-100 scale** — the discrepancy is inherent to batch padding and **predates
> this change**.

Three things are right about that paragraph. It names the mechanism (padding
differs per chunk, so attention over pad tokens produces float32-level
differences). It quantifies the change rather than waving at it — 3.94e-05 →
4.80e-05 on a 0–100 scale is nine orders of magnitude below the display
precision. And it states that the effect **pre-existed the change**, so the
reader can tell that chunking did not introduce approximation, it merely moved
an existing floating-point artefact slightly.

**Files** `src/rba/ats/dense.py`, `src/rba/resume/pipeline.py`,
`tests/test_ats.py`

**Caught** The OOM, plus a second silent failure: *"an expired Groq key made
every summary return 401 and the run returned an empty resume with no reason
given."* Per-repo failures now reach the caller in `stats["errors"]`, surfaced
by `_report_partial_failures` in `app.py`: *"Say what went wrong per repository
instead of returning a bare resume."*

Both halves of this commit are the same bug class as `fe93e24`: **a failure
that renders as plausible output.** An empty resume and a blank page are both
states a user will read as "it did not find much" rather than "it broke".

> Rationale recorded in the commit message body and the `src/rba/ats/dense.py`
> comments and `_report_partial_failures` docstring in `app.py`.

---

## e962065 — feat: start from your own resume, and render it back out   2026-10-29

The largest commit after M0 — 27 files, 3,004 insertions — and the one that
changes what the product *is*: from "generate a resume from your GitHub" to
"improve the resume you already have".

**Problem** *"The document model was flat -- name, contact, sections of bullets
-- which cannot hold what a real resume contains."* Parsing a real PDF
(`AniketSaxena_KPMG.pdf`) found an academic-details table, label-value skill
rows, dated entries with a `Tech:` line, and a certifications table — **none of
which had anywhere to live.**

A flat model does not merely lose formatting. It loses *content*, and the
content it loses is exactly the content repositories cannot supply: an
internship, a teaching assistantship, a grade.

### The pieces

**`document.py` — Entry, Table, KeyValues, Header.** Added **alongside** the
existing shape, *"so attribution and every prior test keep working unchanged"*.
An additive extension rather than a rewrite, which matters because `attribution`
asserts exactness against the document model and a rewrite would have invalidated
every one of those assertions at the same time as changing the model.

`Bullet` gains `origin` (`yours` | `repo` | `assumed`) for provenance colouring —
so a reader can distinguish a measured fact from an assumed one at a glance.

One subtle fix inside this: *"`as_text()` now includes tables and key-value rows
-- without that the scorer silently stops seeing the skills and education
sections."* The document model feeds the scorer through one method; a new
container type that `as_text()` does not traverse is invisible to BM25 and to
the embedding. The resume would have scored as though it had no skills section.
Silent, plausible, and wrong — the same class as `fe93e24` and `667895e`.

**`ingest.py` — deterministic PDF parsing.** The rationale is the governing rule
applied to the input side, and it is stated in the first paragraph of the
docstring:

> An LLM asked to "parse this resume" will happily smooth over what it cannot
> read -- inventing a date, merging two projects, inferring a degree -- and that
> is the same failure this project exists to prevent on the generation side. So
> this is plain parsing: what it cannot prove from the text, it leaves empty for
> the user to fill in.

That is the whole argument in four lines. Nobody would call a hallucinated
parse an "invented achievement", but it is the same thing arriving by a
different door, and an invented degree is worse than an invented metric.

The hard technical problem is that *"PDF text extraction destroys column
boundaries and wraps long lines."* Two rules recover most of it, and both avoid
guessing:

- *"A line that continues the previous one is detected **structurally**, not by
  guessing -- bullets end in a full stop, entry headings end in a date."*
- *"Table rows are accumulated until the row looks complete (ends in a year),
  because a wrapped cell would otherwise read as its own row."*

The commit says it *"Recovers all six academic columns by anchoring on the
provable ends"* — anchoring on what can be proven (a year, a full stop, a
CGPA/percentage suffix) and letting the unprovable middle follow. The regexes
reflect real PDF variance: `_DASH = "[-–—]"` with the comment *"Hyphen,
en dash and em dash: a PDF may carry any of the three."*

`looks_like_bits()` gates the strict parser on finding ≥ 3 known headings;
anything else falls back to `parse_resume_text()` in `pipeline.py`, which makes
weaker assumptions. A format-specific parser that *knows* when it does not apply
is honest; one that applies itself regardless is a hallucination engine.

**`merge.py` — generated lines as opt-in candidates.** The rule, from the
docstring:

> **the user's resume is the baseline and is never edited.** Generated lines
> arrive switched off, marked as coming from repository evidence, and are
> adopted only when the user turns one on. That is what makes "+23.0 against
> your uploaded resume" a measurement rather than a claim -- the baseline is a
> real document, and the delta is a real rescore of it.

And what the rejected alternative would have cost:

> rewriting the user's bullets in place, would make the before and after
> incomparable, and would quietly discard the things the repositories cannot
> prove: an internship, a teaching assistantship, a grade.

`merge_candidates` preserves existing bullets' text, order **and ids**, *"so a
score computed before the merge still describes the same document afterwards as
long as nothing is switched on"* — which is what makes the baseline a stable
reference point rather than a moving one. Duplicate suppression is by normalised
text (`normalise()` strips to `[a-z0-9]`), with the comment *"the user already
said this in their own words"*.

**`render/` — one document, four formats.** Every renderer is a pure function of
`(doc, options)`; *"Nothing here fetches, scores or calls a model, which is what
lets the UI re-render on every keystroke without spending a token."*

The critical piece is `render/model.py`, *"The one place that decides what a
renderer is allowed to show"*: every format walks the same flattened view, *"so
turning numbers off or hiding a disabled bullet cannot behave differently in the
PDF than it does in the preview the user approved."* Four independent renderers
each applying the visibility rules themselves would eventually disagree, and the
disagreement would be between what the user approved and what they sent.

`text_of()` also closes a loop back to the verifier: when the numbers switch is
off, a number is **replaced by the measurement placeholder, not deleted**
(*"Dropping a number has to leave a sentence that still reads"*), and *"the same
`numbers_in` the verifier uses decides what counts as a number, so the two can
never disagree about which tokens are metrics."*

Format choices, each with a stated reason:

| format | choice | why |
|---|---|---|
| PDF | fpdf2, drawn directly | *"Streamlit Community Cloud gives us a Python environment and nothing else. There is no `pdflatex`, and WeasyPrint needs cairo and pango as system libraries. fpdf2 is a pure-Python wheel, so it is the only one of the three that can actually produce a file on the box this runs on."* |
| LaTeX | plain `article`, no custom class | *"it compiles on Overleaf as-is without chasing a missing `.cls`"* |
| HTML | inlined CSS, print-like | *"the preview's job is to answer 'is this still my resume?' -- a question a list of Streamlit widgets cannot answer"* |
| plain | unstyled, predictable order | for the ATS audience, which wants the opposite of what a human does |

The PDF renderer *"shrinks to fit one page and **verifies the page count** rather
than assuming"* — checking the property instead of trusting the layout
arithmetic.

**`diagnostics.py` — advice without invented weights.** The seven-bar rubric
(keywords 44, role fit 18, numbers 12, sections 10, contact 6, length 6, verbs 4)
is discarded as a *score*, because *"Nobody measured that a missing phone number
is worth exactly six points."* But the rubric contained something real: *"it
could say **which keyword is missing** and **that the resume is too long**. That
is advice, and advice does not need a weight to be worth reading."*

This is a careful separation rather than a wholesale rejection — the diagnostic
value and the scoring value were entangled, and only the scoring half was
unfounded. The headline number stays the calibrated three-channel fused score.

`keyword_coverage()` adds a genuinely useful third category beyond
present/absent: terms **demonstrated** in a bullet, **mentioned** only in a
skills list, or **missing**. *"The middle group is the useful one: 'RAG' sitting
in a skills list and never appearing in a project is a gap a reader will
notice."* And everything reuses the detectors in `rba.scoring` *"so the two can
never drift apart on what counts as a number or a strong verb."*

**Files** `README.md`, `app.py` (+489/−…), `evals/budget.json`, `evals/budget.md`,
`fixtures/aniket_bits.txt`, `pyproject.toml`, `scripts/measure_budget.py`,
`src/rba/groq_client.py`, `src/rba/resume/{__init__,attribution,diagnostics,document,ingest,merge}.py`,
`src/rba/resume/render/{__init__,html,latex,model,options,pdf,plain}.py`,
`src/rba/roles.yaml`, `tests/{test_app,test_ingest,test_merge,test_render}.py`,
`uv.lock`

**Caught — two data gaps the real resume exposed:**

> "wrote", "added", "used" and "split" were missing from the action-verb list,
> and section matching was exact-string, so ACADEMIC DETAILS never matched
> "education".

Both are the signature of a word list built from imagination rather than from a
corpus. Testing against a real document found them in one pass. `fixtures/aniket_bits.txt`
is committed for exactly this reason — the README calls it *"real extracted
resume text, so parsing is tested offline"*.

**Caught — a latency claim corrected downward:**

> measured on this 662-word resume a rescore takes **582 ms** and a full
> attribution **5.4 s**, not the 150 ms carried over from a smaller test
> document.

`attribution.py` now carries both figures and the explanation: *"slower per
bullet than the 20-bullet figures above, because every variant embeds the whole
document."* Correcting a published performance number *upward in cost*, on a
real input, when the old number came from the author's own smaller test, is the
same discipline as `b5b76b1`.

### The token-budget study

`evals/budget.md` and `evals/budget.json` land in this commit, produced by
`scripts/measure_budget.py`. The script's purpose is stated bluntly: *"The
plan's 275K-token figure was an estimate of mine. This replaces it with a
measurement."*

**Measured** — 10 repos, cold (local response cache bypassed), 352 s wall clock:

| | tokens |
|---|---|
| prompt | 24,054 |
| completion | 6,533 |
| **total** | **30,587** |
| requests | 10 |
| served from Groq's prefix cache | 768 (3.2% of prompt) |

Per repo: **1 request, 3,058 tokens** — 631 stable system prompt, ~1,774
variable evidence, 653 completion. Prompt is **79%** of the cost, and the
evidence bundle is **74%** of the prompt.

**The study overturned three stated beliefs from the plan**, and `budget.md`
lists them under that heading:

1. **"Requests were never the binding constraint."** The plan treated
   "770 → 110 requests" as the headline fix. Measured: 110 repos at one call
   each is 110 requests — **11% of the 1,000/day limit**. Even the naive
   seven-calls-per-repo shape is 770, which fits. *"Tokens bind; requests do
   not."* The headline optimisation was solving a problem that did not exist.
2. **"The prompt is already prefix-structured."** Step 3 of the plan called for
   restructuring so stable instructions come first and repo facts last.
   `summarize.v2.md` *"already does exactly that — stable `# system`, variable
   evidence in `# user`. There is nothing to restructure."* A planned work item
   deleted by reading the artefact.
3. **"Groq prefix caching does not fire for us."** The significant one.
   Controlled test, identical system prefix, calls seconds apart:

   | stable prefix | `cached_tokens` on the repeat |
   |---|---|
   | 771 tokens | 0 |
   | 1,461 tokens | 0 |
   | 2,841 tokens | 0 |
   | the real 631-token `summarize.v2` system prompt, 5 calls back to back | 0, 0, 0, 0, 0 |

   *"So the plan's third pillar — 'cached tokens are exempt from rate limits' —
   has no measured support on this account. The 768 cached tokens in the real
   run are 3.2% and unexplained; they are not a strategy. **The token budget has
   to close without prefix caching.**"*

   Note the treatment of the 768 anomaly: it is reported, labelled unexplained,
   and explicitly refused as a basis for planning. The tempting move — count it
   as evidence that caching works and extrapolate — is exactly what the project
   forbids.

**A fourth finding about observability.** There is **no daily-token header**.
Groq returns only `x-ratelimit-limit-requests: 1000` and
`x-ratelimit-limit-tokens: 8000`. Both were decoded as continuous-refill
buckets, and the decoding was *checked*: *"12 requests used → 17m16s to refill
is exactly 1,000/day; 1,946 tokens used → 14.6 s is exactly 8,000/min."*
The 200,000/day cap is **documented but not observable**, so *"every percentage
below is against a number we cannot verify"* — a caveat attached to every TPD
percentage in the file rather than stated once and forgotten.

**What stacking is actually worth** (projected from measured per-repo figures;
completion assumed ~4× for the larger stacked output — *"the one estimate
here"*, labelled as such):

| repos | 7 separate calls | 1 stacked call | saving |
|---:|---:|---:|---:|
| 15 | 321,090 | 81,255 | 75% |
| 25 | 535,150 | 135,425 | 75% |
| 58 | 1,241,548 | 314,186 | 75% |
| 110 | 2,354,660 | 595,870 | 75% |

*"Stacking still earns its place — but because it stops re-sending the
1,774-token evidence bundle seven times, not because of request count."* The
optimisation survives; its justification is replaced.

**The conclusion that matters**, and the most self-critical line in the
repository:

> **110 repos was a straw man I invented.** A resume carries on the order of ten
> projects. Selection already filters to repos worth describing, and only those
> reach an LLM.

| | requests | tokens | % of documented TPD | wall clock at 8K/min |
|---|---:|---:|---:|---:|
| 15 repos, stacked | 15 | 81,255 | 40.6% | 10.2 min |
| 25 repos, stacked | 25 | 135,425 | 67.7% | 16.9 min |
| 58 repos, stacked | 58 | 314,186 | 157% | 39.3 min |

*"15–25 repos fits comfortably with no caching tricks at all. Above ~35 it stops
fitting the documented daily cap, and past ~20 minutes the Streamlit socket is
the real limit regardless of quota."* The binding constraint turns out not to be
quota at all but the websocket window from `2023757` — which is why
`config.yaml` sets `selection.max_repos: 25`.

**What still holds** after the study: *"The local SQLite response cache works:
the warm re-run of the same 10 repos cost **0 requests and 0 tokens** in 108 s."*
And *"The 8,000 TPM ceiling is real and is what produced the four 429 backoffs
during the cold run."*

> Rationale recorded in the commit message body, `evals/budget.md`,
> `scripts/measure_budget.py`, and the `src/rba/resume/{ingest,merge,diagnostics,document}.py`
> and `render/*` module docstrings.

---

# Session 2 — live point scoring

Six commits, written as the work happened rather than reconstructed afterwards.
Every number below was measured in this session; the command that produced it is
named.

---

## ac2197a — perf: halve a re-score, and stop the baseline moving with the user   2026-10-29

**Problem** Two separate faults blocking live editing. A re-score cost 590 ms,
which is a wait rather than a response. And `baseline()` carried the user's
current toggles, so switching off one of their own lines switched it off in the
baseline too — "+N vs your uploaded resume" reported *no change* on a document
that had plainly changed.

**Investigation** Timing the embedding call by part was what exposed the first:

| call | cost |
|---|---|
| `embed([doc, jd])` | 590 ms |
| `embed([doc])` | 291 ms |
| `embed([jd])` | 10 ms |

The parts cost half what the whole does, which is only possible if batching is
doing extra work. It is: `tokenizer.encode_batch` pads every text in a chunk to
the longest member, so a ten-word job description sent alongside a 662-word
resume is padded out to the resume's length. The same arithmetic explains the
full attribution exactly — 15 variants measured 4,659 ms against 15 x 291 ms.

**Decision** Cache the job description's vector (`_jd_vector`, an `lru_cache`)
and keep it out of every batch. It does not change while the user edits, so
re-embedding it was waste on top of waste.

**The check that made it safe** An optimisation that silently changed scores
would be worse than the cost it saved. Batched and split embeddings agree to
**6.7e-08** and the resulting cosine to six decimal places — attention masking
makes padding irrelevant, as it should. Asserted in
`test_splitting_the_jd_out_of_the_batch_does_not_change_the_vectors`.

**Measured** 590 ms -> **291 ms**, median of five consecutive toggles.

**Files** `src/rba/resume/attribution.py`, `src/rba/resume/merge.py`,
`tests/test_live.py`, `tests/test_merge.py`

---

## c30e9c8 — fix: check every credential before a run spends anything   2026-10-29

**Problem** A real run produced this, and nothing more useful:

```
Jev unavailable; using the heuristic fallback.
summarize failed for vd_p2: 401: Invalid API Key
```

One repository named, partway through, after minutes of work.

**Investigation** The key in `.env` answered `HTTP 200` when tested directly, so
it was not the key that failed. Reading the plumbing: `pipeline.py` passes
`api_key=groq_key or None`, so a *blank* box correctly falls back to `.env` —
but a **non-empty wrong key silently overrides a working one**. The other
repositories had not succeeded; they came from the SQLite response cache and
never touched the network, which is what disguised a total failure as a
per-repository one.

**Decision** One free metadata call per service before anything else, reporting
the service, the verdict, **and which source the key came from**. The last part
is the actual fix — the failure was never "a key was wrong", it was "nobody said
which key was being used".

**Measured** Against a deliberately invalid key:
`Groq: failed (from argument) — the key was rejected (401). This key overrides
any other source, so a stale value here will mask a working one.`

**Honest gap** The `Jev unavailable` line could not be reproduced locally;
`available` is `True` here with a 107-character key and TypeSafe answers 200.
Preflight will name the real cause on the next run rather than this document
guessing at it now.

**Files** `src/rba/preflight.py`, `src/rba/resume/pipeline.py`

---

## 3238c90 — feat: rephrase every point, then measure whether it actually helps   2026-10-29

**Problem** The artifact this project is measured against offers "best present
work" — rewriting a line so it scores better. Doing that naively means asking a
model to both rewrite *and* estimate the benefit, which reintroduces exactly the
invention the project exists to prevent.

**Options** (a) one call per bullet — simplest, but multiplies requests by the
number of lines; (b) one call per project returning K rewrites for each of its
bullets; (c) one call for the whole document — cheapest, but stacking degrades
past roughly eight outputs.

**Decision** (b). An orchestrator fans out one structured call per project; each
worker returns K=3 rewrites for every point it owns.

**The part that matters** The model never scores anything. Every rewrite is
substituted into the document and re-scored with `rescore()`, so the number
beside a suggestion is a measurement. And the rephrase-only rule is enforced in
code — `reject_reason()` — not requested in the prompt, because a prompt is a
request and a check is a guarantee.

**Measured** 4 projects, **6 requests, 7,018 tokens, 79 s**. 45 rewrites kept,
0 rejected.

Zero rejections is the right outcome only if the guard actually fires, so it was
poisoned along every route against a real bullet:

| poison | verdict |
|---|---|
| "cut review time by 40%" | rejected — number absent from evidence |
| "serving 2 million requests per day" | rejected — number absent from evidence |
| "on Kubernetes with Terraform and Kafka" | rejected — technology absent from evidence |
| "doubled throughput versus the baseline" | rejected — outcome not in evidence |
| "Led a team of fifteen engineers" | rejected — does not restate the original |
| an honest rephrase | **kept** |

**Reported honestly** Gains are small: **+0.44 at best**. Rephrasing one line of
forty barely moves a fused score, and saying otherwise would be the same
overclaim the rest of the project refuses.

**What did not work first** Three rejections of legitimate rewrites, all fixed:
the technology check compared canonicalised names against surface strings, so an
honest rewrite using the evidence's own word was rejected; its word-boundary
rule let `Redis-cached` through while mishandling `Postgres.`; and comparing
impact language term-by-term rejected "reduced" -> "accelerated".

**Files** `src/rba/nodes/variants.py`, `src/rba/prompts/variant.v1.md`,
`tests/test_variants.py`

---

## cec4763 — fix: the LaTeX export had never once been compiled   2026-10-29

**Problem** The `.tex` export was shipped on the assumption that generating
plausible LaTeX is the same as producing a working document. There was no
`pdflatex`, `xelatex`, `tectonic` or `latexmk` on the machine, so nothing had
ever proved it built. This is the project's own rule broken by its author: a
claim asserted without the evidence that would support it.

**Decision** Install tectonic — one self-contained binary rather than a 4 GB TeX
Live — and make a test compile the file and assert a PDF comes out.

**What compiling found** Ten defects. Four were visible by reading: a raw `|` in
a tabularx cell, URLs unlinked while `hyperref` was loaded, inconsistent heading
case, `X` columns not ragged-right. **Six were invisible without an engine:**

- `\bfseries\scshape` was silently substituted — `T1/lmr/bx/sc` does not exist
- `show_numbers=False` emitted em dashes with no slot in T1, so eighteen
  characters vanished from the page with no error
- `\parindent` pushed full-width tables past the margin (3 overfull boxes)
- the six-column table squeezed its only flexible column below its own header
- `INSTITUTE/COLLEGE` is one unbreakable token — TeX will not hyphenate at `/`
- an empty document produced **no PDF at all**

That ratio is the lesson: two thirds of the defects were undetectable by the
method previously used to call the feature done.

**A second inconsistency** At 10pt this ran to two pages while fpdf2 fitted one
— the same document, two different lengths. fpdf2 gets there by compiling and
shrinking until it fits, which is impossible in LaTeX here because Streamlit
Community Cloud has no TeX engine. So the density is *chosen* rather than
discovered: `extarticle` at 8pt, against fpdf2's measured 8.55pt effective size.

**Measured** Both styles compile to **1 page, 0 warnings** — no overfull or
underfull boxes, no font substitutions, no missing characters. Text extracted
back out of the compiled PDF confirms the content survived.

**Files** `src/rba/resume/render/latex.py`, `tests/test_latex.py`

---

## 5ca8ad0 — feat: a point is now a control, not a form field   2026-10-29

**Problem** The per-point deltas were exact and had been treated as finished on
that basis. But they sat behind a *Rescore* button: clicking a line did nothing
until you pressed it. Exact and useful are different things, and a measurement
you wait for is a report, not a control.

**Decision** Put everything that reacts to a click — score, diagnostics,
preview, checkboxes — inside one `st.fragment`. Streamlit reruns the whole
script on any widget change, so a toggle was redrawing the sidebar, the uploader
and the download buttons before the number moved. A fragment scopes that rerun.

**Caught while testing** The first version was wrong in a way that looks right:
the score lagged exactly one click behind. Streamlit keeps widget values in
`session_state`, and `editor()` only copied them onto the bullets when it ran —
*after* the score had been computed from the stale document. `sync_from_widgets()`
now applies widget state first.

The test asserts the behaviour rather than the plumbing: untick a line, and the
score must differ, with no Rescore press in between. That is what caught it.

**The honest compromise** A full delta refresh costs 4.8 s for fifteen lines, so
it cannot run on every click. The figures keep their last exact values, are
marked `~` once the document has moved away from what they measured, and refresh
on request. Nothing shown is ever an approximation — the mean-pool shortcut
remains rejected for getting the *sign* wrong on 6 of 20 bullets.

**Measured** **291 ms** median over five consecutive toggles, against 590 ms
behind a button.

**Files** `app.py`, `tests/test_app.py`

---

## 7d085fd — docs: TRACE.md   2026-10-29

Everything above this section, reconstructed; everything in Session 2, recorded
as it happened. The distinction is marked because a trace that invents its own
history fails the same rule as a resume that invents an achievement.

---

---

# Cross-cutting lessons

Each of these is asserted only where the history supports it, with the commits
cited.

## 1. Plausible designs die of measurement, not of argument

Three times a design that was obviously right was killed by a short experiment,
and in each case the author records having been *committed* to it beforehand.

| Design | Why it was obvious | What killed it | Commit |
|---|---|---|---|
| mean-pool attribution approximation | **4,253× faster** (1.15 ms vs 4.9 s for 20 deltas) | Pearson 0.643, but **sign wrong on 6 of 20 bullets** | `2f96654` |
| ESCO skill taxonomy | EU standard, ~13,900 skills vs O*NET's 177, multilingual, maintained | **no results** for kubernetes, docker, pytorch, tensorflow, aws, terraform, kafka; `react` → "react to emergency situations in a live performance environment" | `4f4f6c1` |
| Reciprocal Rank Fusion (k=60) | the reflex choice for hybrid retrieval | Bruch et al.: weighted sum wins on all five datasets (0.454 vs 0.425 MS MARCO; 0.542 vs 0.514 NQ) **and RRF's tuned parameters do not transfer out of domain** | `bb3972f` |

The common structure is that the obvious design fails on a metric *other than
the one it was chosen for*. The approximation was chosen for speed and fails on
sign. ESCO was chosen for authority and fails on coverage. RRF was chosen for
robustness and fails on transfer. **Choosing the right metric is the whole
skill**, and in the attribution case the right metric (sign agreement) follows
from the user-facing question (keep or delete) rather than from convention
(correlation).

Two of these are recorded in **module docstrings**, not just commit messages —
`attribution.py` and `normalise.py` carry the rejection tables permanently,
explicitly *"so nobody re-proposes it"*. That is the correct half-life for a
negative result: a commit message is read once, a docstring is read by whoever
next edits the module.

## 2. Measurement overturns stated belief, including the author's own

`evals/budget.md` is the clearest case: a measurement study that **contradicted
three claims in its own plan** (requests binding; prompt needs restructuring;
prefix caching will carry the budget), and whose conclusion is *"110 repos was a
straw man I invented."*

The same pattern elsewhere:

- `b5b76b1` — `temperature: 0` was assumed to mean reproducible. Two runs showed
  ATS scores moving **up to 24 points** (`balanced` 59.1 → 34.8) while
  verification was **bit-identical** (same 102/111 verdicts, same 6 rejections,
  same reasons). The better-looking run was not kept.
- `9a65c4e` — "the model cannot judge these repos" was assumed to be a model
  problem. Scores of 0.13–0.35 across 93 repos were *"the right answer to a
  question containing no evidence"*; adding evidence moved them to 0.09–0.81.
- `e962065` — the published 150 ms rescore figure was corrected to **582 ms**
  once measured on a real 662-word resume rather than a smaller test document.
- `3656473` — the calibration reference set was assumed meaningful; **462 of 484
  pairs were cross-family nonsense**, which put a mediocre 0.6 at the 92nd
  percentile.

## 3. The dangerous bug is the one that produces plausible output

Nearly every serious bug in this history failed *silently*, producing something
a reviewer would read as normal.

| Bug | What it looked like | What it was | Commit |
|---|---|---|---|
| citation `#0` parsed as the number 0 | a shorter resume | **19 legitimate bullets dropped in one run, 8 in the next** | `fe93e24` |
| `_strictify` stripping a field named `title` | a slow planner | every plan step failed validation; **nine minutes of retries**, now 3 s | `fe93e24` |
| rate-limit override with no expiry | a slow judge | window pinned forever; *"a three-second judge call became one that never finished"* | `fdbefc4` |
| version mask `\d+\.\d+` | a verifier that passed | swallowed `0.81 accuracy` and `49.8%` — the exact numbers it polices | `01674c3` |
| prose scanned for technology tokens | strict verification | **39 of 48 real claims rejected** over `fallback.` and `scripts.` | `01674c3` |
| `state \|= update` overwriting `errors` | a clean audit trail | *"reported 'no bullets dropped' on a run that dropped five"* | `b570e64` |
| `head -1` SIGPIPE in the secret scanner | a passing scan | *"reported 'no commits yet' on a repo with six commits"* | `beff839` |
| emptiness judged from absent `size_kb` | a strict filter | **all 98 repos rejected** | `789a108` |
| `as_text()` not traversing tables | a normal score | scorer *"silently stops seeing the skills and education sections"* | `e962065` |
| embeddings unchunked | *"Error running app"*, blank page | OOM at **4,133 MB** for 152 texts against a ~1 GB cap | `667895e` |
| 401 on every summary | an empty resume | an expired key, with no reason given | `667895e` |
| GitHub token accepted but unused | `Import failed: RuntimeError` | worked locally via `.env` fallback; failed everywhere else | `0da531f` |

Two structural consequences the project drew from this:

- **Guards that destroy good output are the worst kind of bug in a verification
  system**, because the symptom (fewer claims, shorter resume) is
  indistinguishable from the system working well. Four separate entries above
  are the project's own safety machinery misfiring.
- **A detector must be seen to go red.** `beff839` was verified *"by planting a
  credential-shaped string and confirming a non-zero exit"*; `401b381` refused to
  weaken the scanner to accommodate a test fixture; `73256dc`'s end-to-end test
  uses a **deliberately lying fake generator** emitting *"handling 9,999 requests
  per second"* from claims containing no such number.

## 4. Test in the deficient environment, not the comfortable one

Four separate commits fix a bug that only exists where the developer's machine
is not:

| Commit | Environment reproduced |
|---|---|
| `98e0663` | *"cloning to a fresh directory and running the README's install path end to end"* |
| `1b4cfee` | *"a venv containing exactly what the workflow installs: 31 passed"* |
| `0da531f` | *"with the environment variables removed, so the explicit token is the only possible source"* |
| `520956a` | ruff `--no-cache`, because *"my local ruff passed on a stale cache; CI, which has none, caught it"* |

In every case the comfortable environment supplied a fallback — an ambient
dependency, an installed package, a `.env`, a warm cache — that masked the
defect.

## 5. Probe the platform before building against it

- `6a6cf6c` — the live model list, the real rate-limit headers, the reasoning
  trap and the structured-output/tool-use exclusion were all found by probing,
  and together they determine the graph's node boundaries.
- `789a108` — enumerating both MCP endpoints (46 tools vs 13, write tools vs
  none) determined the default endpoint and a startup gate.
- `b570e64` — *"A creation probe settled that in under a minute rather than at
  the end of the work"*: free cpu-basic cannot host Gradio or Docker Spaces.
- `0f027ef` — Streamlit's dependency-file priority maps `pyproject.toml` to
  Poetry, which would have failed the install; `uv.lock` outranks it.
- `0767667` — the Hub's 60-character `short_description` cap is only enforced at
  publish time, so it is now checked locally.

The economics are consistent: minutes of probing against hours or days of
building the wrong thing.

## 6. Determinism is a design property, deliberately placed

The split that `b5b76b1` measured is not luck. It was engineered:

- `src/rba/nodes/fetch.py` — *"No model touches anything in this module."*
- `src/rba/nodes/verify.py` — *"Code, not a prompt."*
- `src/rba/ats/*` — *"It does not call an LLM… scoring is arithmetic you can
  check by hand."*
- `src/rba/resume/ingest.py` — *"Deliberately deterministic."*
- `src/rba/scoring.py` — *"Being deterministic is the point"*, because the
  planner quotes ATS gains from the same arithmetic.

The language model is confined to three jobs — summarising facts into claims,
writing prose from verified claims, and parsing documents — and each is bounded
by a deterministic check downstream. The consequence is quotable: **what the
system says may vary; what it is permitted to say does not.**

A second-order benefit: because scoring is deterministic and model-free, **exact
leave-one-out attribution is affordable**. The determinism decision made in
`01674c3` and generalised in `bb3972f` is what makes the headline feature of
`2f96654` possible at all. These are usually presented as separate concerns;
here one directly enables the other.

## 7. Report the unflattering number

- `0c832f7` — split one faithfulness figure into claim faithfulness **1.00** and
  whole-summary faithfulness **0.83**, when a single averaged number would have
  been ~0.9 and unremarkable. *"That gap is the honest part."*
- `b5b76b1` — published a 24-point variance and the 34.8 run.
- `e962065` — corrected 150 ms to 582 ms, upward in cost.
- `evals/budget.md` — labelled the 768 cached tokens *"unexplained; they are not
  a strategy"*, and attached *"documented but not observable"* to every TPD
  percentage.
- `3656473` / `build_calibration.py` — stated the PJB finding that cross-domain
  heterogeneity exceeds any module upgrade, i.e. that the calibration *"fixes
  the scale, not the ranking"*.
- `src/rba/ats/fusion.py` — labelled its own channel weights *"a judgement call
  and… reported as such in the UI"*, bounded only by the [0.6, 0.8] dense-share
  band from Bruch et al.
- `src/rba/ats/__init__.py` — refused to claim ATS reproduction at all:
  *"No major ATS shows a candidate a 0-100 number, so neither should we without
  saying whose number it is."*

## 8. Apply your own rule to your own work

The sharpest thread in the history. The project's rule is "never present an
invented number as a measurement". Applied inwards, it cost the author:

- the hand-weighted keyword table (`langgraph: 3.0, python: 1.0` — *"because I
  decided so"*), replaced by BM25 IDF derived from a corpus (`bb3972f`);
- the seven-bar 100-point rubric, demoted from a score to unweighted advice
  (`e962065`), because *"Nobody measured that a missing phone number is worth
  exactly six points"*;
- the `63/100` presentation, replaced by a percentile against a stated reference
  set, because *"it looks interpretable, which is worse"* (`3656473`);
- the 110-repo budget crisis, retracted as *"a straw man I invented"*
  (`evals/budget.md`);
- the 150 ms latency claim, corrected to 582 ms (`e962065`);
- the better-looking 59.1 ATS run, published alongside the 34.8 (`b5b76b1`).

Each of these made the project *look* less impressive and made its claims true.
That is the trade the whole repository is organised around.

---

# What this document could not establish

In the interest of the rule this project runs on, the gaps:

- **`ceff6b8`** (README rewrite) has no commit message body. Its editorial
  rationale — leading the quality section with the project's own failures — is
  **reconstructed** from the content of the diff.
- **`1e849d7`** (the first commit) has no body either. Its reasoning is
  inferred from the subject line and from `README.md`'s later "Safety" section,
  which cites it. The inference is unambiguous but it is an inference.
- **`fdbefc4`** records the symptom (*"a three-second judge call became one that
  never finished"*) but **no post-fix timing** was recorded. The fix is covered
  by `tests/test_limiter.py`, not by a measurement.
- Several small commits (`98e0663`, `e2aaa2e`, `ac4b5f6`, `520956a`, `91164c3`,
  `2d86702`, `401b381`) record **no numbers at all** beyond failing-test counts.
  Nothing has been supplied for them.
- The **768 cached prompt tokens** in the budget run (3.2% of prompt) are
  unexplained in the source and remain unexplained here. `evals/budget.md`
  labels them *"not a strategy"*; no mechanism was identified.
- The **ordinal band thresholds** (0.80 / 0.60 / 0.40) in
  `src/rba/ats/fusion.py` are documented as mirroring Greenhouse's and Workday's
  ordinal grades, but **no derivation of those specific cut points is recorded**.
  They appear to be a judgement call; unlike the channel weights, this is not
  explicitly labelled as one in the source.
- The **22-job background corpus** and the **synthetic calibration reference
  set** are both described as small and synthetic by their own modules. Their
  construction is documented; their adequacy is not measured against anything
  external, and `build_calibration.py` says as much.
- The README stated **185 tests** when this was written; the suite now reports
  **439 passing**, confirmed by `pytest -p no:warnings` at the close of
  session 2.

---

*Compiled from 44 commits — the first 38 reconstructed from their diffs and
the module docstrings, the last six recorded as the work happened — together
with `README.md`,
`run.md`, `evals/budget.md` and `evals/budget.json` in this repository. Every
number above is quoted from one of those sources. Where no source recorded a
number, this document says so.*
