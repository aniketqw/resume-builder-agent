# Resume Builder Agent — v1

> **This is v1, the Streamlit app.** There is also a [v2](../v2) — a FastAPI
> service with a browser workbench over the same engine. See the
> [root README](../README.md) for what differs and which to use.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://aniketqw-resumebuilder.streamlit.app)

**[Try the resume scorer →](https://aniketqw-resumebuilder.streamlit.app)** —
paste a resume and a job description, see the score broken into three channels
with the formula printed on the page. No sign-up, no API key.

> The app sleeps after 12 hours without traffic. A cold link shows a
> "Yes, get this app back up!" button — click it and give it about 30 seconds.

Starts from the resume you already have, reads a GitHub account, turns each
repository into **evidence-backed facts**, and offers them as candidate lines
you can accept one at a time -- each showing exactly what it is worth.

Upload a PDF and it is parsed into its real structure (academic tables, dated
entries, `Tech:` lines), re-rendered as a one-page PDF, LaTeX or ATS-safe text,
and scored. Your resume is the baseline and is never rewritten: generated lines
arrive switched off.

The governing rule: **the agent never invents an achievement or a number.** A claim
that cannot be traced to repository evidence is not softened or hedged — it is
dropped and replaced with a `[MEASURE: <command>]` task telling you how to go and
measure it yourself.

Built with LangGraph + LangChain + Groq, with TypeSafe/Jev for routing decisions
and LangSmith for tracing and evaluation.

---

## Quickstart

### Option A — local virtualenv (works right now)

```bash
git clone <this repo> && cd resume-builder-agent/v1

# Python 3.11+ is required.
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

cp .env.sample .env     # then fill in the four keys (see "Credentials" below)
rba doctor              # verifies every credential and provider
```

`uv` is faster if you have it:

```bash
uv venv --python 3.11 .venv && VIRTUAL_ENV=.venv uv pip install -e ".[dev]"
```

### Option B — Docker

```bash
# Docker Desktop must be running first. Run these from v1/ -- there is a
# separate v2/docker-compose.yml for the web app.
docker compose build
docker compose run --rm rba doctor
```

Note this image builds the **command-line tool**, not a server: no port is
exposed and there is no UI. For a dockerised web app, that is [v2](../v2).

The compose file mounts `/var/run/docker.sock` because the agent launches
GitHub's MCP server as its own short-lived container.

### Verify the install

```bash
pytest                  # 495 tests, offline, no credentials needed
rba config                # shows which model each graph node routes to
rba doctor                # live credential + reachability check
./scripts/scan_secrets.sh # confirms no key is tracked by git
```

### Use it

```bash
rba run <github-username> --dry-run     # projects the token spend, spends nothing
rba run <github-username>               # full run; pauses for your review
rba resume-run --thread default         # continue after reviewing
python evals/run_eval.py --no-geval     # quality harness
```

Output lands in `out/`: `report.md`, one `resume_<role>.tex` per role, and PDFs
if a LaTeX engine is installed.

---

## Credentials

Put these in `.env` (gitignored; never commit it). `.env.sample` lists the names.

| Variable | What it is | Where to get it |
|---|---|---|
| `GROQ_API_KEY` | LLM for summaries, resumes, planning | console.groq.com/keys |
| `TYPESAFE_API_KEY` | Jev System One, for routing decisions | console.typesafe.ai/keys |
| `LANGSMITH_API_KEY` | Tracing and evaluation | smith.langchain.com |
| `GITHUB_PERSONAL_ACCESS_TOKEN` | Read-only repo access | see below |

### GitHub access

The agent talks to GitHub through the **official GitHub MCP server**. The default
is GitHub's hosted endpoint at `https://api.githubcopilot.com/mcp/x/repos/readonly`,
which needs no Docker.

That exact path is deliberate. Enumerating both endpoints:

| Endpoint | Tools | Write tools |
|---|---:|---|
| `/mcp/` | 46 | `create_or_update_file`, `delete_file`, `push_files`, `create_repository`, `fork_repository`, `merge_pull_request` |
| `/mcp/x/repos/readonly` | 13 | none |

The agent refuses to start if a write tool shows up in the toolset. Set
`github.mcp.mode: docker` in `config.yaml` to run `ghcr.io/github/github-mcp-server`
locally with `--read-only` instead, which is also a hard gate.

Two behaviours worth knowing:

- `search_repositories` (there is no `list_repositories`) **excludes forks**, which
  is why MCP returns 98 repos where REST returns 113. It does return all private
  repos. Forks are excluded by rule anyway.
- There is **no `get_repository_tree`** on the hosted endpoint — it lives in the
  `git` toolset, which only the local container exposes. Walking a tree with
  `get_file_contents` costs one call per directory; across ~100 repos that was
  ~500 calls, after which the endpoint started returning empty trees and the
  session timed out. Trees therefore come from a single REST call
  (`github.mcp.tree_via_rest`, on by default). Sessions are also rebuilt
  automatically when they die mid-sweep.

Three ways to authenticate, in order of preference:

1. **Fine-grained PAT** (least privilege): repository permissions
   `Contents: Read-only` + `Metadata: Read-only`.
2. **Classic PAT** with `repo` scope. Works, but classic tokens have no read-only
   repo scope — `repo` grants write too, which is why `--read-only` matters. The
   server reads your `X-OAuth-Scopes` header and hides tools you lack.
3. **Already using `gh`?** `GITHUB_PERSONAL_ACCESS_TOKEN=$(gh auth token)`.

To run against a different account, change `GITHUB_PERSONAL_ACCESS_TOKEN` and
`GITHUB_USERNAME`. Nothing else is account-specific.

Set `github.mcp.mode: remote` in `config.yaml` to use GitHub's hosted endpoint
(`https://api.githubcopilot.com/mcp/`) instead of a local container, or
`mode: rest` for a dependency-free REST fallback.

---

## How it works

```
connect_github     MCP, read-only
list_repos         all repos for the account
select_repos       Jev scores every repo for "claimable"; keep the top N
fetch_facts        deterministic: README, tree, languages, commits, tests, CI
safety_scan        flags .env / tokens / cookies committed in your repos
summarize_repo     strict JSON, one repo per call, rate limited
verify_claims      CODE, not a prompt: unsupported claims are dropped
score_roles        deterministic 100-point ATS rubric
build_resumes      bullets drawn only from verified claims
compare            table built in Python; model writes a short rationale
plan_improvements  evidence-linked pathways with hours and ATS gain
human_review       pauses for your edits (LangGraph interrupt)
report             Markdown + LaTeX + PDF
```

State is checkpointed to SQLite, so a run that stalls on a rate limit resumes
exactly where it stopped rather than starting over.

### Why claims survive or die

`verify_claims` is plain Python. Every digit string and every technology name in a
generated claim must appear in that repository's fact bundle. If it does not, the
claim is demoted — it never reaches the resume. Forks and single-commit repos are
flagged "not claimable" by commit-share analysis and excluded entirely. The
large model that writes the improvement plan never sees raw repository content,
only claims that already passed verification.

---

## Operating inside the free tier

These numbers were measured against the live API, not read off a pricing page.

**Tokens bind; requests do not.** Measured cold over 10 real repositories
(`python scripts/measure_budget.py --repos 10 --cold`, full write-up in
[`evals/budget.md`](evals/budget.md)): **1 request and 3,058 tokens per repo**,
of which 79% is prompt. At one call per repo, even 110 repos is 110 requests --
11% of the 1,000/day limit. It is the token budget that runs out.

Two findings from that measurement worth stating plainly:

- **Groq prefix caching never fired for us.** Identical system prefixes at 771,
  1,461 and 2,841 tokens, and the real 631-token prompt sent five times back to
  back, all returned `cached_tokens: 0`. Any plan that relies on cached tokens
  being exempt from rate limits is unsupported on this account.
- **There is no daily-token header.** Groq exposes only
  `x-ratelimit-limit-requests: 1000` and `x-ratelimit-limit-tokens: 8000`. The
  200,000/day figure is documented but not observable, so it cannot be verified
  from a response.

The 8,000 tokens-per-minute ceiling is real and is what produces the 429
backoffs during a cold run.

Measured on a real account (113 repos, 98 non-fork):

| Stage | Work | Cost | Time |
|---|---|---|---|
| list + signals | 98 repos, 55 with a README | 0 LLM tokens | ~20 s |
| Jev selection | 79 repos scored, 25 selected | 63,011 input tokens, **$0.0027** | ~10 s |
| fetch facts | 25 repos, 23 READMEs, 11 test suites | 0 LLM tokens | ~90 s |
| summarize | 25 repos, 151 claims | ~52,000 Groq tokens | ~9 min |
| verify | 139/151 verified, 7 rejected | 0 tokens (pure Python) | <1 s |
| resumes + plan | 5 resumes, 10 plan steps | ~12,000 Groq tokens | ~2 min |
| re-run (cached) | same 25 repos | **0 tokens, 0 s** | 100% cache hit |

Only the summarize/resume/plan stages spend Groq tokens. Everything upstream —
including the decision about which repos are worth summarizing at all — is free
or costs fractions of a cent, which is the entire point of the split.

What the implementation does about it:

- **Per-model token buckets.** Limits are per-model, so routing a step to a
  different tier also buys throughput.
- **Headers are truth.** Token estimates are reconciled after every call against
  `x-ratelimit-remaining-tokens`; `retry-after` is honoured exactly on a 429.
- **Jev takes the cheap decisions.** Scoring ~100 repos for "is this claimable"
  runs on TypeSafe's Jev instead of Groq: output tokens are free, input is
  $0.042/Mtok, and it draws on a separate quota. Roughly $0.002 for the sweep.
- **SQLite cache** keyed by `(repo, head SHA, prompt version, model)` — an
  unchanged repo costs zero tokens on a re-run.
- **Clean daily stop** at 90% of the daily budget, with a checkpoint.
- `--dry-run` prints projected token spend per node before spending anything.

### Two API behaviours worth knowing

1. **The reasoning trap.** `openai/gpt-oss-*` models spend the completion budget
   on an internal `reasoning` field. With a small `max_tokens` they return
   `content: ""` and HTTP 200 — a success that looks like a model failure. Every
   request here forces `reasoning_effort: "low"` and a floor on `max_tokens`,
   and there is a regression test for it.
2. **Structured output excludes tool use and streaming.** A Groq request may use
   `response_format: json_schema` *or* tool calling, never both. That is why
   `fetch_facts` (tools) and `summarize_repo` (strict JSON) are separate nodes.

Model IDs live in `config.yaml`, never in code — Groq's lineup moves fast.
`rba doctor` fails loudly if a configured model has disappeared.

---

## Quality: how we know the summaries are any good

Three layers, built on published metrics rather than vibes.

1. **Groundedness gate (blocking).** RAGAS-style faithfulness: decompose each
   bullet into atomic claims, then a binary supported/unsupported verdict per
   claim against that repo's evidence. Any unsupported numeric, scope, impact or
   technology claim rejects the bullet. Surviving claims keep their evidence ID.
2. **Summary quality (tracked).** G-Eval over the four SummEval dimensions —
   coherence, consistency, fluency, relevance — using probability-weighted
   scoring, randomized option order and n≥3 samples to blunt known LLM-judge bias.
3. **Regression suite.** Code Consistency and Information Retrieval coverage over
   a frozen held-out set of real repositories.

Run it:

```bash
python evals/run_eval.py                # faithfulness + G-Eval, traces to LangSmith
python evals/run_eval.py --no-geval     # faithfulness only, much cheaper
```

### Measured quality

From `evals/run_eval.py`, judged by `gpt-oss-120b`:

| Metric | Result | Blocking? |
|---|---:|---|
| **claim faithfulness** — the text that reaches a resume | **1.00** | yes |
| **number precision** — numbers traceable to evidence | **1.00** | yes |
| whole-summary faithfulness — *includes unverified prose* | 0.83 | tracked |

That gap is the honest part. `verify_claims` governs the **claims**, and those
are fully grounded. The descriptive `problem` and `approach` fields are framing
and are *not* verified, so they can contain fair-sounding inference — "the
original model does not account for distance between data points" is a
reasonable reading of a repository's purpose and still nowhere in its README.
The two are measured separately rather than averaged into one flattering
number, and only claims reach a resume.

### Reproducibility, measured

Two independent runs over the same repositories, both at `temperature: 0`:

| Half of the pipeline | Result |
|---|---|
| Verification (`verify_claims`) | **Identical.** Same 102/111 verdicts, the same 6 rejections, the same reasons. |
| Generation (`build_resumes`) | **Variable.** ATS scores moved by up to **24 points** (`balanced` 59.1 → 34.8). |

That split is the design working as intended, not an accident: the half that
decides what may be *said* is deterministic Python, so it cannot drift. The
half that decides *how* to say it is a language model, and Groq's `gpt-oss`
models are not reproducible even at temperature 0.

The practical consequence is worth stating plainly: **treat a single ATS score
as indicative, not precise.** What is stable is which claims are allowed
through — and that is the property the project actually promises.

### What judging actually changed

The first live run was judged against the real repositories, and the output was
wrong in ways the tests had not caught:

- The technology check scanned claim prose for "technology-shaped" tokens and
  **rejected 39 of 48 real claims** over words like `fallback.` and `scripts.` —
  a trailing period made them look like package names.
- The impact regex matched bare nouns, so "MongoDB is used for **saving** and
  searching documents" was flagged as an unmeasured performance claim.
- Version masking used `\d+\.\d+`, which swallowed `0.81 accuracy` and `49.8%` —
  precisely the numbers the module exists to police.
- A claim about `tcell` cited a real `go.mod` line about `goccy/go-json`. The
  quote existed, so it passed; it supported nothing in the claim.

After those fixes, on 8 repositories: **48 of 51 claims verified, and all three
rejections were correct.** The best of them: the model wrote *"losing 745 out of
1484"*, having computed `1484 − 739` itself. The evidence contains 739 and 1484
but never 745, so the claim was refused. That is the system working.

Three later bugs were each caught by running the thing rather than reasoning
about it, and each had been **silently destroying good output**:

- `_strictify` stripped `title`, `default` and `examples` as noisy JSON-Schema
  annotations — including when they were the model's own **field names**. The
  plan schema lost its `title` field, so every generated step failed validation
  and the planner burned nine minutes on retries. One plan call now takes 3 s.
- The provenance check read the digits in claim-id citations as invented
  numbers: the generator appends `(AiAgnoSample#0, #1)` — and, unprompted, the
  full-width `【AOA_Controller_ty#0】` — so `#0` parsed as the number 0. This
  dropped **19 legitimate bullets in one run and 8 more in the next**. After
  the fix, all five resumes build with zero false drops and the balanced
  resume went from 32.9 to 61.2/100.
- The rate-limit header override never expired. `x-ratelimit-remaining-tokens`
  is a snapshot of the current minute, but it was cached forever, so once a
  response reported a low figure the limiter believed it had no budget for the
  life of the process and slept a full minute before every call. A 3-second
  judge call became one that never finished.

---

## Layout

Everything below lives under `v1/`; `v2/` is a sibling with its own copy of the
engine. See the [root README](../README.md).

```
src/rba/
  config.py       config.yaml + .env loading, tier routing
  limiter.py      per-model TPM/RPD buckets, header reconciliation
  groq_client.py  retries, strict JSON, the reasoning-trap guard
  jev_client.py   TypeSafe System One: noul / choice / score
  schema.py       Pydantic -> Groq strict JSON schema
  cache.py        SQLite response cache
  github_tools.py MCP client (docker | remote | rest)
  scoring.py      keyword/number/verb detectors, reused by diagnostics
  ats/            BM25 + ONNX dense + skills, convex fusion, calibration
  resume/
    document.py   Bullet/Entry/Table/KeyValues/Header -- the editable model
    ingest.py     PDF -> structure, deterministic, no LLM
    merge.py      generated lines as opt-in candidates
    diagnostics.py  advice, deliberately carrying no invented weights
    attribution.py  exact leave-one-out per-bullet impact
    render/       plain, html, pdf (fpdf2), latex
  nodes/          one module per graph node
  roles.yaml      role definitions and keyword tables
  prompts/        versioned prompt files
evals/            faithfulness + G-Eval judges, golden fixtures
tests/            offline suite with fakes
scripts/          secret scanner, token-budget measurement
fixtures/         real extracted resume text, so parsing is tested offline
```

---

## Safety

- `.gitignore` containing `.env` is the **first commit** in this repository's
  history, before any other file was ever staged.
- `scripts/scan_secrets.sh` greps the tracked tree *and* full git history for
  Groq, LangSmith, TypeSafe and GitHub token patterns. It runs in CI and gates
  every push.
- The GitHub MCP server runs `--read-only`.
- Private repository contents are never logged.
- `safety_scan` reports secrets it finds committed in *your* scanned repos, which
  is often the single most valuable output of a first run.
