# Resume Builder Agent

Reads a GitHub account, turns each repository into **evidence-backed facts**, and
builds role-targeted resumes plus an improvement plan where every item points at a
real file or commit.

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
git clone <this repo> && cd resume-builder-agent

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
# Docker Desktop must be running first.
docker compose build
docker compose run --rm rba doctor
```

The compose file mounts `/var/run/docker.sock` because the agent launches
GitHub's MCP server as its own short-lived container.

### Verify the install

```bash
pytest                    # whole suite, offline, no credentials needed
rba config                # shows which model each graph node routes to
rba doctor                # live credential + reachability check
./scripts/scan_secrets.sh # confirms no key is tracked by git
```

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

The agent talks to GitHub through the **official GitHub MCP server**
(`ghcr.io/github/github-mcp-server`), launched in `--read-only` mode. Read-only is
a hard gate in that server: write tools are stripped from the toolset even if
something requests them.

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

**The binding constraint is 8,000 tokens per minute**, not the 30 requests per
minute. One full evidence-linked repo summary costs ~2,074 tokens, so the ceiling
is roughly **3.8 repo summaries per minute**.

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
python evals/run_eval.py          # traces to the LangSmith project
```

---

## Layout

```
src/rba/
  config.py       config.yaml + .env loading, tier routing
  limiter.py      per-model TPM/RPD buckets, header reconciliation
  groq_client.py  retries, strict JSON, the reasoning-trap guard
  jev_client.py   TypeSafe System One: noul / choice / score
  schema.py       Pydantic -> Groq strict JSON schema
  cache.py        SQLite response cache
  github_tools.py MCP client (docker | remote | rest)
  scoring.py      deterministic ATS rubric
  nodes/          one module per graph node
  roles.yaml      role definitions and keyword tables
  prompts/        versioned prompt files
evals/            faithfulness + G-Eval judges, golden fixtures
tests/            offline suite with fakes
scripts/          secret scanner
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
