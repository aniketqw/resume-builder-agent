# Resume Builder Agent — v2

A FastAPI service and a browser workbench over the same scoring engine as [v1](../v1).
v1's interface is a Streamlit page; this one is a two-pane workbench with a live
preview of the resume and a measured score delta beside every line.

See the [root README](../README.md) for what differs between the two and why.

---

## Run it

### Locally

```bash
cd v2
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

uvicorn rba_api.app:app --reload --port 8000
```

Open <http://localhost:8000>.

Two notes worth having before you start:

- **Use a separate virtualenv from v1.** Both packages install a console script
  called `rba`, so sharing one environment means whichever was installed last wins.
- **The defaults are tuned for a 1 GB container, not your laptop.** `RBA_ONNX_THREADS`
  defaults to 1 because that is right for Streamlit Community Cloud. Locally:

  ```bash
  RBA_ONNX_THREADS=4 RBA_EMBED_BATCH=16 uvicorn rba_api.app:app --port 8000
  ```

  Measured on one 512-token document: **294 ms at one thread, 84 ms at four.**

### With Docker

```bash
cd v2
docker compose up --build
```

The first build downloads the embedding model (~130 MB) and bakes it into the image,
so expect a few minutes once, and no model download ever again. After that the
container scores with **no network access at all**.

`.env` is optional — `docker compose` is configured with `required: false`, and
uploading and scoring a resume need no credentials whatsoever. You only need keys
for the GitHub import; see below.

The healthcheck asserts `"dense": true`, not just that the process is alive. A
container whose embedding model failed to load is not serving degraded scores
quietly — it is reported unhealthy.

---

## Credentials

Only the **GitHub import** needs them. Everything else — upload, parse, score,
attribute, preview, export — is local and deterministic.

```bash
cp .env.sample .env     # then fill in what you need
```

| Variable | Needed for |
|---|---|
| `GROQ_API_KEY` | summarising repositories into resume lines. Required for import. |
| `GITHUB_PERSONAL_ACCESS_TOKEN` | optional. Without it you get 60 requests/hour; with it, 5,000. |
| `TYPESAFE_API_KEY` | optional. Used only to choose which repositories are worth reading. |
| `LANGSMITH_*` | optional tracing. |

You can also type the keys straight into the import panel in the page. Those are
sent with that one request and are never stored.

---

## Configuration

| Variable | Default | What it does |
|---|---|---|
| `RBA_ONNX_THREADS` | `1` | ONNX intra-op threads. Set to 4 on a real machine. |
| `RBA_EMBED_BATCH` | `4` | Texts per embedding chunk. A **memory** limit, not a speed knob — it keeps peak RSS flat within a call. 16 is safe in the 2 GB container. |
| `RBA_REQUIRE_DENSE` | unset | When set, the server refuses to start if the embedding model did not load, instead of serving scores missing half their weight. |
| `RBA_MODEL_LOCAL_ONLY` | unset | Never reach for the network to fetch the model. Set in the image, where it is pre-baked. |
| `RBA_ROOT` | — | Where `config.yaml` and `.env` live. Needed for a non-editable install, where the package's own location is the wrong answer. |
| `RBA_CONFIG` | — | Path to `config.yaml` directly. |
| `RBA_WEB_ROOT` | — | Path to the static frontend. Defaults to `$RBA_ROOT/web`. |
| `RBA_SESSION_TTL` | `7200` | Seconds a working document is held server-side. |
| `RBA_MCP_MODE` | `remote` | `remote` \| `docker` \| `rest`. See below. |

### A note on `RBA_MCP_MODE=docker`

The default is `remote` — GitHub's hosted MCP endpoint — and that is a deliberate
security choice, not an oversight.

In `docker` mode the app does **not** talk to the `github-mcp` compose service over
the network. It shells out to `docker run -i --rm ghcr.io/github/github-mcp-server`
and spawns a **sibling container on your host daemon**. That requires mounting
`/var/run/docker.sock` into the web container, which hands that container effective
root on your machine.

Compose therefore does not mount it. If you want the self-hosted path you have to
add the mount yourself, along with the Docker CLI in the image and a `group_add` for
the socket's GID. The `github-mcp` service exists only to pre-pull the image:

```bash
docker compose --profile mcp up
```

`make_backend` falls back to `remote` cleanly when Docker is unavailable, so nothing
breaks if you leave this alone.

---

## How it fits together

```
src/rba/        the engine — v2's own copy, free to evolve (see the root README)
src/rba_api/    the HTTP layer
  app.py          FastAPI app, lifespan, static mount
  deps.py         app-scoped Config, Cache, rate limiters, thread pools
  sessions.py     the working document, held between requests
  jobs.py         long-running work (the GitHub import) + SSE
  scoring_pool.py where CPU-bound work runs so the event loop stays free
  routes/         meta, documents, score, render, github
web/            the frontend. No bundler, no build step, no node.
  css/            design tokens, layout, components, the paper preview
  js/             ES modules, loaded directly by the browser
fixtures/       sample resumes, including a synthetic one for the demo
scripts/        capture_demo.py — regenerates the README's GIF and stills
```

### The API

| Route | Purpose |
|---|---|
| `GET /api/health` | liveness, and whether the dense channel loaded |
| `GET /api/meta/scoring` | channel weights, bands, token limit — the "how the score works" tab reads this rather than hardcoding anything |
| `GET /api/meta/roles` | the role profiles from `roles.yaml` |
| `POST /api/upload` · `POST /api/parse` | a PDF or pasted text in, a structured document out |
| `PATCH /api/session/{id}/document` | apply edits; optimistic concurrency via `if_match` |
| `POST /api/score` | the fused score. Identical to calling `rescore()` directly. |
| `POST /api/score/batch` | N candidate documents in one pass — what makes a delta beside every control affordable |
| `POST /api/attribution` | exact leave-one-out impact per line |
| `POST /api/session/{id}/github` | start an import; returns a job id |
| `GET /api/jobs/{id}/events` | server-sent progress for that job |
| `POST /api/render/html \| pdf \| latex \| text` · `POST /api/export` | preview and download |

Interactive docs at `/api/docs` when the server is running.

### Two design rules the code enforces

**The browser never computes a score.** Every number on the page came from the
server. The obvious client-side shortcut — approximating a document's embedding by
pooling its bullets' — was measured in `rba/resume/attribution.py` and agreed on
*sign* for only 14 of 20 bullets, so it would tell you a line helps when it hurts.
A delta that has not been computed renders as a dim `·` and waits.

**Scores are the fused three-channel score, never a rubric.** There is deliberately
no seven-bar breakdown summing to 100: those weights were invented, and
`rba/resume/diagnostics.py` says so in its own docstring. The rubric survives only
as *advice* — which keyword is missing, whether the resume runs long.

---

## Tests

```bash
cd v2
pytest                      # offline, no credentials needed
ruff check src tests
```

Everything runs against fakes with dummy keys. If a test needs a real credential,
that test is wrong.

The load-bearing suite is `tests/test_batch.py`, which asserts that batch scoring
equals `rescore()` to **float equality** — not approximately. The whole premise of
the UI is that a delta beside a control is the real fused score, so if those two can
disagree by a rounding step, every number on the page is suspect.

## Regenerating the demo

```bash
cd v2
.venv/bin/python scripts/capture_demo.py
```

Drives the real application in a headless browser against
`fixtures/demo_resume.txt` — an invented person, so no real contact details are ever
burned into an image — and writes the GIF and stills into `../docs/`.
