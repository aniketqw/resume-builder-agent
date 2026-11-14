---
name: langgraph-docs
description: Use when writing, changing, or debugging LangGraph or LangChain code in this repo — StateGraph nodes/edges, conditional edges, state schemas and reducers, astream/stream_mode, checkpointers, interrupts, or LangSmith tracing. Enforces verifying APIs against the INSTALLED package and the current docs instead of recalling them from memory.
---

# LangGraph / LangChain: verify, don't recall

LangGraph's API moves fast and its docs have **moved host**. Guessing a method name or a
keyword argument here produces code that imports fine and fails at runtime. Verify first.

## 1. Check what is actually installed

This repo keeps **separate virtualenvs per version**. Use the right one.

```bash
v2/.venv/bin/python -c "from importlib.metadata import version; \
print('langgraph', version('langgraph')); print('langchain-core', version('langchain-core'))"
```

Known good as of this writing: **langgraph 1.2.19**, **langchain-core 1.7.3**, Python 3.11.
The venv is managed by `uv` and has **no `pip` binary** — use `uv pip ... --python v2/.venv/bin/python`.

## 2. Verify the API against the installed package, not memory

Before using any method or kwarg:

```bash
v2/.venv/bin/python - <<'PY'
import inspect
from langgraph.graph.state import CompiledStateGraph
print(list(inspect.signature(CompiledStateGraph.astream).parameters))
PY
```

If a symbol is not in that output, it does not exist in this build. Do not write it.

## 3. Use the current docs

The old `langchain-ai.github.io/langgraph/...` URLs now **redirect** and fetch as
"Redirecting…" with no content. Use:

- `https://docs.langchain.com/oss/python/langgraph/graph-api`
- `https://docs.langchain.com/oss/python/langgraph/graph-api-control-flow`
- `https://docs.langchain.com/oss/python/langgraph/streaming`
- `https://reference.langchain.com/python/langgraph/graph/state/StateGraph`

## 4. Traps already hit in this repo

**Streaming.** `stream_mode="updates"` yields `{"<node>": <partial state>}` per node but
**never yields the merged final state**. To get progress *and* the result you must use the
multi-mode form, which yields `(mode, chunk)` tuples:

```python
async for mode, chunk in graph.astream(inputs, stream_mode=["updates", "values"]):
    if mode == "updates":
        for node_name, partial in chunk.items():
            report(*NODE_PROGRESS[node_name])
    elif mode == "values":
        final = chunk
```

Do not use `updates` alone and try to reconstruct state by merging.

**Valid stream modes** in this build: `values, updates, checkpoints, tasks, debug,
messages, custom`.

**Checkpointer.** The sync `SqliteSaver` raises `NotImplementedError` under `ainvoke`;
an async graph needs `AsyncSqliteSaver`. (v2 has since removed the checkpointer — do not
reintroduce one without being asked.)

**Serialisation.** A checkpointer msgpack-serialises the *entire* state after every node.
Without one, that constraint disappears and plain dataclasses can live in state.

**Conditional edges.** Key the branch off **state**, not off an attribute of the object
holding the node methods — the graph should be readable standalone:

```python
g.add_conditional_edges(
    "build_document",
    lambda s: "build_resumes" if s.get("full_run") else END,
    ["build_resumes", END],
)
```

## 5. LangSmith tracing

LangGraph traces nodes natively when `LANGSMITH_TRACING` and `LANGSMITH_API_KEY` are set
— no per-node instrumentation needed. In this repo `enable_tracing()` in
`v2/src/rba/config.py` is the single place env vars are read and the `LANGCHAIN_*` aliases
are mirrored.

- A bare `client.create_run(...)` with no `trace_id`/`parent_run_id` creates a **flat
  orphan run**, not a child — that is why traces used to render as a list, not a tree.
- Tracing must never break a run: keep it wrapped so any exception is swallowed and logged.
- `v2/tests/conftest.py` disables tracing for the suite and that is **load-bearing**
  (measured 0.8 s → 5.8 s). Do not remove it.
- Never print, log, or commit the key. `v2/.env` is gitignored and must stay that way.

## 6. Repo-specific rules

- `langchain-groq` is declared but **never imported** — `GroqClient` is raw `httpx`. Do not
  assume a LangChain chat model is in play.
- One graph serves both entry points: the CLI runs the full branch, the browser stops at
  `build_document`. Never add a second parallel workflow to make a test pass.
- The ATS scorer is deterministic and makes **no LLM calls**. Do not route it through a model.
