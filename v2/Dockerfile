# syntax=docker/dockerfile:1
#
# v1's Dockerfile built the *command-line tool*: no port, no server, no UI.
# This one builds the actual application.
#
# Three stages, because the embedding model is the interesting part. It is
# ~130 MB and `dense.load()` fetches it from Hugging Face on first use --
# then catches every exception and returns None if that fails. A silent None
# drops the 0.5-weighted dense channel and every score shifts with no error
# anywhere. So the model is baked into the image and the container never
# needs the network to score.

# ---------------------------------------------------------------- builder --
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src
# Into a clean prefix so the runtime stage copies dependencies without pip,
# setuptools or any build tooling coming along for the ride.
RUN pip install --prefix=/install .

# ----------------------------------------------------------------- models --
FROM python:3.11-slim AS models

ENV PIP_NO_CACHE_DIR=1
RUN pip install "huggingface_hub>=0.23"
# Mirrors rba.ats.dense: same repo, same two files, same cache layout.
RUN python -c "\
from huggingface_hub import hf_hub_download as get; \
get('BAAI/bge-small-en-v1.5', 'onnx/model.onnx', cache_dir='/models'); \
get('BAAI/bge-small-en-v1.5', 'tokenizer.json', cache_dir='/models')"

# ---------------------------------------------------------------- runtime --
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # `rba.config` resolves config.yaml relative to the package by default,
    # which is wrong for a non-editable install -- the file would be looked
    # for outside site-packages entirely. RBA_ROOT is the override.
    RBA_ROOT=/app \
    RBA_CONFIG=/app/config.yaml \
    # The container owns its cores, unlike Streamlit Cloud which the defaults
    # were tuned for. Measured on one 512-token text: 294 ms at 1 thread,
    # 84 ms at 4.
    RBA_ONNX_THREADS=4 \
    RBA_EMBED_BATCH=16 \
    OMP_NUM_THREADS=4 \
    # The model is in the image, so never reach for the network.
    RBA_MODEL_LOCAL_ONLY=1 \
    # Fail loudly rather than serve scores missing half their weight.
    RBA_REQUIRE_DENSE=1 \
    # Default to GitHub's hosted MCP endpoint. The self-hosted path needs the
    # Docker socket mounted, which would give this container root on the host.
    RBA_MCP_MODE=remote \
    HOME=/home/app

RUN useradd --create-home --uid 10001 app

COPY --from=builder /install /usr/local

WORKDIR /app
COPY config.yaml ./
COPY web ./web

# dense._cache_dir() is hardcoded to Path.home()/".cache"/"rba-models", so the
# baked model has to land exactly there, owned by the user that will read it.
COPY --from=models --chown=app:app /models /home/app/.cache/rba-models

RUN mkdir -p /app/.cache /app/out && chown -R app:app /app /home/app

USER app
EXPOSE 8000

# No curl in the image, so the check is Python. It asserts `dense` as well as
# `ok`: a container serving degraded scores is not healthy, it is wrong.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import json,sys,urllib.request; \
d=json.load(urllib.request.urlopen('http://127.0.0.1:8000/api/health',timeout=4)); \
sys.exit(0 if d.get('ok') and d.get('dense') else 1)"

# One worker, deliberately. See the module docstring in rba_api/app.py.
CMD ["uvicorn", "rba_api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
