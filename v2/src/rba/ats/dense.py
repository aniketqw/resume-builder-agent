"""The dense channel: semantic similarity without PyTorch.

Model: **BAAI/bge-small-en-v1.5** (MIT, 33M parameters, 384 dimensions).

Why not `all-MiniLM-L6-v2`, which is what everyone reaches for: its BEIR
retrieval score is **43.81** against bge-small's **51.68**, and it was trained
with ``max_seq_length`` 256, so it *silently truncates* — bad for documents as
long as a resume. The "MiniLM scores 56.3 on MTEB" figure quoted in most blog
posts is the overall average across 56 tasks, not retrieval.

Run through ONNX Runtime rather than sentence-transformers, because
sentence-transformers pulls PyTorch — roughly 1GB installed and 400-700MB
resident — and Streamlit Community Cloud only guarantees 690MB. ONNX Runtime
plus this model is about 150MB resident.

The channel degrades rather than fails: if the model cannot be fetched (no
network on first run, or a corrupted cache) :func:`embed` returns ``None``, the
channel is dropped, and its weight is redistributed across the others.
"""
from __future__ import annotations

import logging
import os
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

MODEL_ID = "BAAI/bge-small-en-v1.5"
ONNX_FILE = "onnx/model.onnx"
MAX_TOKENS = 512          # bge-small genuinely handles this; MiniLM does not
_SESSION: object | None = None
_TOKENIZER: object | None = None
_LOAD_FAILED = False

# v2 serves concurrent requests, so `load()` can be entered twice at once. Both
# callers would see `_SESSION is None` and each build an InferenceSession --
# two copies of a 128 MB model, and the slower one silently wins the global.
_LOAD_LOCK = threading.Lock()


def _env_int(name: str, default: int) -> int:
    """An env override that refuses to make things worse.

    A bad value here degrades scoring silently, which is the failure mode this
    whole module is written to avoid, so anything unparseable or < 1 falls back
    to the default rather than propagating.
    """
    try:
        value = int(os.environ.get(name, "") or default)
    except ValueError:
        log.warning("%s is not an integer; using %d", name, default)
        return default
    if value < 1:
        log.warning("%s must be >= 1; using %d", name, default)
        return default
    return value


def _cache_dir() -> Path:
    return Path.home() / ".cache" / "rba-models"


def load(local_only: bool = False):
    """Load tokenizer and ONNX session once. Returns None if unavailable."""
    global _SESSION, _TOKENIZER, _LOAD_FAILED
    if _LOAD_FAILED:
        return None
    if _SESSION is not None and _TOKENIZER is not None:
        return _SESSION, _TOKENIZER
    with _LOAD_LOCK:
        # Re-check inside the lock: the thread we queued behind may have
        # finished the load (or discovered it fails) while we waited.
        if _LOAD_FAILED:
            return None
        if _SESSION is not None and _TOKENIZER is not None:
            return _SESSION, _TOKENIZER
        return _load_locked(local_only)


def _load_locked(local_only: bool):
    global _SESSION, _TOKENIZER, _LOAD_FAILED
    try:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        cache = _cache_dir()
        cache.mkdir(parents=True, exist_ok=True)
        model_path = hf_hub_download(MODEL_ID, ONNX_FILE, cache_dir=str(cache),
                                     local_files_only=local_only)
        tok_path = hf_hub_download(MODEL_ID, "tokenizer.json", cache_dir=str(cache),
                                   local_files_only=local_only)
        options = ort.SessionOptions()
        # 1 is right for Streamlit Community Cloud, which gives fractions of a
        # core. In a container you control it is leaving 2x on the table:
        # measured on one 512-token text, 294 ms at 1 thread vs 149 ms at 4.
        # Default stays 1 so nothing changes unless the deployment opts in.
        options.intra_op_num_threads = _env_int("RBA_ONNX_THREADS", 1)
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        _SESSION = ort.InferenceSession(model_path, options,
                                        providers=["CPUExecutionProvider"])
        tokenizer = Tokenizer.from_file(tok_path)
        tokenizer.enable_truncation(max_length=MAX_TOKENS)
        tokenizer.enable_padding(length=None)
        _TOKENIZER = tokenizer
        return _SESSION, _TOKENIZER
    except Exception as exc:
        log.warning("dense channel unavailable (%s); it will be skipped", exc)
        _LOAD_FAILED = True
        return None


# How many texts go through the transformer at once.
#
# This is a memory limit, not a speed knob. The attribution engine asks for one
# variant per bullet, so a 150-bullet resume becomes a 152-text call, and the
# model holds a [batch, seq_len, hidden] activation for every layer at once.
# Measured peak RSS embedding full-page resume texts in ONE call:
#
#     152 texts, unchunked -> 4,133 MB      80 -> 2,034 MB
#      40 texts, unchunked -> 1,368 MB      16 ->   574 MB
#
# Streamlit Community Cloud gives about 1 GB, so an unchunked call was killed
# by the OOM reaper -- which surfaces as a blank "Error running app" page with
# no traceback, because the process never gets to raise.
#
# Chunked, peak memory is flat in the number of texts and the chunk size buys
# headroom for free -- 152 texts cost the same time either way:
#
#     BATCH  2 -> 335 MB, 35.0 s        BATCH  8 -> 598 MB, 36.6 s
#     BATCH  4 -> 446 MB, 35.7 s        BATCH 12 -> 744 MB, 36.8 s
#
# 4 leaves room for Streamlit's own footprint on a 1 GB container.
#
# This is a MEMORY limit, not a speed knob -- but on a 2 GB container the
# chunking does cost real time on batch work: 16 texts as 4 chunks at 1 thread
# is 5,017 ms, against 1,996 ms as one call at 4 threads. v2's Docker image
# sets RBA_EMBED_BATCH=16 with a documented memory budget; the default is
# unchanged so v1's deployment target keeps its headroom.
BATCH = _env_int("RBA_EMBED_BATCH", 4)


def embed(texts: list[str], local_only: bool = False,
          batch: int = BATCH) -> np.ndarray | None:
    """CLS embeddings, L2-normalised. None if the model is unavailable.

    Chunked: the result is identical to a single large call, but peak memory
    stays flat in the number of texts instead of growing with it.
    """
    if not texts:
        return None
    loaded = load(local_only=local_only)
    if loaded is None:
        return None
    session, tokenizer = loaded
    expected = {i.name for i in session.get_inputs()}

    out: list[np.ndarray] = []
    size = max(1, batch)
    for start in range(0, len(texts), size):
        chunk = [t or "" for t in texts[start:start + size]]
        encodings = tokenizer.encode_batch(chunk)
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        feed = {"input_ids": input_ids, "attention_mask": attention}
        if "token_type_ids" in expected:
            feed["token_type_ids"] = np.zeros_like(input_ids)

        hidden = session.run(None,
                             {k: v for k, v in feed.items() if k in expected})[0]
        # BGE uses the CLS token, not mean pooling. Mean pooling here measurably
        # degrades retrieval quality for this model family.
        out.append(np.array(hidden[:, 0, :], dtype=np.float32))
        del hidden

    cls = np.concatenate(out, axis=0) if len(out) > 1 else out[0]
    norms = np.linalg.norm(cls, axis=1, keepdims=True)
    return cls / np.clip(norms, 1e-9, None)


def cosine_similarity(resume_text: str, jd_text: str,
                      local_only: bool = False) -> float | None:
    """Semantic similarity in [0, 1], or None when the channel is unavailable."""
    vectors = embed([resume_text, jd_text], local_only=local_only)
    if vectors is None:
        return None
    raw = float(np.dot(vectors[0], vectors[1]))
    # Cosine on unit vectors is in [-1, 1], but for natural-language text it is
    # effectively [0, 1] -- embeddings of two English documents are almost
    # never anti-correlated. Mapping (raw+1)/2 therefore wastes half the scale
    # and compresses everything above 0.5: measured on this model, an entirely
    # irrelevant document scored 0.75 and a strong match 0.91, a spread of
    # 0.16. Clamping the raw value instead gives 0.49 against 0.83.
    return max(0.0, min(1.0, raw))


@lru_cache(maxsize=1)
def available(local_only: bool = False) -> bool:
    return load(local_only=local_only) is not None


def tokenizer(local_only: bool = False):
    """The shared truncating tokenizer, or None when the model is unavailable.

    Exposed so :mod:`rba.ats.batch` can measure truncation without reaching
    into this module's privates.
    """
    loaded = load(local_only=local_only)
    return None if loaded is None else loaded[1]


def truncated_ids(texts: list[str],
                  local_only: bool = False) -> list[tuple[int, ...]] | None:
    """The token ids each text actually reaches the model as.

    This exists because of a property of the scorer that is easy to miss:
    ``MAX_TOKENS`` is 512 and the tokenizer truncates, but a real one-page
    resume is about 1,150 tokens. So two documents that differ only after the
    cut produce **bit-identical** embeddings, and embedding both is pure waste.

    Measured on the attribution path, sixteen leave-one-out variants of a real
    resume collapse to four distinct prefixes.

    Returning the ids rather than a hash keeps the caller's dedup exact: there
    is no collision to reason about.
    """
    tok = tokenizer(local_only=local_only)
    if tok is None:
        return None
    return [tuple(e.ids) for e in tok.encode_batch([t or "" for t in texts])]
