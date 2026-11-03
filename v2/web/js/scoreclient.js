/* Deltas, without the browser ever computing one.
 *
 * The reference artifact showed a signed delta beside every control and could
 * afford it because its scorer was a JS function. Ours is an ONNX model on the
 * server. So the strategy is:
 *
 *   - coalesce: every delta a view wants goes in ONE /api/score/batch request;
 *   - cache: keyed by (rev, jd, ops), because toggling a line off and back on
 *     revisits states we already paid for;
 *   - never guess: a delta we do not have renders as a dim dot.
 *
 * That last point is not pedantry. `rba/resume/attribution.py` records
 * measuring the cheap client-side approximation and finding it agreed on SIGN
 * for only 14 of 20 bullets -- it would tell a user a line helps when it hurts.
 * A dot is honest; a wrong sign is worse than silence.
 */
import { api, reportError } from './api.js';
import { state } from './store.js';
import { deltaClass, signed } from './dom.js';

const CACHE_MAX = 600;
const cache = new Map();          // key -> delta (number)
const known = new Map();          // key -> {score_100, raw}

let queue = new Map();            // key -> ops
let flushTimer = null;
let inflight = null;

/* Keyed on the server's document SIGNATURE, not on `rev`.
 *
 * `rev` increments on every patch, so toggling a line off and back on lands on
 * rev N+2 and could never match the entry stored at rev N -- the cache was
 * structurally incapable of hitting, which is what made every edit blank all
 * the deltas. The signature is a hash of (id, text, enabled) over all bullets,
 * so returning to a previous state returns to the same key. */
/** Is there anything to score against? A chosen role counts: the server
 *  synthesises a job description from its keyword profile. */
export function hasTarget() {
  return Boolean(state.target.jd_text.trim() || state.target.role_key);
}

function keyOf(ops) {
  return JSON.stringify([
    state.signature, state.target.jd_text, state.target.role_key, ops,
  ]);
}

function remember(key, delta, block) {
  cache.set(key, delta);
  if (block) known.set(key, block);
  while (cache.size > CACHE_MAX) cache.delete(cache.keys().next().value);
}

/** Paint one delta cell. Three states, distinguished by glyph and opacity --
 *  never by a made-up number. */
export function paint(node, delta, { stale = false, pending = false } = {}) {
  node.textContent = signed(delta);
  // Built as one string rather than successive classList.add() calls: a delta
  // that rounds to zero gives an empty token, and `classList.add('')` THROWS.
  // That exception used to escape repaintAll()'s loop, so the first ~zero
  // delta stopped every later cell from being painted and they stayed as dots
  // permanently. An empty entry here is just filtered out.
  const classes = [
    delta === null || delta === undefined ? 'dz' : deltaClass(delta),
    stale ? 'stale' : '',
    pending ? 'inflight' : '',
  ].filter(Boolean);
  node.className = classes.join(' ');
}

/** Request a delta for `ops`. Resolves immediately from cache when possible. */
export function want(ops) {
  const key = keyOf(ops);
  if (cache.has(key)) return cache.get(key);
  queue.set(key, ops);
  schedule();
  return undefined;               // undefined = pending, null = known-absent
}

export function cached(ops) {
  return cache.get(keyOf(ops));
}

function schedule() {
  clearTimeout(flushTimer);
  flushTimer = setTimeout(flush, 150);
}

/** Repaint every [data-delta] node from the cache. Called when a batch lands.
 *  Writes text nodes only -- a full re-render here would steal focus from
 *  whatever the user is typing in. */
export function repaintAll() {
  for (const node of document.querySelectorAll('[data-delta]')) {
    const spec = node.dataset.ops;
    if (!spec) continue;
    try {
      const ops = JSON.parse(spec);
      const key = keyOf(ops);
      if (cache.has(key)) paint(node, cache.get(key));
      else paint(node, null, { pending: queue.has(key) });
    } catch (error) {
      // One malformed cell must never stop the others from updating.
      console.warn('could not repaint a delta cell', error);
    }
  }
}

async function flush() {
  if (!state.sessionId || !hasTarget()) { queue.clear(); return; }
  if (inflight) { schedule(); return; }          // one batch at a time
  const pending = queue;
  if (!pending.size) return;
  queue = new Map();

  const entries = [...pending.entries()];
  const variants = entries.map(([key, ops], i) => ({ key: `v${i}`, ops }));
  const controller = new AbortController();
  inflight = controller;
  // What the document looked like when this batch was issued. If it has moved
  // by the time the batch lands, the results describe a document the user has
  // already edited away from and must be dropped -- otherwise a slow batch
  // repaints the scorecard with the pre-edit number.
  const issuedFor = state.signature;
  state.scorePending = variants.length;
  repaintAll();

  try {
    const body = await api.scoreBatch(
      { session: state.sessionId, variants, use_dense: true }, controller.signal);

    if (issuedFor !== state.signature) return;      // stale; discard silently

    // `target` sits beside `base` in the batch response but inside the block
    // in /api/score. Fold it in so the stand-in caveat does not flicker away
    // every time a batch lands.
    state.score = { ...body.base, target: body.target ?? null };
    const byKey = new Map(body.variants.map((v) => [v.key, v]));
    entries.forEach(([key], i) => {
      const got = byKey.get(`v${i}`);
      if (got) remember(key, got.delta, got);
    });
    state.lastBatchStats = body.stats;
  } catch (error) {
    reportError(error, 'scoring');
  } finally {
    inflight = null;
    state.scorePending = 0;
    repaintAll();
    document.dispatchEvent(new CustomEvent('rba:scored'));
  }
}

/** The headline number. Separate from the batch so it is never queued behind
 *  twenty cosmetic deltas. */
let baseController = null;
export async function refreshBase() {
  if (!state.sessionId || !hasTarget()) return null;
  baseController?.abort();
  baseController = new AbortController();
  const issuedFor = state.signature;
  try {
    const block = await api.score(
      { session: state.sessionId, use_dense: true }, baseController.signal);
    if (issuedFor !== state.signature) return null;   // the document moved on
    state.score = block;
    document.dispatchEvent(new CustomEvent('rba:scored'));
    return block;
  } catch (error) {
    reportError(error, 'scoring');
    return null;
  }
}

export function invalidate() {
  // Abort as well as clear. Without this, a request issued against the old
  // document is still running and will happily overwrite `state.score` with a
  // number for a document that no longer exists.
  inflight?.abort();
  baseController?.abort();
  inflight = null;
  cache.clear();
  known.clear();
  queue.clear();
}
