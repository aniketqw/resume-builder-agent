/* One state object, one owner, explicit notification.
 *
 * The document itself is NOT kept here as the source of truth -- it lives on
 * the server, and `doc` below is the copy the server last sent us. That is the
 * clean break from the reference artifact, which kept everything in
 * localStorage and could not survive a scorer that lives elsewhere. */

export const state = {
  ready: false,
  tab: 'sim',

  sessionId: null,
  rev: 0,
  doc: null,              // last document the server sent
  signature: '',

  roles: [],
  meta: null,             // /api/meta/scoring

  target: { jd_text: '', role_key: '', required_skills: [], preferred_skills: [] },

  view: 'bits',           // paper skin
  showNumbers: true,
  showProvenance: false,

  score: null,            // last base score block from the server
  scorePending: 0,

  previewHtml: '',
  busy: null,

  // The import form. Credentials live here only for the lifetime of the page
  // and are deliberately never persisted.
  github: { username: '', maxRepos: 6, result: null },
  importJob: null,          // { running, message, fraction, jobId }
};

const listeners = new Set();

export function subscribe(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

export function set(patch, { silent = false } = {}) {
  Object.assign(state, patch);
  if (!silent) notify();
}

export function notify() {
  for (const fn of listeners) fn(state);
}

/* ---------------------------------------------------------- persistence --
 * Selections only, namespaced per document and validated on read. A stored
 * value from a previous import can reference bullets that no longer exist, so
 * nothing is trusted without checking it against the live document. */

const KEY = (docId, name) => `rba:${docId}:v1:${name}`;

export function save(docId, name, value) {
  if (!docId) return;
  try { localStorage.setItem(KEY(docId, name), JSON.stringify(value)); }
  catch { /* private mode, or quota. Losing a preference is not an error. */ }
}

export function load(docId, name, fallback) {
  if (!docId) return fallback;
  try {
    const raw = localStorage.getItem(KEY(docId, name));
    return raw === null ? fallback : JSON.parse(raw);
  } catch { return fallback; }
}

export const prefs = {
  theme: {
    get: () => { try { return localStorage.getItem('rba:theme'); } catch { return null; } },
    set: (value) => { try { localStorage.setItem('rba:theme', value); } catch {} },
  },
  lastSession: {
    get: () => { try { return localStorage.getItem('rba:session'); } catch { return null; } },
    set: (value) => { try { localStorage.setItem('rba:session', value); } catch {} },
  },
};
