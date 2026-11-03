/* Every call to the server lives here.
 *
 * One rule this module enforces: nothing in the browser computes a score, so
 * there is no local fallback for a failed scoring request. A failure surfaces
 * as a failure. */
import { toast } from './components/toast.js';

const JSON_HEADERS = { 'Content-Type': 'application/json' };

class ApiError extends Error {
  constructor(status, detail) {
    super(typeof detail === 'string' ? detail : JSON.stringify(detail));
    this.status = status;
    this.detail = detail;
  }
}

async function request(path, { method = 'GET', body, signal, raw = false } = {}) {
  const init = { method, signal, headers: {} };
  if (body !== undefined) {
    init.headers = JSON_HEADERS;
    init.body = JSON.stringify(body);
  }
  const response = await fetch(path, init);
  if (!response.ok) {
    let detail;
    try { detail = (await response.json()).detail; }
    catch { detail = await response.text().catch(() => response.statusText); }
    throw new ApiError(response.status, detail);
  }
  return raw ? response : response.json();
}

export const api = {
  health:      ()          => request('/api/health'),
  scoringMeta: ()          => request('/api/meta/scoring'),
  roles:       ()          => request('/api/meta/roles'),

  createSession: (target)  => request('/api/session', { method: 'POST', body: target ?? null }),
  readSession:  (id)       => request(`/api/session/${id}`),
  setTarget:    (id, t)    => request(`/api/session/${id}/target`, { method: 'PATCH', body: t }),

  upload(file, sessionId) {
    const form = new FormData();
    form.append('file', file);
    const query = sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : '';
    return fetch(`/api/upload${query}`, { method: 'POST', body: form })
      .then(async (r) => {
        if (!r.ok) throw new ApiError(r.status, (await r.json().catch(() => ({}))).detail);
        return r.json();
      });
  },
  parse: (text, sessionId) => request(
    `/api/parse${sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : ''}`,
    { method: 'POST', body: { text } }),

  patchDocument: (id, ops, ifMatch) => request(`/api/session/${id}/document`,
    { method: 'PATCH', body: { ops, if_match: ifMatch } }),

  score:      (body, signal) => request('/api/score',       { method: 'POST', body, signal }),
  scoreBatch: (body, signal) => request('/api/score/batch', { method: 'POST', body, signal }),
  attribution:(body, signal) => request('/api/attribution', { method: 'POST', body, signal }),

  renderHtml: (body) => request('/api/render/html', { method: 'POST', body }),
  exportUrl:  '/api/export',

  startImport: (id, body) => request(`/api/session/${id}/github`,
    { method: 'POST', body }),
  job: (jobId) => request(`/api/jobs/${jobId}`),

  /** Follow a job to completion. `onProgress(message, fraction)` fires per
   *  checkpoint; resolves with the job's result, rejects on error. */
  followJob(jobId, onProgress) {
    return new Promise((resolve, reject) => {
      const source = new EventSource(`/api/jobs/${jobId}/events`);
      source.addEventListener('progress', (e) => {
        const d = JSON.parse(e.data);
        onProgress?.(d.message, d.fraction);
      });
      source.addEventListener('done', (e) => {
        source.close();
        resolve(JSON.parse(e.data));
      });
      source.addEventListener('error', (e) => {
        source.close();
        // A payload means the job failed; no payload means the connection did.
        let detail = 'the connection to the server dropped';
        try { detail = JSON.parse(e.data).message; } catch { /* transport */ }
        reject(new ApiError(0, detail));
      });
      source.addEventListener('cancelled', () => { source.close(); reject(new ApiError(0, 'cancelled')); });
    });
  },
};

export { ApiError };

/** Report a failure once, in words, instead of letting it vanish into the
 *  console. An aborted request is not a failure -- it means the user moved on. */
export function reportError(error, context = '') {
  if (error?.name === 'AbortError') return;
  const detail = error?.detail ?? error?.message ?? String(error);
  toast(context ? `${context}: ${detail}` : String(detail), 5000);
  console.error(context, error);
}
