/* Things the user does, and what the app does about them.
 *
 * Kept apart from `events.js` so the wiring stays readable and these stay
 * testable: each one is "call the server, update the store, re-render". */
import { api, reportError } from './api.js';
import { state, set, prefs } from './store.js';
import { render, refreshPreview } from './render.js';
import { toast } from './components/toast.js';
import { refreshBase, invalidate, hasTarget } from './scoreclient.js';

/** The body every render/export call sends. Lives here, not in events.js,
 *  so the import graph stays one-directional. */
export function previewBody() {
  return {
    session: state.sessionId,
    style: state.view,
    show_numbers: state.showNumbers,
    colour_provenance: state.showProvenance,
  };
}

/** Adopt a document returned by upload/parse/patch. */
function adopt(payload) {
  set({
    sessionId: payload.session_id ?? state.sessionId,
    rev: payload.rev,
    doc: payload.document,
    signature: payload.signature ?? '',
  }, { silent: true });
  if (payload.session_id) prefs.lastSession.set(payload.session_id);
  invalidate();
}

export async function loadDocument(fetcher) {
  try {
    toast('Reading your resume…');
    const payload = await fetcher();
    adopt(payload);
    render();
    await Promise.all([refreshPaper(), hasTarget() ? refreshBase() : null]);
    toast('Loaded.');
  } catch (error) { reportError(error, 'upload'); }
}

export async function applyOps(ops) {
  if (!state.sessionId) return;
  try {
    const payload = await api.patchDocument(state.sessionId, ops, state.rev);
    adopt(payload);
    // Repaint the controls FIRST, from the document the server just returned,
    // so the checkbox the user clicked settles immediately. The score and the
    // preview then land into that DOM -- `refreshBase` fires `rba:scored`,
    // which surgically swaps the scorecard. Doing render() last instead tore
    // down the scorecard that had just been updated, and made the user wait
    // for two round trips before anything moved.
    render();
    await Promise.all([refreshPaper(), refreshBase()]);
  } catch (error) {
    if (error.status === 409) {
      // Another tab moved the document. Re-read rather than overwrite.
      toast('This resume changed elsewhere — reloading it.');
      await resync();
      return;
    }
    reportError(error, 'edit');
  }
}

export async function resync() {
  if (!state.sessionId) return;
  try {
    const payload = await api.readSession(state.sessionId);
    if (payload.document) {
      adopt({ ...payload, document: payload.document });
      render();
      // refreshBase() belongs here too. Without it the headline score stayed
      // frozen after any 409 -- which two quick toggles reliably produce --
      // until the next edit happened to succeed.
      await Promise.all([refreshPaper(), refreshBase()]);
    }
  } catch (error) { reportError(error, 'resync'); }
}

export async function setTarget(patch) {
  const target = { ...state.target, ...patch };
  set({ target }, { silent: true });
  invalidate();
  if (!state.sessionId) { render(); return; }
  try {
    await api.setTarget(state.sessionId, target);
    render();
    if (hasTarget()) await refreshBase();
  } catch (error) { reportError(error, 'target'); }
}

/** Run a GitHub import and fold the result into the working document.
 *
 *  The generated lines arrive switched off, so nothing about the current
 *  score changes until the user accepts one -- which is why this re-renders
 *  and re-previews but does not need to invalidate the delta cache beyond
 *  what `adopt()` already does.
 */
export async function runImport(form) {
  if (!state.sessionId) {
    // No document yet: open a session so the import has somewhere to land.
    const created = await api.createSession(null);
    set({ sessionId: created.session_id }, { silent: true });
  }
  set({
    github: { ...state.github, username: form.username,
              maxRepos: form.maxRepos, result: null },
    importJob: { running: true, message: 'Starting…', fraction: 0 },
  }, { silent: true });
  render();

  try {
    const { job_id: jobId } = await api.startImport(state.sessionId, {
      username: form.username,
      github_token: form.token,
      groq_key: form.groqKey,
      max_repos: form.maxRepos,
    });
    set({ importJob: { ...state.importJob, jobId } }, { silent: true });

    const result = await api.followJob(jobId, (message, fraction) => {
      set({ importJob: { running: true, jobId, message, fraction } },
          { silent: true });
      updateProgress();
    });

    adopt(result);
    set({
      importJob: null,
      github: { ...state.github, result },
    }, { silent: true });
    render();
    await Promise.all([refreshPaper(), hasTarget() ? refreshBase() : null]);
    toast(`${result.added} candidate line${result.added === 1 ? '' : 's'} added, switched off.`);
  } catch (error) {
    set({ importJob: null }, { silent: true });
    render();
    reportError(error, 'GitHub import');
  }
}

/** Move the bar without rebuilding the panel, so the form keeps its focus
 *  and the typed keys are not wiped mid-run. */
function updateProgress() {
  const job = state.importJob;
  const bar = document.querySelector('#ghBox .progress i');
  const label = document.querySelector('#ghBox .progress + .note');
  if (bar) bar.style.width = `${Math.round((job?.fraction ?? 0) * 100)}%`;
  if (label) label.textContent = job?.message ?? '';
}

export async function refreshPaper() {
  if (!state.sessionId || !state.doc) return;
  try {
    const { html } = await api.renderHtml(previewBody());
    set({ previewHtml: html }, { silent: true });
    refreshPreview();
  } catch (error) { reportError(error, 'preview'); }
}
