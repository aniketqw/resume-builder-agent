/* Four delegated listeners on `document`. No per-node handlers anywhere, so
 * a re-render never leaves a dangling listener behind. */
import { $, debounce } from './dom.js';
import { api, reportError } from './api.js';
import { state, set } from './store.js';
import { render, refreshPreview, refreshScore } from './render.js';
import { toggleTheme } from './components/theme.js';
import { toast } from './components/toast.js';
import { loadDocument, refreshPaper, applyOps, setTarget, previewBody, runImport } from './actions.js';

export function wireEvents() {
  document.addEventListener('click', onClick);
  document.addEventListener('change', onChange);
  document.addEventListener('input', onInput);
  document.addEventListener('rba:scored', refreshScore);
  wireDropzone();
}

function onClick(event) {
  const button = event.target.closest('button');
  if (!button) return;
  const d = button.dataset;

  if (button.id === 'themeBtn') { toggleTheme(); return; }
  if (d.tab) {
    if (button.getAttribute('aria-disabled') === 'true') return;
    set({ tab: d.tab }); render(); window.scrollTo(0, 0); return;
  }
  if (d.view) { set({ view: d.view }); refreshPaper(); return; }
  if (d.role) { setTarget({ role_key: d.role }); return; }

  if (button.id === 'pickFile') { $('#fileInput')?.click(); return; }
  if (button.id === 'pasteGo') {
    const text = $('#pasteText')?.value ?? '';
    if (!text.trim()) { toast('Paste some text first.'); return; }
    loadDocument(() => api.parse(text, state.sessionId));
    return;
  }
  if (button.id === 'ghGo') { startImport(); return; }
  if (button.id === 'ghCancel') { cancelImport(); return; }
  if (button.id === 'dlBundle') { download('/api/export', 'resumes.zip'); return; }
  if (button.id === 'dlPdf') { download('/api/render/pdf', `resume-${state.view}.pdf`); return; }
  if (button.id === 'copyText') { copyPlain(); return; }
}

function onChange(event) {
  const node = event.target;
  if (node.id === 'fileInput' && node.files?.[0]) {
    loadDocument(() => api.upload(node.files[0], state.sessionId));
    return;
  }
  if (node.dataset.toggle) {
    applyOps([{ op: node.checked ? 'enable' : 'disable', bullet_id: node.dataset.toggle }]);
    return;
  }
  if (node.id === 'numbersChk') { set({ showNumbers: node.checked }, { silent: true }); refreshPaper(); return; }
  if (node.id === 'provChk') { set({ showProvenance: node.checked }, { silent: true }); refreshPaper(); return; }
}

const onJdInput = debounce((value) => setTarget({ jd_text: value }), 400);

function onInput(event) {
  if (event.target.id === 'jdText') onJdInput(event.target.value);
}

function wireDropzone() {
  document.addEventListener('dragover', (e) => {
    const zone = e.target.closest?.('#drop');
    if (zone) { e.preventDefault(); zone.classList.add('over'); }
  });
  document.addEventListener('dragleave', (e) => {
    e.target.closest?.('#drop')?.classList.remove('over');
  });
  document.addEventListener('drop', (e) => {
    const zone = e.target.closest?.('#drop');
    if (!zone) return;
    e.preventDefault();
    zone.classList.remove('over');
    const file = e.dataTransfer?.files?.[0];
    if (file) loadDocument(() => api.upload(file, state.sessionId));
  });
}

function startImport() {
  const username = ($('#ghUser')?.value ?? '').trim();
  if (!username) { toast('Enter a GitHub username.'); return; }
  const raw = Number.parseInt($('#ghMax')?.value ?? '6', 10);
  runImport({
    username,
    token: $('#ghToken')?.value ?? '',
    groqKey: $('#ghGroq')?.value ?? '',
    maxRepos: Number.isFinite(raw) ? Math.min(25, Math.max(1, raw)) : 6,
  });
}

function cancelImport() {
  const jobId = state.importJob?.jobId;
  if (jobId) fetch(`/api/jobs/${jobId}`, { method: 'DELETE' }).catch(() => {});
}

async function download(path, filename) {
  try {
    const response = await fetch(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(previewBody()),
    });
    if (!response.ok) throw new Error(await response.text());
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const anchor = Object.assign(document.createElement('a'),
                                 { href: url, download: filename });
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    URL.revokeObjectURL(url);
    const pages = response.headers.get('X-RBA-Pages');
    toast(pages && Number(pages) > 1
      ? `Downloaded — note it runs to ${pages} pages.`
      : 'Downloaded.');
  } catch (error) { reportError(error, 'download'); }
}

async function copyPlain() {
  try {
    const response = await fetch('/api/render/text', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(previewBody()),
    });
    await navigator.clipboard.writeText(await response.text());
    toast('Plain text copied.');
  } catch (error) { reportError(error, 'copy'); }
}
