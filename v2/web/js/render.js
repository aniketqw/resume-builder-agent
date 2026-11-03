/* Full-string render, with surgical escapes.
 *
 * The whole view is rebuilt on a state change -- it is one readable function
 * and it is fast enough. But async results (a batch of deltas, a preview)
 * arrive while the user may be typing, so those get targeted updates that
 * touch text nodes only and never blow away a caret or a scroll position. */
import { $, html, raw } from './dom.js';
import { state } from './store.js';
import { simView } from './tabs/simulator.js';
import { methodView } from './tabs/method.js';
import { scorecard } from './components/scorecard.js';
import { repaintAll } from './scoreclient.js';

export const TABS = [
  { id: 'sim',    label: 'Simulator',          view: simView,  needsDoc: true },
  { id: 'method', label: 'How the score works', view: methodView, needsDoc: false },
];

export function render() {
  const scrollY = window.scrollY;

  $('#tabs').innerHTML = TABS.map((tab) => {
    const locked = tab.needsDoc && !state.doc;
    return html`<button type="button" role="tab" data-tab="${tab.id}"
      aria-selected="${state.tab === tab.id}"
      ${raw(locked && tab.id !== 'sim' ? 'aria-disabled="true"' : '')}
      >${tab.label}</button>`;
  }).join('');

  const current = TABS.find((t) => t.id === state.tab) ?? TABS[0];
  $('#view').innerHTML = current.view();

  repaintAll();
  window.scrollTo(0, scrollY);
}

/** Replace only the paper, so a preview landing does not rebuild the controls. */
export function refreshPreview() {
  const node = $('#paper');
  if (!node) return;
  node.className = `paper p${state.view}`;
  node.innerHTML = state.previewHtml || '<div class="empty">Rendering…</div>';
}

/** Replace only the scorecard. */
export function refreshScore() {
  const node = document.querySelector('.scorecard');
  if (!node) return;
  const holder = document.createElement('div');
  holder.innerHTML = scorecard();
  node.replaceWith(holder.firstElementChild);
}
