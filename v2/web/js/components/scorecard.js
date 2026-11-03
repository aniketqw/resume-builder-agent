/* The scorecard.
 *
 * Shaped like the reference artifact's, but every value is re-sourced. In
 * particular there is NO seven-bar rubric: `rba/resume/diagnostics.py` rejects
 * those weights in its own docstring ("nobody measured that a missing phone
 * number is worth exactly six points"). What replaces it is the three real
 * channels, with the weights the server actually used. */
import { html, raw, signed } from '../dom.js';
import { state } from '../store.js';

const CHANNEL_LABEL = {
  dense: 'Meaning', lexical: 'Wording', skills: 'Skills',
};
const CHANNEL_NOTE = {
  dense: 'embedding similarity',
  lexical: 'BM25 term overlap',
  skills: 'weighted skills match',
};

export function scorecard() {
  const score = state.score;
  if (!score) {
    return html`<div class="box scorecard">
      <h2>Estimated match</h2>
      <p class="note">Paste a job description, or pick a role above, to see a score.</p>
    </div>`;
  }

  const pct = Math.max(0, Math.min(100, score.score_100));
  const bars = Object.entries(score.channels ?? {}).map(([name, value]) => {
    const weight = (score.weights_used ?? {})[name];
    return html`<div class="part">
      <div class="lbl">
        <span>${CHANNEL_LABEL[name] ?? name}</span>
        <span class="mono">${value.toFixed(3)}</span>
      </div>
      <div class="bar"><i style="width:${Math.round(value * 100)}%"></i></div>
      <div class="note" style="font-size:10.5px">
        ${CHANNEL_NOTE[name] ?? ''}${weight ? ` · weight ${weight}` : ''}
      </div>
    </div>`;
  });

  const pending = state.scorePending > 0
    ? html`<span class="tag w">scoring ${state.scorePending} option${state.scorePending === 1 ? '' : 's'}…</span>`
    : '';

  return html`<div class="box scorecard">
    <h2>Estimated match</h2>
    <div class="big mono">${pct.toFixed(1)}<small> /100</small></div>
    <div class="row" style="margin-top:4px">
      <span class="tag ${bandClass(score.band)}">${score.band}</span>
      ${raw(pending)}
    </div>
    <div class="track" aria-hidden="true">
      <div class="rail"></div>
      <div class="fill" style="width:${pct}%"></div>
    </div>
    <div class="parts">${raw(bars.join(''))}</div>
    ${raw(standIn(score.target))}
    <p class="note" style="margin-top:10px">
      A convex combination of the channels above. A channel that cannot be
      computed is dropped and its weight shared out — it counts as unknown,
      not as zero.
    </p>
  </div>`;
}

/** When the score came from a role rather than a real posting, say so — and
 *  show the exact text that was scored against. A stand-in the user cannot
 *  inspect is indistinguishable from a real result. */
function standIn(target) {
  if (!target || !target.synthesised) return '';
  return html`<div class="callout w" style="margin-top:10px">
    <p class="note" style="margin:0">${target.caveat}</p>
    <details class="more">
      <summary>What it scored against</summary>
      <div class="body">
        <p class="mono" style="font-size:11.5px">${target.jd_text}</p>
        <p class="note">Required: ${target.required_skills.join(', ')}</p>
        <p class="note">Preferred: ${target.preferred_skills.join(', ')}</p>
      </div>
    </details>
  </div>`;
}

function bandClass(band = '') {
  if (band.startsWith('Strong')) return 'g';
  if (band.startsWith('Good')) return 'g';
  if (band.startsWith('Partial')) return 'w';
  return 'b';
}
