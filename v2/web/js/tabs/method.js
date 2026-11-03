/* "How the score works" -- entirely server-sourced.
 *
 * Nothing here hardcodes a weight. If the backend changes CHANNEL_WEIGHTS this
 * page changes with it, which is the only way an explanation stays true. */
import { html, raw } from '../dom.js';
import { state } from '../store.js';

export function methodView() {
  const meta = state.meta;
  if (!meta) return html`<div class="lock">Loading…</div>`;

  const weights = Object.entries(meta.weights)
    .map(([name, w]) => html`<tr><td>${name}</td><td class="num">${w}</td></tr>`)
    .join('');

  const bands = meta.bands
    .map((b) => html`<tr><td>${b.label}</td><td class="num">${b.at_least.toFixed(2)}</td></tr>`)
    .join('');

  const rubric = Object.entries(meta.advice_rubric ?? {})
    .map(([name, w]) => html`<tr><td>${name}</td><td class="num">${w}</td></tr>`)
    .join('');

  return html`<div class="wrap">
    <div class="callout w">
      <strong>These are estimates.</strong> No real applicant tracking system
      publishes its scoring, so this measures the things that are measurable
      and tells you exactly how.
    </div>

    <div class="two">
      <div class="box">
        <h2>The three channels</h2>
        <p class="note">${meta.notes.headline}</p>
        <table class="tbl"><thead><tr><th>Channel</th><th>Weight</th></tr></thead>
        <tbody>${raw(weights)}</tbody></table>
        <p class="note" style="margin-top:8px">Model: <code>${meta.model}</code></p>
      </div>

      <div class="box">
        <h2>Bands</h2>
        <p class="note">Real systems report an ordinal grade rather than a
        percentage, so the number comes with one.</p>
        <table class="tbl"><thead><tr><th>Band</th><th>At least</th></tr></thead>
        <tbody>${raw(bands)}</tbody></table>
      </div>
    </div>

    <div class="box" style="margin-top:14px">
      <h2>Why there is no seven-bar rubric</h2>
      <p class="note">${meta.notes.no_rubric}</p>
      <details class="more"><summary>The advice-only rubric, for reference</summary>
        <div class="body">
          <p class="note">These weights drive <em>advice</em> — which keyword is
          missing, whether the resume is too long — and never the score.</p>
          <table class="tbl"><thead><tr><th>Component</th><th>Points</th></tr></thead>
          <tbody>${raw(rubric)}</tbody></table>
        </div>
      </details>
    </div>

    <div class="two" style="margin-top:14px">
      <div class="box">
        <h2>What a delta means</h2>
        <p class="note">${meta.notes.deltas}</p>
      </div>
      <div class="box">
        <h2>A limit worth knowing</h2>
        <p class="note">${meta.notes.truncation}</p>
      </div>
    </div>
  </div>`;
}
