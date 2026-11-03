/* The workbench: controls left, live paper right.
 *
 * Every version control carries a `data-delta` cell and a `data-ops` payload
 * describing the candidate document it represents. `scoreclient` collects all
 * of them into one batch request; nothing here knows how a score is computed. */
import { html, raw, esc } from '../dom.js';
import { state } from '../store.js';
import { scorecard } from '../components/scorecard.js';
import { want } from '../scoreclient.js';
import { githubBox } from '../components/github.js';

export function simView() {
  if (!state.doc) return onboardingPrompt();
  return html`<div class="grid">
    <div class="col">
      ${raw(targetBox())}
      ${raw(scorecard())}
      ${raw(itemsBox())}
      ${raw(githubBox({ compact: true }))}
    </div>
    <div class="col">
      ${raw(paperPane())}
    </div>
  </div>`;
}

function onboardingPrompt() {
  return html`<div class="wiz">
    <div class="box">
      <h2>Start with the resume you already have</h2>
      <p class="sub">It is parsed into real structure — tables, dated entries,
      tech lines — and never rewritten. Nothing is sent to a model to do this.</p>
      <div class="drop" id="drop" style="margin-top:12px">
        <p><strong>Drop a PDF here</strong>, or</p>
        <p class="row" style="justify-content:center">
          <button class="btn pri" id="pickFile" type="button">Choose a file</button>
        </p>
        <input type="file" id="fileInput" accept=".pdf,.txt,.md" hidden>
      </div>
      <details class="more" style="margin-top:10px">
        <summary>Or paste the text instead</summary>
        <div class="body">
          <textarea id="pasteText" rows="8"
            placeholder="Paste your resume text…"></textarea>
          <p class="row" style="margin-top:8px">
            <button class="btn" id="pasteGo" type="button">Use this text</button>
          </p>
        </div>
      </details>
    </div>
    ${raw(githubBox())}
  </div>`;
}

function targetBox() {
  const roles = state.roles.map((role) => html`
    <button class="chip" type="button" data-role="${role.key}"
      aria-pressed="${state.target.role_key === role.key}">${role.short ?? role.title}</button>`).join('');

  return html`<div class="box">
    <h2>Target</h2>
    <p class="note">Paste the job description you are actually applying to — a
    real posting scores far more meaningfully than a role profile.</p>
    <textarea id="jdText" rows="5" placeholder="Paste the job description…"
      >${esc(state.target.jd_text)}</textarea>
    <div class="row" style="margin-top:8px">${raw(roles)}</div>
    <p class="note" style="margin-top:6px">
      Choosing a role fills in a keyword profile as a stand-in when you have no
      posting to hand.
    </p>
  </div>`;
}

function itemsBox() {
  const doc = state.doc;
  const rows = [];
  for (const section of doc.sections ?? []) {
    for (const [index, entry] of (section.entries ?? []).entries()) {
      rows.push(itemRow(section, entry, `${section.title}#${index}`));
    }
  }
  if (!rows.length) {
    return html`<div class="box"><h2>Your resume</h2>
      <p class="note">No entries were found in this document.</p></div>`;
  }
  return html`<div class="box">
    <h2>Lines on your resume</h2>
    <p class="note">Each line shows what removing it would do to the score.
    A dot means not measured yet.</p>
    ${raw(rows.join(''))}
  </div>`;
}

function itemRow(section, entry, key) {
  const bullets = (entry.bullets ?? []).map((bullet) => {
    // Always ask about the move the user can actually make from here. For a
    // line that is ON that means "what does removing it cost?"; for a
    // candidate that is OFF it means "what would accepting it gain?".
    // Asking to disable an already-disabled line is a no-op, and every
    // imported line rendered a useless 0.00.
    const ops = [{ op: bullet.enabled ? 'disable' : 'enable',
                   bullet_id: bullet.id }];
    const title = bullet.enabled
      ? 'What the score does if this line is removed'
      : 'What the score does if you accept this line';
    want(ops);
    return html`<li>
      <input type="checkbox" data-toggle="${bullet.id}"
        ${bullet.enabled ? 'checked' : ''} aria-label="Show this line">
      <span class="bt">${bullet.text}
        ${raw(bullet.source_repo
          ? html`<span class="ev">from ${bullet.source_repo}</span>` : '')}
      </span>
      <span data-delta class="dz" data-ops='${JSON.stringify(ops)}'
        title="${title}">·</span>
    </li>`;
  }).join('');

  return html`<div class="item">
    <div class="hd">
      <div>
        <div class="ti">${entry.title || section.title}</div>
        <div class="meta">${entry.subtitle || ''}${entry.dates ? ` · ${entry.dates}` : ''}</div>
      </div>
    </div>
    <ul class="bullets">${raw(bullets)}</ul>
  </div>`;
}

function paperPane() {
  return html`<div class="paperwrap">
    <div class="pagebar">
      <div class="seg" role="group" aria-label="Paper style">
        <button type="button" data-view="bits"
          aria-pressed="${state.view === 'bits'}">BITS format</button>
        <button type="button" data-view="clean"
          aria-pressed="${state.view === 'clean'}">Clean / ATS</button>
      </div>
      <label class="chk"><input type="checkbox" id="numbersChk"
        ${state.showNumbers ? 'checked' : ''}> Numbers</label>
      <label class="chk"><input type="checkbox" id="provChk"
        ${state.showProvenance ? 'checked' : ''}> Provenance</label>
    </div>
    <div class="pagebar">
      <button class="btn pri" type="button" id="dlBundle">Download both resumes</button>
      <button class="btn" type="button" id="dlPdf">This one as PDF</button>
      <button class="btn" type="button" id="copyText">Copy plain text</button>
    </div>
    <div class="paper p${state.view}" id="paper">
      ${raw(state.previewHtml || '<div class="empty">Rendering…</div>')}
    </div>
  </div>`;
}
