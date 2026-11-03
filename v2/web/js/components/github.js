/* The "Add from GitHub" panel.
 *
 * Credentials are typed here and sent per request -- never stored, never put
 * in localStorage. The max-repos cap is a spend control, not a preference:
 * roughly 3,000 tokens and one request per repository.
 */
import { html, raw, esc } from '../dom.js';
import { state } from '../store.js';

export function githubBox({ compact = false } = {}) {
  const job = state.importJob;
  const running = Boolean(job && job.running);

  return html`<div class="box" id="ghBox">
    <h2>${compact ? 'Or add projects from GitHub' : 'Add projects from GitHub'}</h2>
    <p class="note">Reads your public repositories and turns each into
    evidence-backed lines. They arrive <strong>switched off</strong> — your
    resume stays exactly as it is until you accept one.</p>

    <div class="grid2" style="margin-top:10px">
      <label class="note">GitHub username
        <input type="text" id="ghUser" placeholder="octocat"
               value="${esc(state.github.username)}" ${raw(running ? 'disabled' : '')}>
      </label>
      <label class="note">Repositories to read
        <input type="text" id="ghMax" inputmode="numeric"
               value="${state.github.maxRepos}" ${raw(running ? 'disabled' : '')}>
      </label>
    </div>
    <details class="more" style="margin-top:8px">
      <summary>Keys (needed to read repositories and summarise them)</summary>
      <div class="body grid2">
        <label class="note">GitHub token <span class="muted">(optional)</span>
          <input type="password" id="ghToken" placeholder="ghp_…"
                 ${raw(running ? 'disabled' : '')}>
        </label>
        <label class="note">Groq API key
          <input type="password" id="ghGroq" placeholder="gsk_…"
                 ${raw(running ? 'disabled' : '')}>
        </label>
        <p class="note" style="grid-column:1/-1">
          Sent with this one request and not stored. Without a GitHub token you
          get 60 requests an hour; with one, 5,000. A Groq key is required —
          summarising is what turns a repository into resume lines.
        </p>
      </div>
    </details>

    <div class="row" style="margin-top:10px">
      <button class="btn pri" type="button" id="ghGo" ${raw(running ? 'disabled' : '')}>
        ${running ? 'Reading…' : 'Import from GitHub'}</button>
      ${raw(running ? html`<button class="btn" type="button" id="ghCancel">Stop</button>` : '')}
      <span class="note">Roughly 3,000 tokens and one request per repository.</span>
    </div>

    ${raw(progress(job))}
    ${raw(outcome(state.github.result))}
  </div>`;
}

function progress(job) {
  if (!job) return '';
  const pct = Math.round((job.fraction ?? 0) * 100);
  return html`<div style="margin-top:10px">
    <div class="progress"><i style="width:${pct}%"></i></div>
    <p class="note" style="margin-top:4px">${job.message || 'Starting…'}</p>
  </div>`;
}

function outcome(result) {
  if (!result) return '';
  const failure = result.failure
    ? html`<div class="callout ${result.failure.level === 'error' ? 'b' : 'w'}">
        <p class="note" style="margin:0">${result.failure.message}</p>
        <details class="more"><summary>Detail (${result.failure.count})</summary>
          <div class="body">${raw(result.failure.detail
            .map((line) => html`<p class="mono" style="font-size:11px">${line}</p>`)
            .join(''))}</div>
        </details>
      </div>`
    : '';

  const excluded = (result.excluded ?? []).length
    ? html`<details class="more"><summary>Repositories left out
        (${result.excluded.length})</summary>
        <div class="body"><table class="tbl"><tbody>${raw(result.excluded
          .map((r) => html`<tr><td>${r.name}</td>
            <td class="note">${r.reason || 'not claimable'}</td></tr>`).join(''))}
        </tbody></table></div></details>`
    : '';

  return html`<div style="margin-top:10px">
    ${raw(failure)}
    <div class="callout">
      <p class="note" style="margin:0"><strong>${result.added}</strong> candidate
      line${result.added === 1 ? '' : 's'} added, all switched off. Tick one to
      see what it is worth.</p>
    </div>
    ${raw(excluded)}
  </div>`;
}
