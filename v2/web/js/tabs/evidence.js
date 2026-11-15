/* Evidence & Call-Graph: makes "how did we get this sentence?" clickable.
 *
 * Every claim a GitHub import produced is shown here, not only the ones that
 * made it onto the resume -- a `needs_measure` or `rejected` claim is exactly
 * what this tab exists to make legible: *why* a line did or didn't survive
 * verification. The call-graph panel is deliberately a plain list, not a
 * force-directed diagram -- `rba.repobrain`'s own scope is "symbol / callers
 * / callees by bare name, Python only" (see its module docstring), and a
 * graph layout would visually overclaim precision this data doesn't have.
 */
import { html, raw, esc } from '../dom.js';
import { state } from '../store.js';

const STATUS_LABEL = {
  verified: 'Verified',
  needs_measure: 'Needs a number',
  rejected: 'Rejected',
};

export function evidenceView() {
  if (!state.sessionId) {
    return html`<div class="lock">Import from GitHub first — this tab explains
      the claims and code structure behind whatever it finds.</div>`;
  }
  if (state.repoGraph.loading) {
    return html`<div class="lock">Loading evidence…</div>`;
  }
  const repoNames = Object.keys(state.repoGraph.repos);
  if (!repoNames.length) {
    return html`<div class="lock">No repository evidence yet — run a GitHub
      import from the Simulator tab, then come back here.</div>`;
  }

  const selected = state.repoGraph.selectedRepo in state.repoGraph.repos
    ? state.repoGraph.selectedRepo : repoNames[0];
  const repo = state.repoGraph.repos[selected];

  return html`<div class="wrap">
    <div class="row" style="margin-bottom:10px;flex-wrap:wrap">
      ${raw(repoNames.map((name) => html`<button type="button" class="btn
        ${name === selected ? 'pri' : ''}" data-evidence-repo="${name}"
        >${name}</button>`).join(''))}
    </div>

    <div class="two">
      <div class="box">
        <h2>Claim &rarr; evidence chain</h2>
        <p class="note">${repo.claims.length} claim${repo.claims.length === 1 ? '' : 's'}
        found in <code>${selected}</code>. Each one shows the exact quote and
        line it was verified against, or why it wasn't.</p>
        ${raw(claimChain(selected, repo.claims))}
      </div>

      <div class="box">
        <h2>Call graph</h2>
        ${raw(callGraph(repo.graph))}
      </div>
    </div>
  </div>`;
}

function claimChain(repo, claims) {
  if (!claims.length) {
    return html`<p class="note muted">No claims recorded for this repository.</p>`;
  }
  return `<ul class="bullets evidence-chain">` + claims.map((claim) => {
    const first = claim.evidence[0];
    return html`<li class="status-${esc(claim.status)}">
      <span class="bt">
        <span class="chip chip-${esc(claim.status)}">${STATUS_LABEL[claim.status] ?? claim.status}</span>
        <span>${claim.text}</span>
        ${raw(first ? evidenceLine(repo, first) : noEvidence(claim))}
      </span>
    </li>`;
  }).join('') + `</ul>`;
}

function evidenceLine(repo, ev) {
  const where = ev.line_start
    ? (ev.line_start === ev.line_end ? `L${ev.line_start}` : `L${ev.line_start}-${ev.line_end}`)
    : '';
  const location = `${ev.path}${where ? `#${where}` : ''}`;
  return html`<span class="ev">
    “${ev.quote}”
    — ${ev.permalink
      ? html`<a href="${ev.permalink}" target="_blank" rel="noopener">${location}</a>`
      : location}
  </span>`;
}

function noEvidence(claim) {
  if (claim.status === 'needs_measure' && claim.measure_hint) {
    return html`<span class="ev">${claim.measure_hint}</span>`;
  }
  if (claim.reason) {
    return html`<span class="ev">${claim.reason}</span>`;
  }
  return '';
}

function callGraph(graph) {
  if (!graph) {
    return html`<p class="note muted">No code structure indexed for this
      repository (non-Python source, or indexing failed).</p>`;
  }
  const paths = Object.keys(graph.files);
  if (!paths.length) {
    return html`<p class="note muted">No Python files were indexed.</p>`;
  }

  const gaps = graph.partial_source.length
    ? html`<details class="more"><summary>Not indexed (${graph.partial_source.length})</summary>
        <div class="body">${raw(graph.partial_source
          .map((g) => html`<p class="mono" style="font-size:11px">${g}</p>`).join(''))}</div>
      </details>`
    : '';

  const files = paths.map((path) => {
    const file = graph.files[path];
    if (!file.symbols.length) return '';
    const symbols = file.symbols.map((sym) => {
      const callees = file.calls
        .filter((c) => c.caller === sym.qualname)
        .map((c) => c.callee);
      const uniqueCallees = [...new Set(callees)];
      return html`<div class="call-sym">
        <span class="mono">${sym.kind} <strong>${sym.name}</strong></span>
        <span class="note">lines ${sym.line_start}-${sym.line_end}</span>
        ${uniqueCallees.length
          ? html`<span class="note">calls &rarr; ${uniqueCallees.join(', ')}</span>`
          : ''}
      </div>`;
    }).join('');
    return html`<details class="more" open><summary class="mono">${path}</summary>
      <div class="body">${raw(symbols)}</div></details>`;
  }).join('');

  return files === '' ? html`<p class="note muted">No functions or classes found.</p>`
    : html`${raw(files)}${raw(gaps)}`;
}
