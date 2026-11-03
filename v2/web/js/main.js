import { state, set, prefs } from './store.js';
import { api, reportError } from './api.js';
import { render } from './render.js';
import { wireEvents } from './events.js';
import { initTheme } from './components/theme.js';
import { resync, refreshPaper } from './actions.js';
import { refreshBase } from './scoreclient.js';

async function boot() {
  initTheme();
  wireEvents();
  render();

  try {
    const [meta, roles] = await Promise.all([api.scoringMeta(), api.roles()]);
    set({
      meta,
      roles: roles.roles.map((r) => ({ ...r, short: shortName(r.title) })),
    }, { silent: true });
  } catch (error) {
    reportError(error, 'startup');
  }

  // Pick up where the last visit left off, if that session is still alive.
  const previous = prefs.lastSession.get();
  if (previous) {
    try {
      const payload = await api.readSession(previous);
      if (payload.document) {
        set({
          sessionId: payload.session_id, rev: payload.rev,
          doc: payload.document, signature: payload.signature ?? '',
          target: { ...state.target, ...(payload.target ?? {}) },
        }, { silent: true });
        await refreshPaper();
        if (state.target.jd_text) await refreshBase();
      }
    } catch { /* expired, which is normal after the TTL. Start fresh. */ }
  }

  set({ ready: true }, { silent: true });
  render();
}

function shortName(title) {
  return title.replace(/ ?\/ ?/g, ' / ').split(' ').slice(0, 3).join(' ');
}

boot();
