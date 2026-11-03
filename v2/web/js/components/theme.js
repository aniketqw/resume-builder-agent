/* Tri-state theme: OS preference by default, overridden by data-theme.
 * The toggle reads the OS setting so the first click always visibly flips,
 * rather than setting the value it already effectively had. */
import { prefs } from '../store.js';

export function initTheme() {
  const stored = prefs.theme.get();
  if (stored === 'light' || stored === 'dark') {
    document.documentElement.dataset.theme = stored;
  }
}

export function toggleTheme() {
  const root = document.documentElement;
  const current = root.dataset.theme;
  const osDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  const effective = current || (osDark ? 'dark' : 'light');
  const next = effective === 'dark' ? 'light' : 'dark';
  root.dataset.theme = next;
  prefs.theme.set(next);
  return next;
}
