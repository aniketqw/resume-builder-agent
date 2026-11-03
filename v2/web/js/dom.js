/* Tiny DOM helpers. No framework, no build step. */

export const $  = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

/** Escape for interpolation into markup. Everything user- or server-supplied
 *  goes through this: resume text is arbitrary and a stray `<` must not
 *  become an element. */
export function esc(value) {
  return String(value ?? '')
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

/** Tagged template that escapes every interpolation by default.
 *  Pass a value through `raw()` to opt out. */
export function html(strings, ...values) {
  return strings.reduce((out, chunk, i) => {
    if (i === 0) return chunk;
    const value = values[i - 1];
    const text = value && value.__raw ? value.value
      : Array.isArray(value) ? value.join('')
      : esc(value);
    return out + text + chunk;
  }, '');
}

export const raw = (value) => ({ __raw: true, value: String(value ?? '') });

export function mount(target, markup) {
  const node = typeof target === 'string' ? $(target) : target;
  if (node) node.innerHTML = markup;
  return node;
}

/** A signed, fixed-width number. `null` is "not computed", which renders as a
 *  dim dot rather than a zero -- a zero is a claim, a dot is not. */
export function signed(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) return '·';
  const fixed = Math.abs(value).toFixed(digits);
  if (Number(fixed) === 0) return `0.${'0'.repeat(digits)}`;
  return (value > 0 ? '+' : '−') + fixed;
}

export function deltaClass(value) {
  if (value === null || value === undefined) return 'dz';
  if (value > 0.005) return 'up';
  if (value < -0.005) return 'down';
  return '';
}

export function debounce(fn, ms) {
  let timer = null;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), ms);
  };
}
