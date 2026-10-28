"""The live preview.

Styled to look like the printed page rather than like a web app, because the
preview's job is to answer "is this still my resume?" -- a question a list of
Streamlit widgets cannot answer.

Self-contained: the CSS is inlined, so the same string can be dropped into
`st.html`, written to a file, or printed from a browser.
"""
from __future__ import annotations

from html import escape

from ..document import KeyValues, ResumeDoc, Table
from .model import (
    keyvalue_rows,
    table_rows,
    text_of,
    visible_bullets,
    visible_entries,
    visible_sections,
)
from .options import ORIGIN_COLOURS, ORIGIN_LABELS, RenderOptions

CSS = """
.resume{background:#fff;color:#111;font-family:Georgia,'Times New Roman',serif;
 font-size:12.5px;line-height:1.42;padding:28px 32px;max-width:860px;margin:0 auto;
 border:1px solid #d8d8d8;border-radius:4px}
.resume h1{font-size:21px;letter-spacing:.06em;text-align:center;margin:0 0 10px;
 font-weight:700;text-transform:uppercase}
.resume h2{font-size:12px;letter-spacing:.09em;text-transform:uppercase;
 border-bottom:1.1px solid #222;margin:16px 0 7px;padding-bottom:2px;font-weight:700}
.resume .head{display:grid;grid-template-columns:auto 1fr;gap:1px 10px;
 font-family:Helvetica,Arial,sans-serif;font-size:11.5px;margin-bottom:4px}
.resume .head .k{color:#444}
.resume .entry{margin-top:7px}
.resume .entry .top{display:flex;justify-content:space-between;gap:12px;
 align-items:baseline}
.resume .entry .title{font-weight:700}
.resume .entry .dates{font-size:11px;color:#444;white-space:nowrap;
 font-family:Helvetica,Arial,sans-serif}
.resume .tech{font-size:11.5px;font-style:italic;color:#333;margin:1px 0 2px}
.resume ul{margin:3px 0 0;padding-left:17px}
.resume li{margin:2px 0}
.resume table{border-collapse:collapse;width:100%;font-size:11.5px;
 font-family:Helvetica,Arial,sans-serif}
.resume th{text-align:left;font-size:10px;letter-spacing:.05em;color:#333;
 border-bottom:1px solid #bbb;padding:3px 6px 3px 0;text-transform:uppercase}
.resume td{padding:3px 6px 3px 0;vertical-align:top;border-bottom:1px solid #eee}
.resume .kv{display:grid;grid-template-columns:132px 1fr;gap:2px 10px;
 font-family:Helvetica,Arial,sans-serif;font-size:11.5px}
.resume .kv .k{font-weight:700}
.resume .summary{margin:2px 0 0;text-align:justify}
.resume .legend{font-family:Helvetica,Arial,sans-serif;font-size:10.5px;
 color:#555;margin:0 0 10px;text-align:center}
.resume .legend span{margin:0 7px;border-bottom:2px solid currentColor}
"""


def _esc(text: str) -> str:
    return escape(text, quote=False)


def _bullet_html(bullet, opts: RenderOptions) -> str:
    text = _esc(text_of(bullet.text, opts))
    if not opts.colour_provenance:
        return f"<li>{text}</li>"
    colour = ORIGIN_COLOURS.get(bullet.origin, ORIGIN_COLOURS["assumed"])
    title = ORIGIN_LABELS.get(bullet.origin, bullet.origin)
    return (f'<li style="color:{colour}" title="{_esc(title)}">{text}</li>')


def to_html(doc: ResumeDoc, options: RenderOptions | None = None, *,
            standalone: bool = False) -> str:
    opts = options or RenderOptions()
    out: list[str] = ['<div class="resume">']

    if doc.name:
        out.append(f"<h1>{_esc(doc.name)}</h1>")
    if opts.colour_provenance:
        out.append('<div class="legend">'
                   + "".join(f'<span style="color:{ORIGIN_COLOURS[k]}">{_esc(v)}</span>'
                             for k, v in ORIGIN_LABELS.items())
                   + "</div>")
    if doc.header is not None and doc.header.fields:
        out.append('<div class="head">')
        for label, value in doc.header.fields:
            out.append(f'<div class="k">{_esc(label)}</div><div>{_esc(value)}</div>')
        out.append("</div>")
    elif doc.contact:
        out.append(f'<div class="head"><div></div><div>{_esc(doc.contact)}</div></div>')

    if doc.summary:
        out.append("<h2>Summary</h2>")
        out.append(f'<p class="summary">{_esc(text_of(doc.summary, opts))}</p>')

    for section in visible_sections(doc, opts):
        out.append(f"<h2>{_esc(section.title)}</h2>")
        if isinstance(section.body, KeyValues):
            out.append('<div class="kv">')
            for label, value in keyvalue_rows(section.body):
                out.append(f'<div class="k">{_esc(label)}</div>'
                           f"<div>{_esc(text_of(value, opts))}</div>")
            out.append("</div>")
        elif isinstance(section.body, Table):
            out.append("<table>")
            if section.body.headers:
                out.append("<tr>" + "".join(f"<th>{_esc(h)}</th>"
                                            for h in section.body.headers) + "</tr>")
            for row in table_rows(section.body):
                out.append("<tr>" + "".join(f"<td>{_esc(text_of(c, opts))}</td>"
                                            for c in row) + "</tr>")
            out.append("</table>")
        if section.text:
            out.append(f'<p class="summary">{_esc(text_of(section.text, opts))}</p>')

        for entry in visible_entries(section, opts):
            out.append('<div class="entry"><div class="top"><div class="title">')
            out.append(_esc(" - ".join(p for p in (entry.title, entry.subtitle) if p)))
            out.append("</div>")
            if entry.dates:
                out.append(f'<div class="dates">{_esc(entry.dates)}</div>')
            out.append("</div>")
            if entry.tech:
                out.append(f'<div class="tech">Tech: {_esc(", ".join(entry.tech))}</div>')
            out.append("<ul>")
            out += [_bullet_html(b, opts) for b in visible_bullets(entry, opts)]
            out.append("</ul></div>")

        own = visible_bullets(section, opts)
        if own:
            out.append("<ul>")
            out += [_bullet_html(b, opts) for b in own]
            out.append("</ul>")

    out.append("</div>")
    body = "\n".join(out)
    if standalone:
        return (f"<!doctype html><meta charset='utf-8'>"
                f"<title>{_esc(doc.name or 'Resume')}</title>"
                f"<style>{CSS}</style>{body}")
    return f"<style>{CSS}</style>{body}"
