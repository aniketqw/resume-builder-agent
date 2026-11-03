"""Record a complete, auditable trace of one pipeline run.

The point of this file is to make the agent's behaviour **checkable by a
human** rather than taken on trust. It records what every stage cost, which
repositories were chosen and why, and — most importantly — every claim the
verifier threw out and its reason. A run that only showed successes would be
the least useful possible artefact for judging quality.
"""
from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path


def merge(state: dict, update: dict) -> dict:
    """Merge a node result, accumulating `errors` instead of overwriting them.

    `state |= update` silently replaces the previous `errors` list, which made
    the run trace report "no bullets dropped" on a run that dropped five. The
    graph uses an `operator.add` reducer for this field; a plain script has to
    do it by hand.

    Lives here rather than in `build_sample.py` so it can be imported — and
    tested — without pulling in the whole agent package.
    """
    errors = list(state.get("errors") or []) + list(update.get("errors") or [])
    state |= update
    state["errors"] = errors
    return state


@dataclass
class Stage:
    name: str
    detail: str = ""
    seconds: float = 0.0
    tokens: int = 0
    calls: int = 0


@dataclass
class Trace:
    command: str
    username: str
    started: float = field(default_factory=time.time)
    stages: list[Stage] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    _t0: float = field(default_factory=time.time)

    def stage(self, name: str, detail: str = "", tokens: int = 0, calls: int = 0) -> None:
        now = time.time()
        self.stages.append(Stage(name, detail, now - self._t0, tokens, calls))
        self._t0 = now

    def note(self, text: str) -> None:
        self.notes.append(text)


def _status_counts(claims) -> dict[str, int]:
    out: dict[str, int] = {}
    for c in claims:
        key = c.status.value if hasattr(c.status, "value") else str(c.status)
        out[key] = out.get(key, 0) + 1
    return out


def render(trace: Trace, state: dict, cfg, eval_summary: dict | None = None) -> str:
    claims = state.get("claims") or []
    counts = _status_counts(claims)
    verified = counts.get("verified", 0)
    needs = counts.get("needs_measure", 0)
    rejected = counts.get("rejected", 0)
    total = len(claims) or 1
    usage = state.get("usage") or {}
    groq = usage.get("groq") or {}
    jev = usage.get("jev") or {}

    L: list[str] = []
    w = L.append

    w("# Run trace\n")
    w("A complete record of one pipeline run, written so the output can be "
      "**audited rather than trusted**. Every claim the verifier rejected is "
      "listed below with its reason — that section is the most informative "
      "part of this file.\n")
    w(f"- **Command:** `{trace.command}`")
    w(f"- **Account:** `{trace.username}` (public repositories only)")
    w(f"- **Python:** {platform.python_version()} on {platform.system()}")
    w(f"- **Models:** small/mid `{cfg.model_for_tier('small')}`, "
      f"big `{cfg.model_for_tier('big')}`")
    lim = cfg.limits_for(cfg.model_for_tier("small"))
    w(f"- **Free-tier ceiling:** {lim.tpm:,} tokens/min, {lim.rpd:,} requests/day "
      f"(per model)\n")

    w("## Stages\n")
    w("| Stage | What happened | Tokens | Seconds |")
    w("|---|---|---:|---:|")
    for s in trace.stages:
        w(f"| {s.name} | {s.detail} | {s.tokens or '—'} | {s.seconds:.1f} |")
    w("")

    w("## Headline numbers\n")
    w(f"- **{verified}/{total} claims verified** ({verified / total:.0%}); "
      f"{needs} need measurement; {rejected} rejected outright.")
    w(f"- **{len(state.get('resumes') or {})} resumes** built from verified claims only.")
    w(f"- **Groq:** {groq.get('total', 0):,} tokens across {groq.get('calls', 0)} node(s).")
    if jev:
        w(f"- **Jev:** {jev.get('input_tokens', 0):,} input tokens, "
          f"${jev.get('cost_usd', 0):.5f} (output tokens are free).")
    if usage.get("cache_hit_rate") is not None:
        w(f"- **Cache hit rate:** {usage['cache_hit_rate']:.0%} — an unchanged repo "
          f"costs nothing on a re-run.")
    w(f"- **Wall clock:** {usage.get('elapsed_s', 0):.0f}s, most of it waiting on "
      f"the {lim.tpm:,} tokens/minute limit.\n")

    sel = state.get("selected") or []
    if sel:
        w("## Repository selection\n")
        w("Scored by Jev (a typed decision model), not by Groq — this costs "
          "fractions of a cent and leaves the scarce Groq budget for text.\n")
        w("| Repo | Claimable | Tier | Language |")
        w("|---|---:|---|---|")
        for r in sel:
            w(f"| `{r.name}` | {r.claimable:.2f} | {r.tier or '—'} | {r.language or '—'} |")
        w("")

    w("## Claim verdicts\n")
    w("Verification is **plain Python**, not a prompt. Every number in a claim "
      "must appear in that repository's evidence bundle; every quote must be "
      "present *and* relevant to the claim it supports.\n")

    if rejected:
        w(f"### Rejected ({rejected})\n")
        w("These were proposed by the model and thrown out. They never reach a resume.\n")
        for c in claims:
            if str(getattr(c.status, "value", c.status)) == "rejected":
                w(f"- `{c.repo}` — {c.text}")
                w(f"  - *{c.reason}*")
        w("")
    if needs:
        w(f"### Needs measurement ({needs})\n")
        w("Plausible but unproven. The agent refused to attach a number it could "
          "not find, and emitted a measurement task instead.\n")
        for c in claims:
            if str(getattr(c.status, "value", c.status)) == "needs_measure":
                w(f"- `{c.repo}` — {c.text}")
                w(f"  - *{c.reason}* → **`{c.measure_hint}`**")
        w("")

    scores = state.get("scores") or {}
    if scores:
        w("## Resume scores\n")
        w("Deterministic 100-point ATS rubric, no model involved.\n")
        w("| Role | Score | Keywords | Role fit | Numbers |")
        w("|---|---:|---:|---:|---:|")
        for k, s in sorted(scores.items(), key=lambda kv: -kv[1].total):
            c = s.components
            w(f"| {k} | **{s.total}** | {c.get('keywords', 0)} | "
              f"{c.get('role_fit', 0)} | {c.get('numbers', 0)} |")
        w("")

    dropped = [e for e in (state.get("errors") or []) if "dropped bullet" in e]
    w("### Bullets dropped during resume writing\n")
    if dropped:
        w("The generator is a second chance to invent: it can merge two claims "
          "into a total that appeared in neither. Every bullet is re-checked "
          "against the claims it cites.\n")
        for d in dropped[:20]:
            w(f"- {d}")
    else:
        w("None. Every generated bullet traced back to a verified claim.")
    w("")

    plan = state.get("plan") or []
    if plan:
        w("## Improvement plan\n")
        w(f"{len(plan)} steps, ~{sum(s.hours for s in plan):.0f} hours, "
          f"+{sum(s.ats_gain for s in plan):.1f} ATS points. The gain figure comes "
          f"from the same rubric that produced the score, not from the model.\n")
        w("| Step | Repo | Hours | ATS gain |")
        w("|---|---|---:|---:|")
        for s in plan:
            w(f"| {s.title} | `{s.repo or 'new work'}` | {s.hours:.0f} | +{s.ats_gain} |")
        w("")

    safety = state.get("safety") or []
    w("## Security findings in the scanned repositories\n")
    if safety:
        for f in safety:
            w(f"- **[{f.severity}]** `{f.repo}/{f.path}` — {f.detail}")
    else:
        w("None.")
    w("")

    if eval_summary:
        w("## Measured quality\n")
        w("| Metric | Result |")
        w("|---|---:|")
        for k, v in eval_summary.items():
            w(f"| {k} | {v} |")
        w("")

    if trace.notes:
        w("## Notes\n")
        for n in trace.notes:
            w(f"- {n}")
        w("")

    w("## How to reproduce\n")
    w("```bash")
    w("pip install -e \".[dev]\"")
    w("cp .env.sample .env   # add GROQ_API_KEY, TYPESAFE_API_KEY, GITHUB_PERSONAL_ACCESS_TOKEN")
    w("rba doctor")
    w(f"{trace.command}")
    w("```\n")
    w("Summaries are cached by `(repo, head SHA, prompt version, model)`, so a "
      "second run over unchanged repositories spends no tokens at all.\n")
    return "\n".join(L) + "\n"


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
