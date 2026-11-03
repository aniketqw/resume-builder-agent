"""Actionable advice, deliberately kept out of the score.

The artifact this is modelled on shows seven bars that sum to 100 — keywords
44, role fit 18, numbers 12, sections 10, contact 6, length 6, verbs 4. Those
weights were invented. Nobody measured that a missing phone number is worth
exactly six points, and presenting a number built from invented weights as
"your ATS score" is the same category of error as inventing an achievement.

So the weights are not used here. The headline number stays the three-channel
fused score, which is calibrated against a reference set. What these functions
recover is the part of the rubric that was genuinely useful: it could say
*which keyword is missing* and *that the resume is too long*. That is advice,
and advice does not need a weight to be worth reading.

Everything below reuses the detection logic already in `rba.scoring`, so the
two can never drift apart on what counts as a number or a strong verb.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..scoring import EMAIL, GITHUB, MEASURE_TAG, PHONE, QUANTITY, YEAR, Role, _normalise
from .document import KeyValues, ResumeDoc

# Where a term appearing only here earns partial credit: a skill named in a
# list but never demonstrated in a bullet is weaker evidence than one a project
# actually used, which is a distinction the artifact draws and is right to.
CONTEXT_SECTIONS = ("TECHNICAL SKILLS", "SKILLS", "SUMMARY")


@dataclass(frozen=True)
class Diagnostic:
    key: str
    label: str
    status: str                      # "ok" | "warn" | "info"
    detail: str = ""
    items: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def _bullet_texts(doc: ResumeDoc) -> list[str]:
    return [b.text for b in doc.enabled()]


def _context_text(doc: ResumeDoc) -> str:
    """Everything that is not a bullet: skills rows, the summary, headings."""
    parts = [doc.summary]
    for section in doc.sections:
        if isinstance(section.body, KeyValues) or \
                section.title.upper() in CONTEXT_SECTIONS:
            if section.body is not None:
                parts.append(section.body.as_text())
            parts.append(section.text)
    return "\n".join(p for p in parts if p)


def keyword_coverage(doc: ResumeDoc, role: Role) -> tuple[list[str], list[str], list[str]]:
    """Role terms demonstrated in a bullet, named only in passing, or absent.

    The middle group is the useful one: "RAG" sitting in a skills list and
    never appearing in a project is a gap a reader will notice.
    """
    bullets = _normalise("\n".join(_bullet_texts(doc)))
    context = _normalise(_context_text(doc))
    demonstrated, mentioned, missing = [], [], []
    for term in role.keywords:
        if _contains(term, bullets):
            demonstrated.append(term)
        elif _contains(term, context):
            mentioned.append(term)
        else:
            missing.append(term)
    return demonstrated, mentioned, missing


def _contains(term: str, text: str) -> bool:
    return bool(re.search(rf"(?<![\w/+#-]){re.escape(term)}(?![\w/+#-])", text))


def quantified_bullets(doc: ResumeDoc) -> tuple[list[str], list[str]]:
    """Bullets carrying a real quantity, and those that do not.

    A `[MEASURE: ...]` placeholder counts as unquantified, so an unmeasured
    claim is never credited as if it had a number.
    """
    with_number, without = [], []
    for text in _bullet_texts(doc):
        stripped = MEASURE_TAG.sub("", text)
        found = [m.group(0) for m in QUANTITY.finditer(YEAR.sub(" ", stripped))]
        (with_number if any(c.strip() for c in found) else without).append(text)
    return with_number, without


def weak_openers(doc: ResumeDoc, role: Role) -> list[str]:
    """Bullets that do not start with one of the role's action verbs."""
    weak = []
    for text in _bullet_texts(doc):
        first = re.sub(r"^[^A-Za-z]+", "", text).split(" ")[0].lower().rstrip(".,:;")
        if first and first not in role.action_verbs:
            weak.append(text)
    return weak


def missing_contact(doc: ResumeDoc) -> list[str]:
    text = doc.as_text()
    absent = []
    if not EMAIL.search(text):
        absent.append("email address")
    if not PHONE.search(text):
        absent.append("phone number")
    if not GITHUB.search(text):
        absent.append("GitHub link")
    return absent


# A role asks for "education"; a real resume calls it ACADEMIC DETAILS. Matching
# on the exact word reported a complete resume as missing three sections.
SECTION_SYNONYMS = {
    "education": ("academic", "education", "qualification", "schooling"),
    "skills": ("skill", "technical", "technology", "competenc"),
    "experience": ("experience", "internship", "employment", "work",
                   "position of responsibility", "professional"),
    "projects": ("project", "portfolio"),
    "summary": ("summary", "objective", "profile", "about"),
    "certifications": ("certification", "certificate", "credential"),
}


def missing_sections(doc: ResumeDoc, role: Role) -> list[str]:
    titles = [s.title.lower() for s in doc.sections]
    if doc.summary:
        titles.append("summary")
    missing = []
    for required in role.required_sections:
        needles = SECTION_SYNONYMS.get(required.lower(), (required.lower(),))
        if not any(n in title for title in titles for n in needles):
            missing.append(required)
    return missing


def unsupported_claims(doc: ResumeDoc) -> list[str]:
    return [b.text for b in doc.enabled() if not b.verified]


def word_count(doc: ResumeDoc) -> int:
    from .render import RenderOptions, to_plain

    return len(to_plain(doc, RenderOptions()).split())


def diagnose(doc: ResumeDoc, role: Role) -> list[Diagnostic]:
    """Everything worth telling the user, ordered by how actionable it is."""
    out: list[Diagnostic] = []

    demonstrated, mentioned, missing = keyword_coverage(doc, role)
    if missing:
        out.append(Diagnostic(
            "keywords_missing", "Role keywords not present", "warn",
            "Absent from the resume entirely. Only add one if a project "
            "genuinely used it.", missing))
    if mentioned:
        out.append(Diagnostic(
            "keywords_listed_only", "Named but never demonstrated", "info",
            "These appear in your skills or summary but in no project bullet. "
            "A reader treats a listed skill as weaker than a used one.",
            mentioned))
    if demonstrated and not missing:
        out.append(Diagnostic(
            "keywords_ok", "Every role keyword appears in a project", "ok",
            items=demonstrated))

    with_number, without = quantified_bullets(doc)
    total = len(with_number) + len(without)
    if total:
        share = len(with_number) / total
        out.append(Diagnostic(
            "numbers", f"{len(with_number)} of {total} bullets carry a number",
            "ok" if share >= 0.5 else "warn",
            "Numbers you can defend. Do not invent one to raise this — an "
            "unverifiable figure costs you the interview, not the screen."))

    unsupported = unsupported_claims(doc)
    out.append(Diagnostic(
        "verified", "No unsupported claims" if not unsupported
        else f"{len(unsupported)} claim(s) not backed by evidence",
        "ok" if not unsupported else "warn",
        "Every enabled line traces to your resume or to repository evidence."
        if not unsupported else "These need measuring before you send this.",
        unsupported))

    words = word_count(doc)
    low, high = role.target_words
    out.append(Diagnostic(
        "length", f"{words} words",
        "ok" if low <= words <= high else "warn",
        f"Target for this role is {low}\u2013{high}." if not low <= words <= high
        else f"Within the {low}\u2013{high} band for this role."))

    absent = missing_contact(doc)
    if absent:
        out.append(Diagnostic("contact", "Contact details incomplete", "warn",
                              "Missing: " + ", ".join(absent), absent))

    gaps = missing_sections(doc, role)
    if gaps:
        out.append(Diagnostic("sections", "Expected sections missing", "warn",
                              "This role usually shows: " + ", ".join(gaps), gaps))

    weak = weak_openers(doc, role)
    if weak:
        out.append(Diagnostic(
            "verbs", f"{len(weak)} bullet(s) do not open with an action verb",
            "info", "Leading with the verb puts the work before the context.",
            weak))

    return out
