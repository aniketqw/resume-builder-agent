"""Skill normalisation: mapping surface forms to one canonical name.

"JS", "JavaScript" and "ECMAScript" must score as one skill, or a resume is
penalised for word choice. The question is what to normalise *against*.

**Why not ESCO.** The plan was to use ESCO, the EU occupational taxonomy —
~13,900 skills against O*NET's 177, multilingual, officially maintained. I
queried its live API for the skills this tool actually sees:

===============  ==========================================================
python           Python (computer programming)
postgresql       PostgreSQL
javascript       JavaScript
kubernetes       **no results**
docker           **no results**
pytorch          **no results**
tensorflow       **no results**
aws              **no results**
terraform        **no results**
kafka            **no results**
react            "react to emergency situations in a live performance
                 environment"
===============  ==========================================================

ESCO is built around occupations, so it has "use scripting programming" but
not "Kubernetes". Embedding-nearest-neighbour cannot rescue this: it is a
coverage gap, not a matching problem, and the `react` result shows the failure
mode is not a miss but a confident wrong answer that would silently corrupt a
score.

**So: a curated alias table.** Smaller claim, better accuracy, no 21MB
embedding matrix, no network call, and every mapping is readable in this file.
Unknown skills pass through normalised-but-unmapped rather than being forced
onto a wrong canonical id.
"""
from __future__ import annotations

import re
from functools import lru_cache

# canonical -> aliases. Canonical names are the form a human would write.
ALIASES: dict[str, tuple[str, ...]] = {
    "javascript": ("js", "ecmascript", "es6", "es2015", "vanilla javascript"),
    "typescript": ("ts",),
    "python": ("python3", "py", "cpython"),
    "kubernetes": ("k8s", "kube"),
    "docker": ("docker compose", "docker-compose", "containerisation",
               "containerization", "containers"),
    "postgresql": ("postgres", "psql", "postgre"),
    "mongodb": ("mongo",),
    "amazon web services": ("aws", "amazon aws"),
    "google cloud platform": ("gcp", "google cloud"),
    "microsoft azure": ("azure",),
    "continuous integration": ("ci", "ci/cd", "cicd", "continuous delivery",
                               "continuous deployment"),
    "machine learning": ("ml",),
    "deep learning": ("dl", "neural networks", "neural nets"),
    "natural language processing": ("nlp",),
    "computer vision": ("cv", "image processing"),
    "large language models": ("llm", "llms", "genai", "generative ai"),
    "retrieval augmented generation": ("rag",),
    "pytorch": ("torch",),
    "tensorflow": ("tf", "tensorflow2", "keras"),
    "scikit-learn": ("sklearn", "scikit learn"),
    "pandas": ("pd",),
    "numpy": ("np",),
    "rest api": ("rest", "restful", "restful api", "rest apis", "http api"),
    "graphql": ("gql",),
    "postgres administration": (),
    "sql": ("structured query language",),
    "nosql": (),
    "git": ("github", "gitlab", "version control"),
    "linux": ("unix", "bash", "shell scripting"),
    "terraform": ("infrastructure as code", "iac"),
    "apache kafka": ("kafka",),
    "redis": (),
    "elasticsearch": ("elastic search", "elk"),
    "react": ("react.js", "reactjs"),
    "node.js": ("node", "nodejs"),
    "next.js": ("nextjs",),
    "django": (),
    "flask": (),
    "fastapi": (),
    "spring boot": ("spring", "springboot"),
    "java": (),
    "go": ("golang",),
    "rust": (),
    "c++": ("cpp", "cplusplus"),
    "c#": ("csharp", "c sharp"),
    "unit testing": ("pytest", "unittest", "jest", "testing", "test automation"),
    "agile": ("scrum", "kanban"),
    "data analysis": ("data analytics", "analytics"),
    "data engineering": ("etl", "elt", "data pipelines"),
    "microservices": ("microservice architecture",),
    "websockets": ("websocket",),
    "langchain": (),
    "langgraph": (),
    "model context protocol": ("mcp",),
}

# Reverse index, built once.
_LOOKUP: dict[str, str] = {}
for _canonical, _aliases in ALIASES.items():
    _LOOKUP[_canonical] = _canonical
    for _alias in _aliases:
        _LOOKUP[_alias] = _canonical

_PUNCT = re.compile(r"[^\w+#./-]+")


@lru_cache(maxsize=4096)
def canonical(skill: str) -> str:
    """Canonical name for a skill, or the cleaned input if we do not know it.

    Passing an unknown skill through unchanged is deliberate. Forcing it onto
    the nearest known name is how "react" becomes "react to emergency
    situations".
    """
    cleaned = _PUNCT.sub(" ", (skill or "").lower()).strip()
    cleaned = " ".join(cleaned.split())
    if not cleaned:
        return ""
    if cleaned in _LOOKUP:
        return _LOOKUP[cleaned]
    # "experience with kubernetes" -> kubernetes, but only on an exact token
    # run, never a fuzzy nearest neighbour.
    words = cleaned.split()
    for size in range(min(4, len(words)), 0, -1):
        for start in range(len(words) - size + 1):
            phrase = " ".join(words[start:start + size])
            if phrase in _LOOKUP:
                return _LOOKUP[phrase]
    return cleaned


def normalise_all(skills) -> list[str]:
    """Canonicalise a collection, de-duplicated, order preserved."""
    seen: dict[str, None] = {}
    for skill in skills or []:
        name = canonical(skill)
        if name:
            seen.setdefault(name, None)
    return list(seen)


def known_skills() -> set[str]:
    return set(ALIASES)


def coverage(skills) -> tuple[list[str], list[str]]:
    """Split into (recognised, unrecognised) — for showing what we did know."""
    known: list[str] = []
    unknown: list[str] = []
    for skill in normalise_all(skills):
        (known if skill in ALIASES else unknown).append(skill)
    return known, unknown
