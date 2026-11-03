"""Standing in for a job description when the user has not got one.

The fused scorer compares a resume against *a text*: `rescore(doc, jd_text,
required_skills=..., preferred_skills=...)`. It has no concept of a role. So
picking "Software Engineer" with an empty job-description box gave the scorer
nothing to compare against, and the API refused with a 400.

v1 solved this in nine lines (`app.py:126-137`) and this is that fix, lifted
out of the Streamlit page so both the API and its tests can use it: sort the
role's keywords by descending weight, make a short posting-shaped sentence out
of the top of the list, and split the rest into required and preferred skills.

Two things to be clear about:

* **The number is weaker than one from a real posting.** A keyword profile is
  not prose and a real job advert carries context this cannot. `CAVEAT` is the
  wording the UI must show beside it, and the synthesised text is returned so
  the page can display exactly what was scored against. A stand-in that cannot
  be inspected is indistinguishable from a real result, which is the failure
  this module is trying not to introduce.
* **It activates the skills channel.** Without `required_skills` the skills
  channel is absent entirely and `convex_fuse` redistributes its 0.3 across the
  other two -- so a role-derived score uses all three channels where a bare
  job-description paste often uses two.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..scoring import Role, load_roles

#: How many of the highest-weighted terms go into the synthesised posting.
JD_TERMS = 18
#: The top slice treated as hard requirements...
REQUIRED_TERMS = 8
#: ...and the next slice as nice-to-haves. Beyond this the weights are low
#: enough that calling them "preferred" overstates them.
PREFERRED_TERMS = 14

CAVEAT = ("No job description, so this uses the role's keyword profile as a "
          "stand-in. A real posting scores more meaningfully.")


@dataclass(frozen=True)
class RoleTarget:
    role_key: str
    title: str
    jd_text: str
    required_skills: list[str]
    preferred_skills: list[str]
    synthesised: bool = True
    caveat: str = CAVEAT

    def to_dict(self) -> dict:
        return {"role_key": self.role_key, "title": self.title,
                "jd_text": self.jd_text,
                "required_skills": list(self.required_skills),
                "preferred_skills": list(self.preferred_skills),
                "synthesised": self.synthesised, "caveat": self.caveat}


def ranked_keywords(role: Role) -> list[str]:
    """Role terms, heaviest first.

    Ties are broken alphabetically so the synthesised text is stable across
    runs -- an unstable job description would change the score for no reason
    and invalidate every cached delta.
    """
    return [term for term, _weight in
            sorted(role.keywords.items(), key=lambda kv: (-kv[1], kv[0]))]


def role_to_target(role: Role, role_key: str = "") -> RoleTarget:
    """A scoreable target built from a role definition."""
    top = ranked_keywords(role)
    jd_text = f"{role.title}. {role.headline}. " + ", ".join(top[:JD_TERMS])
    return RoleTarget(
        role_key=role_key or getattr(role, "key", ""),
        title=role.title,
        jd_text=jd_text,
        required_skills=top[:REQUIRED_TERMS],
        preferred_skills=top[REQUIRED_TERMS:PREFERRED_TERMS],
    )


def target_for_key(role_key: str) -> RoleTarget | None:
    """`None` for an unknown key, so a stale role in a saved session degrades
    to "no target" rather than raising."""
    role = load_roles().get(role_key)
    return None if role is None else role_to_target(role, role_key)
