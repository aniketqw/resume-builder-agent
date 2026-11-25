"""Tests for the opt-in scoring fixes. No network, no embedding model."""
from rba.ats import improved
from rba.ats.lexical import bm25_similarity, tokenize
from rba.ats.corpus import background_corpus
from rba.resume.document import Bullet, Entry, ResumeDoc, Section

JD = ("We need a Python engineer with AWS and Docker experience. "
      "You will design and maintain data pipelines.")


def _doc(*texts):
    return ResumeDoc(sections=[Section("Resume", entries=[
        Entry(title="e", bullets=[Bullet(t) for t in texts])])])


def test_defaults_are_off():
    f = improved.Fixes()
    assert not any([f.validity_guard, f.capped_lexical, f.topk_pooling,
                    f.stuffing_guard, f.llm_skills, f.requirement_match])
    assert all([improved.Fixes.all().stuffing_guard, improved.Fixes.all().llm_skills])


def test_existing_modules_do_not_import_improved():
    import importlib
    for name in ("rba.ats.batch", "rba.ats.fusion", "rba.ats.score", "rba.resume.attribution"):
        src = open(importlib.import_module(name).__file__).read()
        assert "improved" not in src, f"{name} must not depend on the opt-in module"


def test_stuffed_line_flagged_real_sentence_not():
    terms = set(tokenize(JD))
    assert improved.looks_stuffed("python engineer aws docker pipelines data design maintain experience need", terms)
    assert not improved.looks_stuffed("Built and maintained data pipelines in Python on AWS for the finance team", terms)
    assert not improved.looks_stuffed("Python AWS", terms)          # too short to judge


def test_drop_stuffed_copies_and_disables():
    doc = _doc("Built data pipelines in Python and deployed them on AWS for the team",
               "python engineer aws docker pipelines data design maintain experience need")
    out, removed = improved.drop_stuffed(doc, JD)
    assert removed == 1
    assert [b.enabled for b in out.all_bullets()] == [True, False]
    assert all(b.enabled for b in doc.all_bullets())               # original untouched


def test_capped_bm25_does_not_reward_made_up_words():
    gib = "Zerbits flibberwock snazzled wibberwonks drimbly squoil ignups crangled thoppo riffle"
    resume = "Sales manager with experience in retail operations and customer service across regions"
    capped = improved.bm25_capped(resume + " " + gib, gib)
    assert 0.0 <= capped <= 1.0
    assert improved.bm25_capped(resume, JD) <= 1.0
    assert improved.bm25_capped("", JD) == 0.0


def test_requirement_split():
    reqs = improved.jd_requirements("Skills:\n- 5 years of Python experience\n- AWS\n- Strong SQL and data modelling skills.")
    assert reqs == ["5 years of Python experience", "Strong SQL and data modelling skills."]


def test_skills_from_llm_counts_only_evidenced_skills():
    fake = lambda jd: {"required": ["Python", "Docker"], "preferred": ["Kubernetes"]}
    full = improved.skills_from_llm("Built Python services in Docker containers", JD, extract=fake)
    part = improved.skills_from_llm("Built Python services", JD, extract=fake)
    none = improved.skills_from_llm("Managed a retail store", JD, extract=fake)
    assert full > part > none == 0.0
    assert improved.skills_from_llm("anything", JD, extract=lambda jd: None) is None
    assert improved.skills_from_llm("anything", JD, extract=lambda jd: {"required": [], "preferred": []}) is None


def test_parse_skill_json_tolerates_fences_and_junk():
    p = improved._parse_skill_json('```json\n{"required": ["A b", 3, "x"], "preferred": ["Go"]}\n```')
    assert p == {"required": ["A b"], "preferred": ["Go"]}
    assert improved._parse_skill_json("no json here") is None


def test_fuse_redistributes_missing_channels():
    raw, used = improved.fuse({"dense": 0.8, "lexical": 0.4, "skills": None})
    assert abs(raw - (0.5 * 0.8 + 0.2 * 0.4) / 0.7) < 1e-9 and "skills" not in used
    assert improved.fuse({"dense": None})[0] == 0.0
    raw2, _ = improved.fuse({"dense": 1.0, "req": 0.0}, {"dense": 0.5, "req": 0.5})
    assert raw2 == 0.5
