"""Tests for the evidence-grounded scoring core.

These assert *properties* rather than magic numbers wherever possible, because
the weights are a judgement call and should be changeable without rewriting the
suite. What must not change is the behaviour: ordering, bounds, and the
handling of unknown inputs.
"""
import pytest

from rba.ats import (
    CHANNEL_WEIGHTS,
    ChannelScores,
    bm25_similarity,
    convex_fuse,
    ordinal_band,
    theoretical_min_max,
    weighted_jaccard,
)
from rba.ats.lexical import idf, matched_terms, tokenize
from rba.ats.skills import has_skills

JD = "We need a backend engineer with Python, FastAPI, Docker and Kubernetes on AWS."
STRONG = "Built Python FastAPI services deployed with Docker and Kubernetes on AWS."
PARTIAL = "Built Python services and some Docker containers."
IRRELEVANT = "Taught introductory history and organised community events."


# -- tokenizer ------------------------------------------------------------
def test_internal_punctuation_survives_but_trailing_does_not():
    """'node.js' must stay intact; 'aws.' at a sentence end must not."""
    assert tokenize("AWS. node.js ci/cd C++ 3.11") == [
        "aws", "node.js", "ci/cd", "c++", "3.11"]


def test_short_language_names_are_not_stopworded():
    """An aggressive stop list eats 'r' and 'go', which are languages."""
    tokens = tokenize("Experience in R and Go")
    assert "r" in tokens and "go" in tokens


# -- BM25 -----------------------------------------------------------------
def test_relevance_ordering_is_monotonic():
    strong = bm25_similarity(STRONG, JD)
    partial = bm25_similarity(PARTIAL, JD)
    irrelevant = bm25_similarity(IRRELEVANT, JD)
    assert strong > partial > irrelevant
    assert irrelevant == 0.0


def test_a_document_identical_to_the_jd_scores_one():
    """The ceiling is the JD scored against itself, so this is exactly 1.0."""
    assert bm25_similarity(JD, JD) == pytest.approx(1.0)


def test_scores_stay_in_range():
    for text in (STRONG, PARTIAL, IRRELEVANT, "", JD * 50):
        assert 0.0 <= bm25_similarity(text, JD) <= 1.0


def test_empty_inputs_score_zero_not_crash():
    assert bm25_similarity("", JD) == 0.0
    assert bm25_similarity(STRONG, "") == 0.0


def test_idf_never_goes_negative():
    """A term in most of the corpus can otherwise subtract from a score."""
    corpus = [["python"], ["python"], ["python"], ["rust"]]
    assert idf("python", corpus) >= 0.0


def test_matched_terms_are_auditable():
    present, missing = matched_terms(STRONG, JD)
    assert {"python", "docker", "kubernetes", "aws"} <= set(present)
    assert "engineer" in missing


# -- skills ---------------------------------------------------------------
def test_full_coverage_scores_one():
    match = weighted_jaccard({"python", "docker"}, required={"python", "docker"})
    assert match.score == pytest.approx(1.0)
    assert match.missing_required == []


def test_required_outweighs_preferred():
    only_required = weighted_jaccard({"python"}, required={"python"},
                                     preferred={"aws"})
    only_preferred = weighted_jaccard({"aws"}, required={"python"},
                                      preferred={"aws"})
    assert only_required.score > only_preferred.score


def test_extra_skills_are_not_penalised():
    """Symmetric Jaccard would punish a broad resume. Coverage must not."""
    narrow = weighted_jaccard({"python"}, required={"python"})
    broad = weighted_jaccard({"python", "rust", "go", "haskell"},
                             required={"python"})
    assert narrow.score == broad.score == pytest.approx(1.0)


def test_skill_matching_is_case_and_space_insensitive():
    match = weighted_jaccard({"  Machine   Learning "}, required={"machine learning"})
    assert match.score == pytest.approx(1.0)


def test_a_job_with_no_skills_is_unknown_not_zero():
    assert not has_skills(set())
    assert weighted_jaccard({"python"}, required=set()).score == 0.0


def test_summary_is_human_checkable():
    match = weighted_jaccard({"python"}, required={"python", "kubernetes"},
                             preferred={"aws"})
    assert match.summary == "1 of 2 required, 0 of 1 preferred"


# -- fusion ---------------------------------------------------------------
def test_weights_sum_to_one():
    assert sum(CHANNEL_WEIGHTS.values()) == pytest.approx(1.0)


def test_all_channels_present_uses_the_declared_weights():
    fused = convex_fuse(ChannelScores(dense=1.0, lexical=1.0, skills=1.0))
    assert fused.raw == pytest.approx(1.0)
    assert fused.weights_used == pytest.approx(CHANNEL_WEIGHTS)


def test_missing_channel_redistributes_weight_rather_than_scoring_zero():
    """A channel we could not compute is unknown. Scoring it zero would
    penalise the candidate for our missing inputs."""
    fused = convex_fuse(ChannelScores(dense=0.8, lexical=0.8, skills=None))
    assert fused.raw == pytest.approx(0.8)
    assert sum(fused.weights_used.values()) == pytest.approx(1.0)
    assert "skills" not in fused.weights_used


def test_no_channels_at_all_is_zero_not_a_crash():
    fused = convex_fuse(ChannelScores())
    assert fused.raw == 0.0
    assert fused.channels == {}


def test_fused_score_tracks_its_inputs_for_display():
    fused = convex_fuse(ChannelScores(dense=0.9, lexical=0.1, skills=0.5))
    assert set(fused.channels) == {"dense", "lexical", "skills"}
    assert "uncalibrated" in fused.display


# -- normalisation --------------------------------------------------------
def test_theoretical_min_max_clamps():
    assert theoretical_min_max(5, 0, 10) == pytest.approx(0.5)
    assert theoretical_min_max(-1, 0, 10) == 0.0
    assert theoretical_min_max(99, 0, 10) == 1.0


def test_degenerate_bounds_do_not_divide_by_zero():
    assert theoretical_min_max(5, 3, 3) == 0.0


# -- ordinal bands --------------------------------------------------------
@pytest.mark.parametrize("raw,expected", [
    (0.95, "Strong match"), (0.80, "Strong match"),
    (0.70, "Good match"), (0.45, "Partial match"), (0.10, "Limited match"),
])
def test_ordinal_bands_mirror_real_ats_vocabulary(raw, expected):
    """Greenhouse reports Strong/Good/Partial/Limited; Workday A-D. No real
    ATS shows a candidate a 0-100 number, so a band sits beside ours."""
    assert ordinal_band(raw) == expected


# -- end to end -----------------------------------------------------------
def test_a_strong_resume_outscores_an_irrelevant_one_overall():
    def score(text, skills):
        match = weighted_jaccard(skills, required={"python", "docker"})
        return convex_fuse(ChannelScores(
            dense=None, lexical=bm25_similarity(text, JD), skills=match.score)).raw

    assert score(STRONG, {"python", "docker"}) > score(IRRELEVANT, {"teaching"})


# -- skill normalisation --------------------------------------------------
@pytest.mark.parametrize("surface,expected", [
    ("JS", "javascript"), ("ECMAScript", "javascript"),
    ("k8s", "kubernetes"), ("K8S", "kubernetes"),
    ("Postgres", "postgresql"), ("sklearn", "scikit-learn"),
    ("CI/CD", "continuous integration"), ("golang", "go"),
    ("experience with Kubernetes", "kubernetes"),
])
def test_aliases_collapse_to_one_canonical_name(surface, expected):
    from rba.ats.normalise import canonical
    assert canonical(surface) == expected


def test_unknown_skills_pass_through_rather_than_being_forced():
    """ESCO's API answers 'react' with 'react to emergency situations in a
    live performance environment'. A confident wrong answer is worse than no
    answer, so unknown skills stay as themselves."""
    from rba.ats.normalise import canonical
    assert canonical("quantum basket weaving") == "quantum basket weaving"


def test_normalisation_makes_aliases_score_as_one_skill():
    assert weighted_jaccard({"JS"}, required={"ECMAScript"}).score == pytest.approx(1.0)
    assert weighted_jaccard({"k8s"}, required={"Kubernetes"}).score == pytest.approx(1.0)


def test_coverage_reports_what_we_recognised():
    from rba.ats.normalise import coverage
    known, unknown = coverage(["JS", "k8s", "quantum basket weaving"])
    assert known == ["javascript", "kubernetes"]
    assert unknown == ["quantum basket weaving"]


# -- background corpus ----------------------------------------------------
def test_background_corpus_is_bundled_and_diverse():
    """BM25's IDF is meaningless without one: the same resume scored 0.180
    with no corpus and 0.476 against a background."""
    from rba.ats.corpus import background_corpus, families
    corpus = background_corpus()
    assert len(corpus) >= 15
    # Non-software roles matter: they keep "Python" rare, as it is in the
    # world, rather than universal as it would be in an all-engineering sample.
    assert {"teacher", "nurse", "accountant", "marketing"} <= set(families())


def test_corpus_raises_bm25_for_a_relevant_resume():
    from rba.ats.corpus import background_corpus
    jd = "Backend Engineer. Required: Python, Docker, Kubernetes, REST APIs."
    resume = "Built Python services with Docker and Kubernetes, exposing REST APIs."
    bare = bm25_similarity(resume, jd)
    backed = bm25_similarity(resume, jd,
                             corpus=list(background_corpus()) + [jd, resume])
    assert backed > bare


# -- end-to-end scoring ---------------------------------------------------
JD_BACKEND = ("Backend Engineer. Required: Python, Docker, Kubernetes, "
              "REST APIs. Preferred: AWS, Kafka.")
REQUIRED = ["Python", "Docker", "Kubernetes", "REST API"]


def _score(text, skills, **kw):
    from rba.ats import score_resume_against_jd
    return score_resume_against_jd(text, JD_BACKEND, resume_skills=skills,
                                   required_skills=REQUIRED,
                                   preferred_skills=["AWS", "Kafka"],
                                   use_dense=False, **kw)


def test_relevance_ordering_end_to_end():
    strong = _score("Built Python FastAPI services with Docker on Kubernetes, "
                    "designing REST APIs.", ["Python", "Docker", "k8s", "REST"])
    partial = _score("Wrote Python scripts and some Docker files.",
                     ["Python", "Docker"])
    wrong = _score("Taught history and organised community events.",
                   ["lesson planning"])
    assert strong.score_100 > partial.score_100 > wrong.score_100


def test_report_names_the_missing_required_skills():
    report = _score("Wrote Python scripts.", ["Python"])
    assert "kubernetes" in report.skills.missing_required
    assert any("Required skills not evidenced" in line
               for line in report.explain())


def test_disabling_dense_is_recorded_not_silent():
    report = _score("Python and Docker work.", ["Python", "Docker"])
    assert "dense" not in report.fused.channels
    assert any("Semantic channel disabled" in note for note in report.notes)


def test_missing_structured_skills_skips_the_channel():
    from rba.ats import score_resume_against_jd
    report = score_resume_against_jd("Python and Docker work.", JD_BACKEND,
                                     use_dense=False)
    assert "skills" not in report.fused.channels
    assert any("skipped rather than scored zero" in n for n in report.notes)


def test_explain_is_human_readable_and_weighted():
    report = _score("Python Docker Kubernetes REST APIs.",
                    ["Python", "Docker", "Kubernetes", "REST API"])
    text = " ".join(report.explain())
    assert "Keyword/BM25 match" in text and "weight" in text


def test_score_stays_within_bounds():
    for text, skills in (("", []), (JD_BACKEND, REQUIRED), ("x" * 5000, [])):
        assert 0.0 <= _score(text, skills).score_100 <= 100.0
