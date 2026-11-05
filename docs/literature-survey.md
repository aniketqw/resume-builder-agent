# Improving a resume's match score: a literature survey

What is known about matching a résumé to a job description, what this project
implements, and what it could do next.

---

## 1. Scope and method

This surveys the whole pipeline — parsing, matching, generation, evaluation —
because the score a candidate sees is the product of all four, and a gain in one
can be erased by a loss in another.

**How the citations were made.** Every paper cited here was opened and the
specific claim located. Each entry gives the **section, table or figure** the
claim comes from, with the arXiv id or DOI. Page numbers are given only where
the published version has a stable page range (Robertson & Zaragoza); for
preprints they are omitted deliberately, because arXiv pagination and published
pagination differ and a section reference survives both.

Where a number could not be located in the source it is marked **unverified**
rather than repeated. A survey that invents a citation fails exactly the
standard `v1/src/rba/nodes/verify.py` applies to the claims this tool generates
about other people's repositories.

---

## 2. What an applicant tracking system actually is

### 2.1 Prevalence and exclusion

The most solid source on scale is not a retrieval paper. Fuller, Raman,
Sage-Gavin and Hines, *Hidden Workers: Untapped Talent* (Harvard Business School
/ Accenture, September 2021), surveyed **8,000 workers and 2,250 executives**
across the US, UK and Germany. It reports that **99% of Fortune 500 companies**
use automated applicant tracking, and argues that rigid filtering makes a large
population of qualified candidates structurally invisible to employers.

That reframes the problem. A matching system is not only trying to rank well —
it is operating inside a filter that is already known to discard qualified
people.

### 2.2 The folklore problem

Searching for empirical evidence on "ATS-friendly formatting" returns almost
entirely vendor blogs, résumé-writing services and university careers pages, not
studies. The advice may still be sound, but it is largely untested in public,
and the numbers drift as they are retold.

A concrete example found while writing this: a widely-syndicated careers page
states **90%** of Fortune 500 companies use an ATS, where the Harvard/Accenture
source it ultimately traces to says **99%**. Other frequently-repeated figures —
"75% of résumés are rejected before a human sees them", "candidates whose title
matches are 10.6× more likely to be interviewed" — are quoted without a locatable
primary source and are **not** relied on here.

**Practical consequence for this project.** Claims about what an ATS "does" with
formatting should be stated as convention, not evidence. `v1/src/rba/resume/render/plain.py:5-6`
currently says hyphen bullets are "the shape the research consistently finds
survives extraction" and names no source — that wording overstates what can be
shown and should be softened to describe the deterministic property it actually
has (no tables, no columns, no glyphs that a text extractor can mangle).

---

## 3. Parsing and extraction

Nothing downstream can beat the quality of the parse.

### 3.1 Résumé parsing is close to solved

Zhu et al., *Layout-Aware Parsing Meets Efficient LLMs: A Unified, Scalable
Framework for Resume Information Extraction and Evaluation* (arXiv 2510.09722,
deployed at Alibaba). **Table 2** on their RealResume set: zero-shot LLMs inside
a layout-aware pipeline reach **0.954 (GPT-4o)** and **0.959 (Claude-4)** F1,
and a fine-tuned 0.6B model reaches **0.964**. They also report the pipeline
itself matters: integrating Claude-4 into the layout-aware pipeline raises F1
from **0.919 to 0.959** — the layout step is worth ~4 points on its own.

### 3.2 Deciding what counts as a skill is not

Nguyen, Zhang, Montariol and Bosselut, *Rethinking Skill Extraction in the Job
Market Domain using Large Language Models* (arXiv 2402.03832). **Table 3, §5.2**:
on SkillSpan, GPT-4 in an extraction setting scores **27.8** strict-span F1
against a supervised ESCOXLM-R baseline at **62.6** — a 34.8-point gap. The
paper notes LLMs "can better handle syntactically complex skill mentions" while
remaining well behind supervised models overall.

So the two halves of extraction are in very different states, and a system that
uses an LLM for both is strong on one and weak on the other.

### 3.3 Taxonomies: an empirical note from this repository

The obvious move is a standard taxonomy — ESCO has ~13,900 skills against
O*NET's 177 and is multilingual and officially maintained. This project queried
ESCO's live API for the skills it actually encounters and recorded the result in
`v1/src/rba/ats/normalise.py:6-34`: **no results** for kubernetes, docker,
pytorch, tensorflow, aws, terraform or kafka, and `react` returning *"react to
emergency situations in a live performance environment"*.

That is primary evidence, not a citation, and it is the reason the code uses a
curated alias table instead. It also bounds the approach: an alias table of ~50
canonical entries generalises poorly outside software, which §8 returns to.

---

## 4. Matching and ranking

### 4.1 Lexical retrieval

Robertson and Zaragoza, *The Probabilistic Relevance Framework: BM25 and
Beyond*, **Foundations and Trends in Information Retrieval 3(4), pp. 333–389
(2009)** — the standard reference for BM25 and the saturation and
length-normalisation behaviour that `k1` and `b` control.

This project implements Okapi BM25 with Robertson/Sparck-Jones IDF, `k1 = 1.5`,
`b = 0.75` (`v1/src/rba/ats/lexical.py:48-62`), with IDF floored at zero so a
term in more than half the corpus cannot go negative.

### 4.2 Dense retrieval, and its zero-shot weakness

Thakur, Reimers, Rücklé, Srivastava and Gurevych, *BEIR: A Heterogeneous
Benchmark for Zero-shot Evaluation of Information Retrieval Models*
(arXiv 2104.08663). **18 datasets, 9 tasks**; main comparison in **§5, Table 2**.
The finding that matters most here:

> "Dense retrieval models (esp. ANCE and TAS-B), that map queries and documents
> independently to vector spaces, perform strongly on certain datasets, while on
> many other datasets perform significantly worse than BM25."

A résumé/JD pair is zero-shot by construction — there is no in-domain training
set — so this is the regime BEIR describes, and it is the single strongest
argument for not shipping a dense-only scorer.

Xiao, Liu, Zhang, Muennighoff, Lian and Nie, *C-Pack: Packed Resources for
General Chinese Embeddings* (arXiv 2309.07597) introduces the BGE family.
**Table 5** reports English MTEB averages: BGE-large 64.23, base 63.55,
**small 62.17** at 384 dimensions.

> **A caveat on the figures in this repository.** `v1/src/rba/ats/dense.py:5-9`
> cites "BEIR retrieval 43.81 (MiniLM) vs 51.68 (bge-small)". Those are MTEB
> *leaderboard* retrieval averages, not numbers from the C-Pack paper, and
> leaderboard values change as the benchmark is re-run. The accompanying point —
> that the widely-quoted "MiniLM 56.3" is a 56-task average and not a retrieval
> score — is correct and is the more important half of the claim.

### 4.3 Hybrid fusion — the load-bearing citation

Bruch, Gai and Ingber, *An Analysis of Fusion Functions for Hybrid Retrieval*
(arXiv 2210.11934; ACM TOIS, doi 10.1145/3596512).

**Table 2** — convex combination (TM2C2) against Reciprocal Rank Fusion, nDCG@1000:

| Dataset | Convex combination | RRF |
|---|---|---|
| MS MARCO | **0.454** | 0.425 |
| Natural Questions | **0.542** | 0.514 |

**§5.1** on why RRF's tuning does not travel:

> "While tuning a parametric RRF does indeed lead to gains in NDCG on in-domain
> datasets, the tuned function does not generalize well to out-of-domain
> datasets."

**§4.2** establishes that the choice of linear normalisation is rank-equivalent —
"there always exist convex combinations of scores normalized by min-max,
standard score, or any other linear transformation that are rank-equivalent" —
which is why normalising by theoretical bounds rather than observed min/max
costs nothing in ranking and carries one fewer data-dependent statistic.

This is the evidence behind the project's fusion *form*. The code is careful to
claim no more than that: `v1/src/rba/ats/fusion.py:23-24` states that the
particular weights are "a judgement call and are reported as such in the UI".

### 4.4 Cross-encoder re-ranking

Nogueira and Cho, *Passage Re-ranking with BERT* (arXiv 1901.04085). **Table 1**:
BERT-LARGE as a cross-encoder reaches **36.5 MRR@10** on the MS MARCO dev set
against **16.7** for untuned BM25. Query and document are encoded *jointly*,
which is what a bi-encoder cosine cannot do.

Relevant here as a known-strong technique this project does **not** use (§8.3).

### 4.5 Person–job fit as its own literature

A decade of work treats résumé↔job matching as a supervised problem over
historical application outcomes.

- **PJFNN** — Zhu, Zhu, Xiong, Ma, Xie, Ding and Li, *Person-Job Fit: Adapting
  the Right Talent for the Right Job with Joint Representation Learning*
  (arXiv 1810.04040). **§3.2**: a bipartite CNN projecting postings and résumés
  into a shared latent space, trained on 2M+ résumés, 15,039 postings and 31,928
  successful applications from 2013–2016.
- **APJFNN** — Qin, Zhu, Xu et al., *Enhancing Person-Job Fit for Talent
  Recruitment: An Ability-aware Neural Network Approach* (SIGIR '18,
  arXiv 1812.08947). **§4**: four hierarchical ability-aware attention
  strategies, treating a posting as a set of distinct ability requirements
  rather than one blob of text, and weighting each experience by its
  contribution to a specific requirement.
- **ConFit** — Yu, Zhang and Yu, *ConFit: Improving Resume-Job Matching using
  Data Augmentation and Contrastive Learning* (arXiv 2401.16349). **Tables 3–4**:
  contrastive learning plus paraphrase augmentation, reporting up to **19% and
  31% absolute nDCG@10** gains over baselines including BM25 and OpenAI
  text-ada-002. Its framing of the core obstacle is worth quoting: "Since a
  candidate applies to only a small selection of jobs, interactions between
  resumes and jobs is *very sparse*."

**Why this project does not sit in this line.** Every one of these needs
interaction labels — who applied, who was hired. A single-user tool scoring one
résumé against one posting has no such data, and there is no public English
dataset with real outcome labels. That constraint, not a preference, is what
puts this project in the unsupervised hybrid-retrieval camp.

### 4.6 The shared task and the benchmark

**TalentCLEF 2026** — Gasco, Fabregat, García-Sardiña et al., *Overview of the
TalentCLEF 2026: Skill and Job Title Intelligence for Human Capital Management*
(arXiv 2606.31692). Task A is this problem: rank candidates against a vacancy.
**Table 5, §4.1**: winner `classum` at **MAP 0.7140**, runner-up `QTPride` at
**0.6632**. **§3.1** notes BM25 recurs across submissions "as an exact-matching
mechanism … particularly useful in the HCM domain, where normalized skill names
and labor-market expressions can play an important role", with top teams pairing
dense encoders with lexical retrieval.

This is the direct evidence that hybrid beats either channel alone *in this
domain*, not just on MS MARCO.

**PJB** — Wang, Yang, Li et al., *PJB: A Reasoning-Aware Benchmark for Person-Job
Retrieval* (arXiv 2603.17386): ~200,000 résumés across six industry domains. Its
central warning:

> "performance heterogeneity across industry domains far exceeds the gains from
> module upgrades" … "aggregate scores alone can severely mislead optimization
> decisions."

Which is to say: a single global number, of the kind this tool shows, will
misrank some sectors regardless of how good the modules are.

---

## 5. Calibration and reporting a score

A raw similarity in [0, 1] means nothing to a user. Two options exist: map it to
a reference distribution, or report it as ordinal.

This project does both (`v1/src/rba/ats/calibrate.py`, `fusion.py:33-42`), and
its honest limitation is stated in `calibrate.py:3-13`: proper calibration needs
interview-outcome labels, which do not exist publicly in English — the one
public person-job dataset with real labels is Chinese. PJB (§4.6) is the
evidence that a single global distribution is insufficient anyway.

---

## 6. Generation under grounding constraints

Once a system *writes* résumé lines, the failure mode changes from bad ranking to
invented fact.

- **FActScore** — Min, Krishna, Lyu et al., *FActScore: Fine-grained Atomic
  Evaluation of Factual Precision in Long Form Text Generation*
  (arXiv 2305.14251). **§3.1** defines the unit: "An atomic fact is a short
  sentence conveying one piece of information." Precision is the share of atomic
  facts supported by a trusted source; their automated estimator tracks human
  judgement with an error rate **under 2%**.
- **RAGAS** — Es, James, Espinosa-Anke and Schockaert, *Ragas: Automated
  Evaluation of Retrieval Augmented Generation* (arXiv 2309.15217). **§3**
  defines faithfulness as **F = |V| / |S|** — verified statements over total
  statements — on the principle that "the answer is faithful to the context if
  the claims that are made in the answer can be inferred from the context."
- **G-Eval** — Liu, Iter, Xu, Wang, Xu and Zhu, *G-Eval: NLG Evaluation using
  GPT-4 with Better Human Alignment* (arXiv 2303.16634). **§2** proposes
  probability-weighted scoring, `score = Σ p(sᵢ)·sᵢ`, because raw integer
  ratings collapse: "One digit usually dominates the distribution of the scores,
  such as 3 for a 1-5 scale", and "LLMs usually only output integer scores …
  This leads to many ties."
- **SummEval** — Fabbri, Kryściński, McCann, Xiong, Socher and Radev,
  *SummEval: Re-evaluating Summarization Evaluation* (arXiv 2007.12626). **§4.3**
  defines the four dimensions — coherence, consistency, fluency, relevance —
  over 12,800 summary-level human annotations.

---

## 7. What this project implements

Mapped to the sections above, with `file:line` so each can be checked.

### 7.1 Scoring

| Mechanism | Where | Relates to |
|---|---|---|
| Dense channel: bge-small-en-v1.5 via ONNX, CLS pooling, cosine **clamped** to [0,1] | `ats/dense.py:100-151` | §4.2 |
| Lexical channel: Okapi BM25, k1 1.5 / b 0.75, self-normalised by the JD scored against itself | `ats/lexical.py:48-114` | §4.1 |
| Skills channel: asymmetric weighted **coverage**, required 1.0 / preferred 0.5 | `ats/skills.py:49-81` | — |
| Convex fusion, dense .5 / lexical .2 / skills .3 | `ats/fusion.py:94-117` | §4.3 |
| A missing channel is `None`, not 0, and its weight is **redistributed** | `ats/fusion.py:53-66`, `:109-114` | §4.3 |
| IDF background corpus of 22 JDs across 22 families | `ats/corpus.py` | §4.1 |
| Percentile calibration over a 66-pair reference set, gated on channel-set match | `ats/calibrate.py:37-60` | §5 |
| Ordinal bands (Strong/Good/Partial/Limited) | `ats/fusion.py:33-42` | §5 |

Two details worth singling out. The cosine is **clamped** rather than rescaled:
`dense.py:145-151` records that mapping `(raw+1)/2` put an irrelevant document at
0.75 and a strong match at 0.91 — a spread of 0.16 — where clamping gives 0.49
against 0.83. And the lexical channel documents its own earlier failure: before
self-normalisation "a near-perfect resume scored 0.12 and the channel was
effectively dead weight in the fusion" (`lexical.py:110`).

### 7.2 Diagnosis and attribution

- **Keyword coverage, three-way**: *demonstrated* (in a bullet) / *mentioned*
  (only in skills or summary) / *missing* — `resume/diagnostics.py:61-77`. The
  middle class is the useful one: a skill listed but never shown.
- **Exact leave-one-out per bullet** — `resume/attribution.py:164-214`. Feasible
  because scoring makes no LLM call and all N+1 variants go through one batched
  `embed()`.
- **An approximation measured and rejected** — `attribution.py:24-28`:
  mean-pooling bullet embeddings is **4,253× faster** but scored Pearson 0.643
  and agreed on *sign* for only **14 of 20** bullets. Rejected because a
  sign error tells a user a line helps when it hurts.
- **Negative contributions surfaced** — `Attribution.harmful()` `:84-91`.

### 7.3 Generation, and the gates on it

- **Verification as code, not a prompt** — `nodes/verify.py`. Numbers must appear
  in evidence (versions and years masked first); unquantified impact claims and
  unsupported quotes are demoted or rejected. Rejected claims "become measurement
  tasks" rather than being deleted (`:20-21`).
- **Six rejection rules for a rewrite** — `nodes/variants.py:145-188`: empty,
  identical, introduces an absent number, names an absent technology, asserts an
  unsupported outcome, or fails to restate the original.
- **Additions held to the same standard**, with a duplicate re-offered as a
  rewrite rather than discarded — `nodes/additions.py:133-150`, `:207-255`.
- **Grounding may delete, never write** — `resume/grounding.py:9-12`, enforced by
  `is_subsequence_edit` `:293-309`, which asserts the edited text is a
  content-word subsequence of the original. An evidence-competence gate
  (`:78-122`) prevents absence of evidence being treated as evidence of absence.
- **The plan never states its own gain** — `nodes/plan.py:3-5`; the number comes
  from `scoring.max_gain`, the same arithmetic that produced the score.

### 7.4 Output and targeting

- **ATS-safe plain text**: no tables, no columns, no glyph bullets; tables
  flattened to one line per row — `resume/render/plain.py:22-64`.
- **Deterministic parsing**, no LLM — `resume/ingest.py:1-20`.
- **Role targeting** when no posting is available, with the synthesised text
  shown so it can be inspected — `v2/src/rba/resume/role_target.py`.
- **The 100-point rubric is no longer the score.** `resume/diagnostics.py:4-13`
  disowns it: those weights were invented, and "presenting a number built from
  invented weights as 'your ATS score' is the same category of error as
  inventing an achievement." It survives only as advice detectors and as the
  plan's gain budget.

### 7.5 What is measured

`v1/evals/judge.py` implements RAGAS-style faithfulness (§6) as
`supported / total` with binary per-claim verdicts, G-Eval over the four
SummEval dimensions, and a deterministic `number_precision` that needs no model.
Judge-bias mitigations are explicit (`judge.py:17-19`): dimensions scored in
separate calls, samples averaged, and the judge run on a different model tier
than the generator.

Latest recorded run (`v1/evals/results/latest.json`): **faithfulness 0.9184,
number_precision 1.0**, over **39 claims from 6 repositories** — 36 verified, 2
rejected, 1 needs-measure.

**This measures summary faithfulness, not match quality.** There is no
retrieval or ranking evaluation anywhere in the repository. See §9.

---

## 8. Gaps, and what to do about them

Ordered by expected value against cost.

### 8.1 Half the résumé is invisible to the heaviest channel

`dense.MAX_TOKENS = 512` (`ats/dense.py:32`) while a real one-page résumé is
**~1,150 tokens** (`v2/src/rba/ats/batch.py:11-12`). The whole document is
embedded as a single text, so everything past the cut contributes **nothing** to
the 0.5-weight dense channel. The code states this plainly in the API metadata
(`v2/src/rba_api/routes/meta.py:51-55`), and v2 currently exploits the
truncation as a *speed* optimisation — identical prefixes dedupe to one forward
pass — rather than fixing it.

**Proposal.** Embed per section or per entry and aggregate, rather than
embedding one truncated blob. Costs more forward passes; the existing batching
machinery (`v2/src/rba/ats/batch.py`, which exists only in v2) already makes
that affordable. This is the largest
correctness gap in the scorer and should be fixed before any new channel is
added.

### 8.2 The demonstrated/mentioned distinction never reaches the score

`diagnostics.keyword_coverage` already separates a skill shown in a project from
one merely listed, and tells the user about it — but `doc.as_text()`
(`resume/document.py:256`) flattens everything, so both channels weight a
skills-list token and a project-bullet token identically.

**Proposal.** Section-aware weighting in the lexical channel. APJFNN (§4.5) is
the precedent for treating a document as structured rather than flat, and the
distinction is already computed — it simply is not used.

### 8.3 No cross-encoder re-ranking

Pure bi-encoder cosine (`dense.py:138-151`). Nogueira and Cho (§4.4) report
36.5 against 16.7 MRR@10 for joint encoding.

**Proposal, with a caveat.** For ranking one résumé there is nothing to re-rank,
so the gain is not the MS MARCO gain. The real use is *within* the document —
scoring each bullet against the posting with a cross-encoder to drive the
improve-this-line feature more sharply than cosine can. Worth prototyping behind
the existing attribution interface; the cost is a second model.

### 8.4 No query-side expansion

BM25 tokenises the posting verbatim (`lexical.py:94`). The alias table
(`normalise.py:42-99`) canonicalises skills but never touches the lexical
channel — so a posting saying "containerisation" and a résumé saying "Docker"
match in the skills channel and miss entirely in the lexical one. TalentCLEF
§3.1 notes exactly this role for normalised skill names.

**Proposal.** Apply the alias table to the JD before BM25 tokenisation. Cheap,
deterministic, and reuses code that exists.

### 8.5 Weights are asserted, not fitted

`CHANNEL_WEIGHTS` is self-described as a judgement call. Bruch et al. support the
convex *form*, not the values.

**Proposal.** This cannot be fixed without labels, and labels do not exist
publicly in English (§4.5, §5). The honest options are: keep asserting and keep
labelling it as asserted (current behaviour, defensible), or run a small
preference study. Do not fit weights on synthetic data and present the result as
calibrated.

### 8.6 Nothing checks whether the user's actual file parses

`ingest.py` parses a PDF and `plain.py` emits safe text, but nothing warns that
*your uploaded* PDF uses two columns or a text box and will be shredded. Given
§3.1 — the layout step alone is worth ~4 F1 points — this is a real gap and the
one most directly aimed at the user's actual problem.

**Proposal.** A parseability report: extract with a naive single-column reader,
diff against the structured parse, and surface where they disagree.

### 8.7 Smaller items

- **Years of experience, seniority, disqualifiers.** `weighted_jaccard` is pure
  coverage (`skills.py:52-56`); "5+ years required" is not modelled at all.
- **Ordering effects.** Attribution measures removal, never reordering — there
  is no "move this up" lever.
- **Multilingual.** English throughout, by model and by alias table.
- **Taxonomy breadth.** ~50 canonical aliases; anything outside software gets
  essentially no canonicalisation, despite the JD corpus deliberately including
  nursing, teaching, accounting and marketing families.
- **Percentile resolution.** 66 breakpoints means ~1.5 points of granularity, so
  a genuine improvement can round to +0.0 (`routes/meta.py:56-60`).

---

## 9. Threats to validity

Stated plainly, because the rest of this document argues that unstated limits
are the problem.

1. **The calibration set is synthetic and small** — 66 pairs from three sentence
   templates across 22 job families (`v1/scripts/build_calibration.py`). It fixes
   the *scale*, not the ranking, and `calibration.json` says so in its own
   description field.
2. **The eval measures the wrong thing for this survey's purposes.** It measures
   whether generated repository summaries are faithful to evidence — which is
   the project's core safety property — but says nothing about whether the match
   score ranks well. **There is no retrieval or ranking evaluation in the
   repository at all.** Every claim in §7.1 is a claim about mechanism, not about
   accuracy.
3. **The 0.92 faithfulness figure is over 6 repositories**, chosen to span hard
   cases. It is a small sample and not a random one.
4. **No outcome labels exist**, so nothing here is validated against whether a
   résumé actually got an interview. PJB (§4.6) further implies any single global
   score misranks some industries.
5. **ATS behaviour is assumed, not observed.** No commercial ATS was tested. The
   ordinal bands are modelled on published Greenhouse and Workday vocabulary, and
   the code is explicit that it "does not claim to reproduce any real ATS"
   (`ats/__init__.py:26-29`).

---

## References

All verified by opening the source; section/table references point to where the
cited claim appears.

1. Bruch, S., Gai, S., Ingber, A. *An Analysis of Fusion Functions for Hybrid
   Retrieval.* ACM TOIS, 2023. arXiv:2210.11934, doi:10.1145/3596512. — §4.2,
   §5.1, Table 2.
2. Thakur, N., Reimers, N., Rücklé, A., Srivastava, A., Gurevych, I. *BEIR: A
   Heterogeneous Benchmark for Zero-shot Evaluation of IR Models.* 2021.
   arXiv:2104.08663. — §5, Table 2.
3. Robertson, S., Zaragoza, H. *The Probabilistic Relevance Framework: BM25 and
   Beyond.* Foundations and Trends in IR **3(4), pp. 333–389**, 2009.
4. Nogueira, R., Cho, K. *Passage Re-ranking with BERT.* 2019.
   arXiv:1901.04085. — Table 1.
5. Xiao, S., Liu, Z., Zhang, P., Muennighoff, N., Lian, D., Nie, J-Y. *C-Pack:
   Packed Resources for General Chinese Embeddings.* arXiv:2309.07597. — Table 5.
6. Zhu, C., Zhu, H., Xiong, H., Ma, C., Xie, F., Ding, P., Li, P. *Person-Job
   Fit: Adapting the Right Talent for the Right Job with Joint Representation
   Learning.* 2018. arXiv:1810.04040. — §3.2.
7. Qin, C., Zhu, H., Xu, T., et al. *Enhancing Person-Job Fit for Talent
   Recruitment: An Ability-aware Neural Network Approach.* SIGIR '18.
   arXiv:1812.08947. — §4.
8. Yu, X., Zhang, J., Yu, Z. *ConFit: Improving Resume-Job Matching using Data
   Augmentation and Contrastive Learning.* 2024. arXiv:2401.16349. — Tables 3–4.
9. Gasco, L., Fabregat, H., García-Sardiña, L., et al. *Overview of the
   TalentCLEF 2026: Skill and Job Title Intelligence for Human Capital
   Management.* arXiv:2606.31692. — §3.1, §4.1, Table 5.
10. Wang, G., Yang, X., Li, K., et al. *PJB: A Reasoning-Aware Benchmark for
    Person-Job Retrieval.* 2026. arXiv:2603.17386.
11. Zhu, F., Yu, J., Chen, Z., et al. *Layout-Aware Parsing Meets Efficient LLMs:
    A Unified, Scalable Framework for Resume Information Extraction and
    Evaluation.* 2025. arXiv:2510.09722. — Table 2.
12. Nguyen, K.C., Zhang, M., Montariol, S., Bosselut, A. *Rethinking Skill
    Extraction in the Job Market Domain using Large Language Models.* 2024.
    arXiv:2402.03832. — §5.2, Table 3.
13. Min, S., Krishna, K., Lyu, X., et al. *FActScore: Fine-grained Atomic
    Evaluation of Factual Precision in Long Form Text Generation.* 2023.
    arXiv:2305.14251. — §3.1.
14. Es, S., James, J., Espinosa-Anke, L., Schockaert, S. *Ragas: Automated
    Evaluation of Retrieval Augmented Generation.* 2023. arXiv:2309.15217. — §3.
15. Liu, Y., Iter, D., Xu, Y., Wang, S., Xu, R., Zhu, C. *G-Eval: NLG Evaluation
    using GPT-4 with Better Human Alignment.* 2023. arXiv:2303.16634. — §2.
16. Fabbri, A.R., Kryściński, W., McCann, B., Xiong, C., Socher, R., Radev, D.
    *SummEval: Re-evaluating Summarization Evaluation.* 2020.
    arXiv:2007.12626. — §4.3.
17. Fuller, J., Raman, M., Sage-Gavin, E., Hines, K. *Hidden Workers: Untapped
    Talent.* Harvard Business School / Accenture, September 2021.

### Cited in the codebase and checked here

All five arXiv identifiers appearing in docstrings were resolved and their
attached figures located: 2210.11934 (fusion), 2606.31692 (TalentCLEF MAP
0.6632 / 0.7140), 2510.09722 (~0.96 F1 zero-shot parsing), 2402.03832 (27.8 vs
62.6 strict-span F1), 2603.17386 (domain heterogeneity). The BEIR 51.68 / 43.81
pair in `ats/dense.py` is a leaderboard value rather than a paper figure — see
the caveat in §4.2 — and the unsourced extraction claim in `render/plain.py:5-6`
is discussed in §2.2.
