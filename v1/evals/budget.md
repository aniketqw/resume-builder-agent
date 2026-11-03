# Measured token budget

Run: `python scripts/measure_budget.py --repos 10 --cold`
Raw numbers in `evals/budget.json`. Everything below is measured unless labelled
otherwise.

## What one repo actually costs

10 repos, cold (local response cache bypassed), 352 s wall clock:

| | tokens |
|---|---|
| prompt | 24,054 |
| completion | 6,533 |
| **total** | **30,587** |
| requests | 10 |
| served from Groq's prefix cache | 768 (3.2% of prompt) |

Per repo: **1 request, 3,058 tokens** — 631 stable system prompt, ~1,774
variable evidence, 653 completion. Prompt is **79%** of the cost, and the
evidence bundle is 74% of the prompt.

## Three claims in the plan that the measurement overturns

**1. Requests were never the binding constraint.** The plan treated
"770 → 110 requests" as the headline fix. Measured: 110 repos at one call each
is 110 requests — **11% of the 1,000/day limit**. Even the naive 7-calls-per-repo
shape is 770, which fits. Tokens bind; requests do not.

**2. The prompt is already prefix-structured.** Step 3 called for restructuring
so the stable instructions come first and repo facts last. `summarize.v2.md`
already does exactly that — stable `# system`, variable evidence in `# user`.
There is nothing to restructure.

**3. Groq prefix caching does not fire for us.** This is the significant one.
Controlled test, identical system prefix, calls seconds apart:

| stable prefix | cached_tokens on the repeat |
|---|---|
| 771 tokens | 0 |
| 1,461 tokens | 0 |
| 2,841 tokens | 0 |
| the real 631-token `summarize.v2` system prompt, 5 calls back to back | 0, 0, 0, 0, 0 |

So the plan's third pillar — "cached tokens are exempt from rate limits" — has
no measured support on this account. The 768 cached tokens in the real run are
3.2% and unexplained; they are not a strategy. **The token budget has to close
without prefix caching.**

Separately: there is **no daily-token header**. Groq returns only
`x-ratelimit-limit-requests: 1000` and `x-ratelimit-limit-tokens: 8000`. Both
decode as continuous-refill buckets (12 requests used → 17m16s to refill is
exactly 1,000/day; 1,946 tokens used → 14.6 s is exactly 8,000/min). The
200,000/day cap is **documented but not observable**, so every percentage below
is against a number we cannot verify.

## What stacking is actually worth

Projected from the measured per-repo figures (completion assumed ~4× for the
larger stacked output — the one estimate here):

| repos | 7 separate calls | 1 stacked call | saving |
|---|---|---|---|
| 15 | 321,090 | 81,255 | 75% |
| 25 | 535,150 | 135,425 | 75% |
| 58 | 1,241,548 | 314,186 | 75% |
| 110 | 2,354,660 | 595,870 | 75% |

Stacking still earns its place — but because it stops re-sending the 1,774-token
evidence bundle seven times, not because of request count.

## The conclusion that matters

**110 repos was a straw man I invented.** A resume carries on the order of ten
projects. Selection already filters to repos worth describing, and only those
reach an LLM.

| | requests | tokens | % of documented TPD | wall clock at 8K/min |
|---|---|---|---|---|
| 15 repos, stacked | 15 | 81,255 | 40.6% | 10.2 min |
| 25 repos, stacked | 25 | 135,425 | 67.7% | 16.9 min |
| 58 repos, stacked | 58 | 314,186 | 157% | 39.3 min |

**15–25 repos fits comfortably** with no caching tricks at all. Above ~35 it
stops fitting the documented daily cap, and past ~20 minutes the Streamlit
socket is the real limit regardless of quota.

## What still holds

- The local SQLite response cache works: the warm re-run of the same 10 repos
  cost **0 requests and 0 tokens** in 108 s. Verification item 3 passes.
- The 8,000 TPM ceiling is real and is what produced the four 429 backoffs
  during the cold run.
