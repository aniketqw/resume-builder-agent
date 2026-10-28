# Run trace

A complete record of one pipeline run, written so the output can be **audited rather than trusted**. Every claim the verifier rejected is listed below with its reason — that section is the most informative part of this file.

- **Command:** `python scripts/build_sample.py --username aniketqw`
- **Account:** `aniketqw` (public repositories only)
- **Python:** 3.11.15 on Darwin
- **Models:** small/mid `openai/gpt-oss-20b`, big `openai/gpt-oss-120b`
- **Free-tier ceiling:** 8,000 tokens/min, 1,000 requests/day (per model)

## Stages

| Stage | What happened | Tokens | Seconds |
|---|---|---:|---:|
| list_repos | 99 repositories found via the GitHub MCP server; 57 public. Private repositories are filtered out here, before anything is fetched. | — | 7.7 |
| select_repos | Jev scored 47 repositories for $0.00161; kept 21. No Groq tokens spent. | — | 37.8 |
| fetch_facts | Deterministic fetch for 21 repositories (19 with a README, 9 with tests). No model involved. | — | 25.3 |
| safety_scan | 1 finding(s) in the scanned repositories. | — | 0.0 |
| summarize_repo | 21 repositories summarized, 111 raw claims proposed. | — | 0.0 |
| verify_claims | 102/111 claims survived. Pure Python: no model, no tokens. | — | 0.0 |
| build_resumes | 5 resumes written from verified claims only. | 9665 | 62.0 |
| plan_improvements | 10 evidence-linked steps. | 3823 | 2390.0 |

## Headline numbers

- **102/111 claims verified** (92%); 3 need measurement; 6 rejected outright.
- **5 resumes** built from verified claims only.
- **Groq:** 13,488 tokens across 3 node(s).
- **Cache hit rate:** 100% — an unchanged repo costs nothing on a re-run.
- **Wall clock:** 2523s, most of it waiting on the 8,000 tokens/minute limit.

## Repository selection

Scored by Jev (a typed decision model), not by Groq — this costs fractions of a cent and leaves the scarce Groq budget for text.

| Repo | Claimable | Tier | Language |
|---|---:|---|---|
| `AiAgentDemo` | 0.69 | mid | Python |
| `file-upload-server` | 0.67 | mid | HTML |
| `locallend` | 0.67 | mid | Java |
| `AOA_Controller_ty` | 0.66 | mid | C |
| `GithubRepoAnalyzer` | 0.65 | mid | JavaScript |
| `next-express-ejs-crud` | 0.65 | mid | JavaScript |
| `spring-gemini-chat` | 0.64 | mid | HTML |
| `nextjs-express-crud-hf` | 0.61 | mid | JavaScript |
| `AOA_Controller_ty_1` | 0.59 | mid | C |
| `chronosModelImprovement` | 0.58 | mid | Python |
| `pehenava-backend` | 0.58 | mid | TypeScript |
| `websocket_mcqs1` | 0.58 | small | JavaScript |
| `AiTerminal` | 0.56 | small | Go |
| `tender_search_app` | 0.56 | small | Python |
| `aniketqw.github.io` | 0.55 | mid | HTML |
| `Rule-BasedCustomKafkaPartitioner` | 0.54 | small | Java |
| `sandbox-agent` | 0.51 | mid | Python |
| `Booking-System` | 0.50 | mid | Python |
| `TaskAssignWithDjango` | 0.49 | big | Python |
| `websocket_chatroom` | 0.48 | big | HTML |
| `chat-starter-code` | 0.47 | small | HTML |

## Claim verdicts

Verification is **plain Python**, not a prompt. Every number in a claim must appear in that repository's evidence bundle; every quote must be present *and* relevant to the claim it supports.

### Rejected (6)

These were proposed by the model and thrown out. They never reach a resume.

- `locallend` — The backend is implemented in Java using Spring Boot.
  - *every supporting quote is absent from the evidence*
- `locallend` — The frontend is written in TypeScript with React.
  - *every supporting quote is absent from the evidence*
- `locallend` — The repository includes image upload functionality.
  - *every supporting quote is absent from the evidence*
- `locallend` — The repository implements CRUD operations for items.
  - *every supporting quote is absent from the evidence*
- `AiTerminal` — The project uses the tcell library for terminal UI.
  - *supporting quote is real but does not mention anything in the claim*
- `AiTerminal` — The repository does not include a CI configuration.
  - *supporting quote is real but does not mention anything in the claim*

### Needs measurement (3)

Plausible but unproven. The agent refused to attach a number it could not find, and emitted a measurement task instead.

- `chronosModelImprovement` — The distance‑aware model does not win the majority of series, losing 745 out of 1484.
  - *contains number(s) absent from the evidence: 745* → **`measure this and record the command and result`**
- `tender_search_app` — Enables users to add and refine keywords to improve search accuracy.
  - *asserts an effect without a measured quantity* → **`run the evaluation set and record the metric`**
- `sandbox-agent` — The agent graph is lazily loaded to improve startup performance.
  - *asserts an effect without a measured quantity* → **`measure this and record the command and result`**

## Resume scores

Deterministic 100-point ATS rubric, no model involved.

| Role | Score | Keywords | Role fit | Numbers |
|---|---:|---:|---:|---:|
| balanced | **59.1** | 32.0 | 14.31 | 0.0 |
| backend_platform_engineer | **43.02** | 19.4 | 10.97 | 0.0 |
| ai_llm_agent_engineer | **35.75** | 13.25 | 9.8 | 0.0 |
| data_scientist | **20.15** | 2.56 | 4.9 | 0.0 |
| ml_engineer | **15.17** | 0.98 | 1.89 | 0.0 |

### Bullets dropped during resume writing

None. Every generated bullet traced back to a verified claim.

## Improvement plan

10 steps, ~67 hours, +34.5 ATS points. The gain figure comes from the same rubric that produced the score, not from the model.

| Step | Repo | Hours | ATS gain |
|---|---|---:|---:|
| Add a PyTorch image classification model to AiAgentDemo | `AiAgentDemo` | 12 | +5.94 |
| Document a computer‑vision pipeline in AiAgentDemo README | `AiAgentDemo` | 3 | +5.78 |
| Add a Dockerfile and CI pipeline for reproducible ML builds | `AiAgentDemo` | 5 | +3.86 |
| Create a benchmark suite for model evaluation in AiAgentDemo | `AiAgentDemo` | 6 | +3.86 |
| Add CI/CD Automation for Tests and Docker Builds | `new work` | 10 | +3.53 |
| Integrate TensorFlow inference endpoint into AiAgentDemo | `AiAgentDemo` | 8 | +3.05 |
| Add System Architecture Diagram and Docs to AiAgentDemo | `AiAgentDemo` | 6 | +2.82 |
| Document LLM Integration and Prompt Engineering | `AiAgentDemo` | 5 | +2.12 |
| Implement Performance Benchmarks for Tool Calls | `AiAgentDemo` | 8 | +2.12 |
| Introduce Git Contribution Guidelines and Changelog | `new work` | 4 | +1.41 |

## Security findings in the scanned repositories

- **[high]** `pehenava-backend/pehnava-ai-frontend/.env.local` — An .env file is committed. These normally hold live credentials.

## Measured quality

| Metric | Result |
|---|---:|
| claim faithfulness (text that reaches a resume) | n/a |
| number precision | 1.0 |
| whole-summary faithfulness (includes unverified prose) | 0.9184 |
| repositories judged | 6 |

## Notes

- Private repositories are excluded before any fetch, so no private content enters this pipeline at all.
- Summaries are cached by (repo, head SHA, prompt version, model); the cache hit rate above shows how much of this run was free.

## How to reproduce

```bash
pip install -e ".[dev]"
cp .env.sample .env   # add GROQ_API_KEY, TYPESAFE_API_KEY, GITHUB_PERSONAL_ACCESS_TOKEN
rba doctor
python scripts/build_sample.py --username aniketqw
```

Summaries are cached by `(repo, head SHA, prompt version, model)`, so a second run over unchanged repositories spends no tokens at all.

