---
title: Resume Builder Agent
emoji: 📄
colorFrom: green
colorTo: gray
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: Evidence-grounded resumes built from your GitHub repos
---

# Resume Builder Agent

Reads a GitHub account, turns each repository into evidence-backed claims, and
builds role-targeted resumes plus an improvement plan where every item cites a
real file or commit.

The governing rule: **it never invents an achievement or a number.** A claim
that cannot be traced to repository evidence is not softened — it is dropped and
replaced with a `[MEASURE: <command>]` task.

This Space publishes a real run: the generated report, the full run trace
(including every claim the verifier refused), and how the thing works. It is a
static page rather than a live app, because Gradio and Docker Spaces require a
paid plan on this account.

Source, tests and the agent itself:
**https://github.com/aniketqw/resume-builder-agent**

Built with LangGraph + Groq, the official GitHub MCP server (read-only),
TypeSafe/Jev for routing decisions, and LangSmith for tracing.
