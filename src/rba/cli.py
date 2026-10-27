"""rba — command line entry point."""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess

import typer
from rich.console import Console
from rich.table import Table

from .config import REQUIRED_KEYS, Config, missing_secrets

app = typer.Typer(add_completion=False, help="Evidence-grounded resume builder.")
console = Console()


def _check(name: str, ok: bool, detail: str) -> tuple[str, bool, str]:
    return (name, ok, detail)


@app.command()
def doctor(
    network: bool = typer.Option(True, help="Also make live calls to each provider."),
) -> None:
    """Check credentials, tooling and provider reachability."""
    cfg = Config.load()
    rows: list[tuple[str, bool, str]] = []

    missing = missing_secrets()
    for key, what in REQUIRED_KEYS.items():
        present = key not in missing
        rows.append(_check(key, present, what if present else "MISSING — see .env.sample"))

    docker = shutil.which("docker")
    daemon_up = False
    if docker:
        daemon_up = subprocess.run(
            ["docker", "info"], check=False, capture_output=True, timeout=20
        ).returncode == 0
    rows.append(_check(
        "docker", bool(docker) and daemon_up,
        "daemon running" if daemon_up else
        ("installed, DAEMON NOT RUNNING — start Docker Desktop" if docker else "not installed"),
    ))
    rows.append(_check("gh", bool(shutil.which("gh")), "github cli"))

    if network:
        rows.extend(asyncio.run(_network_checks(cfg)))

    table = Table(title="rba doctor", show_lines=False)
    table.add_column("check")
    table.add_column("ok", justify="center")
    table.add_column("detail")
    for name, ok, detail in rows:
        table.add_row(name, "[green]yes[/]" if ok else "[red]no[/]", detail)
    console.print(table)

    blocking = [n for n, ok, _ in rows if not ok and n in REQUIRED_KEYS]
    if blocking:
        console.print(f"[red]Missing credentials: {', '.join(blocking)}[/]")
        raise typer.Exit(1)


async def _network_checks(cfg: Config) -> list[tuple[str, bool, str]]:
    import httpx

    from .config import secret

    out: list[tuple[str, bool, str]] = []
    async with httpx.AsyncClient(timeout=30.0) as c:
        # Groq: confirm the configured tier models actually exist today.
        try:
            r = await c.get(f"{cfg.get('groq.base_url')}/models",
                            headers={"Authorization": f"Bearer {secret('GROQ_API_KEY')}"})
            ids = {m["id"] for m in r.json().get("data", []) if m.get("active")}
            wanted = set(cfg.get("groq.tiers", {}).values())
            gone = wanted - ids
            out.append(_check("groq api", not gone and r.status_code == 200,
                              f"{len(ids)} active models" if not gone
                              else f"CONFIGURED MODEL GONE: {sorted(gone)}"))
        except Exception as exc:
            out.append(_check("groq api", False, str(exc)[:80]))

        # Jev: a trivially cheap decision call.
        try:
            r = await c.post(
                f"{cfg.get('jev.base_url')}/v1/systemone",
                headers={"Authorization": f"Bearer {secret('TYPESAFE_API_KEY')}"},
                json={"model": cfg.get("jev.model"), "state": {"ok": True},
                      "questions": {"up": {"type": "noul", "question": "Is this reachable?",
                                           "criteria": {"true": "yes", "false": "no"}}}},
            )
            out.append(_check("jev / typesafe", r.status_code == 200,
                              f"{r.json().get('model')}" if r.status_code == 200
                              else r.text[:80]))
        except Exception as exc:
            out.append(_check("jev / typesafe", False, str(exc)[:80]))

        # LangSmith.
        try:
            r = await c.get(f"{os.environ.get('LANGSMITH_ENDPOINT', 'https://api.smith.langchain.com')}"
                            "/api/v1/sessions?limit=1",
                            headers={"x-api-key": secret("LANGSMITH_API_KEY")})
            out.append(_check("langsmith", r.status_code == 200,
                              f"project={os.environ.get('LANGSMITH_PROJECT', '(unset)')}"))
        except Exception as exc:
            out.append(_check("langsmith", False, str(exc)[:80]))

        # GitHub token.
        try:
            r = await c.get("https://api.github.com/user",
                            headers={"Authorization": f"Bearer {secret('GITHUB_PERSONAL_ACCESS_TOKEN')}"})
            scopes = r.headers.get("x-oauth-scopes", "")
            out.append(_check("github token", r.status_code == 200,
                              f"{r.json().get('login')} scopes=[{scopes}]"
                              if r.status_code == 200 else r.text[:80]))
        except Exception as exc:
            out.append(_check("github token", False, str(exc)[:80]))
    return out


@app.command()
def run(
    username: str = typer.Argument(..., help="GitHub username to build from."),
    contact: str = typer.Option("", help="Contact line for the resume header."),
    review: bool = typer.Option(True, help="Pause for human review before the report."),
    dry_run: bool = typer.Option(False, "--dry-run",
                                 help="Estimate the token budget and spend nothing."),
    thread: str = typer.Option("default", help="Checkpoint thread id, for resuming."),
    mode: str = typer.Option("", help="Override github.mcp.mode: remote|docker|rest."),
) -> None:
    """Run the full graph: GitHub -> claims -> resumes -> plan -> report."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for noisy in ("httpx", "httpcore", "mcp", "fastmcp"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    cfg = Config.load()
    if mode:
        cfg.raw["github"]["mcp"]["mode"] = mode
    if dry_run:
        asyncio.run(_dry_run(cfg, username))
        return

    from .graph import Runner, build_graph, make_checkpointer

    runner = Runner(cfg, contact=contact, review=review)
    graph = build_graph(runner, make_checkpointer())
    cfg_arg = {"configurable": {"thread_id": thread}, "recursion_limit": 50}

    async def go():
        try:
            result = await graph.ainvoke({"username": username}, cfg_arg)
            if "__interrupt__" in result:
                payload = result["__interrupt__"][0].value
                console.print("\n[bold yellow]Paused for review[/]")
                console.print(payload)
                console.print(
                    f"\nResume with: [bold]rba resume --thread {thread}[/]")
                return result
            _print_summary(result)
            return result
        finally:
            await runner.aclose()

    asyncio.run(go())


@app.command()
def resume_run(
    thread: str = typer.Option("default", help="Checkpoint thread id to resume."),
    approve: bool = typer.Option(True, help="Approve as-is."),
    note: str = typer.Option("", help="Edits or instructions to record."),
) -> None:
    """Resume a run that paused at human review."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from langgraph.types import Command

    from .graph import Runner, build_graph, make_checkpointer

    cfg = Config.load()
    runner = Runner(cfg)
    graph = build_graph(runner, make_checkpointer())

    async def go():
        try:
            result = await graph.ainvoke(
                Command(resume={"approved": approve, "note": note}),
                {"configurable": {"thread_id": thread}, "recursion_limit": 50})
            _print_summary(result)
        finally:
            await runner.aclose()

    asyncio.run(go())


def _print_summary(result: dict) -> None:
    scores = result.get("scores") or {}
    claims = result.get("claims") or []
    table = Table(title="ATS scores")
    table.add_column("role")
    table.add_column("score", justify="right")
    for k, v in sorted(scores.items(), key=lambda kv: -kv[1].total):
        table.add_row(k, f"{v.total:.1f}")
    console.print(table)
    verified = sum(1 for c in claims if c.status.value == "verified")
    needs = sum(1 for c in claims if c.status.value == "needs_measure")
    rejected = sum(1 for c in claims if c.status.value == "rejected")
    console.print(f"claims: [green]{verified} verified[/], "
                  f"[yellow]{needs} need measurement[/], [red]{rejected} rejected[/]")
    for finding in (result.get("safety") or []):
        console.print(f"[red]security[/] {finding.repo}/{finding.path}: {finding.kind}")
    for label, path in (result.get("report_files") or {}).items():
        console.print(f"{label}: {path}")
    usage = result.get("usage") or {}
    if usage.get("groq"):
        console.print(f"groq tokens: {usage['groq'].get('total', 0)}")
    if usage.get("jev"):
        console.print(f"jev: ${usage['jev'].get('cost_usd', 0):.5f}")


async def _dry_run(cfg: Config, username: str) -> None:
    """Project the token spend without making a single Groq call."""
    from .github_tools import make_backend
    from .nodes.select import select_repos

    backend = await make_backend(cfg)
    try:
        repos = await backend.list_repos(username)
        state = await select_repos({"repos": repos, "username": username},
                                   config=cfg, backend=backend)
    finally:
        await backend.aclose()

    n = len(state["selected"])
    # 2,074 tokens was the measured cost of one real evidence-linked summary.
    per_summary = 2100
    roles = len(cfg.get("report.roles") or [])
    plan_calls = 2
    projected = n * per_summary + roles * 2000 + plan_calls * 2500
    tpm = cfg.limits_for(cfg.model_for_tier("small")).tpm

    table = Table(title=f"dry run — {username}")
    for col in ("step", "calls", "est. tokens"):
        table.add_column(col, justify="right" if col != "step" else "left")
    table.add_row("summarize_repo", str(n), f"{n * per_summary:,}")
    table.add_row("build_resumes", str(roles), f"{roles * 2000:,}")
    table.add_row("plan_improvements", str(plan_calls), f"{plan_calls * 2500:,}")
    table.add_row("[bold]total", f"[bold]{n + roles + plan_calls}",
                  f"[bold]{projected:,}")
    console.print(table)
    console.print(f"At {tpm:,} tokens/min that is roughly "
                  f"[bold]{projected / tpm:.1f} minutes[/] of rate-limited time.")
    console.print(f"Jev already scored {len(repos)} repos for "
                  f"${state['usage']['jev']['cost_usd']:.5f} (not Groq tokens).")


@app.command()
def config() -> None:
    """Show resolved model routing and limits."""
    cfg = Config.load()
    table = Table(title="model routing")
    for col in ("node", "tier", "model"):
        table.add_column(col)
    table.add_column("tpm", justify="right")
    table.add_column("rpd", justify="right")
    for node, tier in (cfg.get("routing") or {}).items():
        model = cfg.model_for_tier(tier)
        lim = cfg.limits_for(model)
        table.add_row(node, tier, model, str(lim.tpm), str(lim.rpd))
    console.print(table)


if __name__ == "__main__":
    app()
