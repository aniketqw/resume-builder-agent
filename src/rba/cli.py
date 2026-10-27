"""rba — command line entry point."""
from __future__ import annotations

import asyncio
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
