from rba.nodes.prepare_evidence import prepare_evidence
from rba.state import RepoFacts


async def test_builds_a_repo_graph_per_repo_and_surfaces_gaps():
    facts = {
        "a/good": RepoFacts(repo="a/good", source_files={
            "src/app.py": "def f():\n    return g()\n\ndef g():\n    return 1\n",
            "README.md": "# hi",
        }),
        "a/empty": RepoFacts(repo="a/empty", source_files={}),
    }
    out = await prepare_evidence({"facts": facts})
    graph = out["repo_graph"]
    assert set(graph) == {"a/good", "a/empty"}
    assert "src/app.py" in graph["a/good"].files
    assert graph["a/good"].callers("g")

    gaps = out["partial_source"]
    assert any(g.startswith("a/good:README.md") for g in gaps)
    assert all(not g.startswith("a/empty:") for g in gaps)  # nothing to index, nothing to flag
