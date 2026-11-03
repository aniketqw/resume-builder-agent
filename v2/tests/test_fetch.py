from rba.config import Config
from rba.nodes.fetch import (
    _interesting_paths,
    extract_dependencies,
    fetch_repo_facts,
)
from rba.state import RepoRef


def test_extracts_dependencies_without_versions():
    deps = extract_dependencies({
        "requirements.txt": "fastapi==0.110.0\nhttpx>=0.27\n# comment\n",
        "package.json": '{"dependencies": {"react": "^18.2.0", "next": "14.1.0"}}',
        "go.mod": "module x\n\nrequire (\n\tgithub.com/gin-gonic/gin v1.9.1\n)\n",
    })
    assert "fastapi" in deps and "httpx" in deps
    assert "react" in deps
    assert "github.com/gin-gonic/gin" in deps
    assert not any("0.110" in d for d in deps), "versions are not claims"


def test_dependency_extraction_is_capped():
    big = "\n".join(f"pkg{i}==1.0" for i in range(200))
    assert len(extract_dependencies({"requirements.txt": big}, limit=40)) == 40


def test_interesting_paths_drop_vendored_bulk():
    tree = ([f"node_modules/p{i}/index.js" for i in range(500)]
            + ["src/app.py", "tests/test_app.py", "README.md"])
    kept = _interesting_paths(tree, 50)
    assert not any("node_modules" in p for p in kept)
    assert "src/app.py" in kept


def test_interesting_paths_prefer_source_over_assets():
    tree = ["assets/img/a.png", "assets/img/b.png", "src/main.go"]
    assert _interesting_paths(tree, 1) == ["src/main.go"]


class StubBackend:
    def __init__(self, tree, commits, files):
        self._tree, self._commits, self._files = tree, commits, files

    async def get_tree(self, repo, ref=""):
        return self._tree

    async def list_commits(self, repo, limit=30):
        return self._commits

    async def get_file(self, repo, path, ref=""):
        return self._files.get(path)

    async def get_languages(self, repo):
        return {"Python": 1000}


async def test_fetch_builds_a_complete_fact_bundle():
    backend = StubBackend(
        tree=["README.md", "src/app.py", "tests/test_app.py",
              ".github/workflows/ci.yml", "pyproject.toml"],
        commits=[{"sha": "abc1234567", "author": "me", "message": "init",
                  "date": "2026-10-01T00:00:00Z"},
                 {"sha": "def", "author": "other", "message": "fix", "date": ""}],
        files={"README.md": "# Demo\nDoes a thing.",
               "pyproject.toml": 'dependencies = ["httpx>=0.27"]'},
    )
    facts = await fetch_repo_facts(backend, RepoRef(name="demo"), "me", Config.load())
    assert facts.has_tests and facts.has_ci
    assert facts.readme.startswith("# Demo")
    assert facts.head_sha == "abc1234567"
    assert facts.commit_count == 2 and facts.author_commits == 1
    assert facts.author_share == 0.5
    assert "httpx" in facts.dependencies
    assert "pyproject.toml" in facts.key_files


async def test_fetch_survives_a_repo_with_nothing_in_it():
    facts = await fetch_repo_facts(StubBackend([], [], {}), RepoRef(name="empty"),
                                   "me", Config.load())
    assert facts.repo == "empty"
    assert facts.readme == "" and facts.commit_count == 0


async def test_readme_is_truncated_to_the_configured_budget():
    cfg = Config.load()
    limit = cfg.get("github.fetch.readme_max_chars")
    backend = StubBackend(["README.md"], [], {"README.md": "x" * (limit * 3)})
    facts = await fetch_repo_facts(backend, RepoRef(name="big"), "me", cfg)
    assert len(facts.readme) == limit, "prompt budget must be bounded"
