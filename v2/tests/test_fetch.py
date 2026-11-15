from rba.config import Config
from rba.nodes.fetch import (
    SOURCE_FILE_LIMIT,
    SOURCE_FILE_MAX_CHARS,
    _interesting_paths,
    _ranked_source_paths,
    extract_dependencies,
    fetch_repo_facts,
)
from rba.repobrain import index
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


async def test_a_long_source_file_still_parses_after_fetch():
    """Live bug: `fetch_repo_facts` used to truncate every source file to
    4000 characters before `repobrain.index()` ever saw it, so a file
    longer than that got cut off mid-token -- a guaranteed `SyntaxError`
    that had nothing to do with the code's real quality. This builds a file
    long enough to have been cut mid-string-literal at the old 4000-char
    cutoff and confirms it survives indexing whole."""
    padding = "\n".join(f"# comment line {i}" for i in range(400))  # ~6,800 chars
    assert len(padding) > 4000, "fixture must actually exceed the old cutoff"
    source = (
        f'{padding}\n\n'
        'def build_thing():\n'
        '    message = "this closing quote used to land past the old cutoff"\n'
        '    return message\n'
    )
    backend = StubBackend(
        tree=["src/app.py"], commits=[], files={"src/app.py": source})
    facts = await fetch_repo_facts(backend, RepoRef(name="demo"), "me", Config.load())

    assert facts.source_files["src/app.py"] == source, \
        "the full file must be kept for AST indexing, not cut to the old 4000-char cap"
    brain = index(facts.source_files)
    assert brain.partial_source == [], f"should parse cleanly: {brain.partial_source}"
    assert brain.symbol("build_thing"), "the function past the old cutoff must be indexed"


async def test_a_pathological_file_is_still_bounded():
    """The new cap is generous (see `fetch.py`'s docstring for why), not
    unbounded -- a single huge generated/vendored file must not blow up
    `AgentState` without limit."""
    backend = StubBackend(
        tree=["src/huge.py"], commits=[],
        files={"src/huge.py": "x = 1\n" * 100_000})
    facts = await fetch_repo_facts(backend, RepoRef(name="demo"), "me", Config.load())
    assert len(facts.source_files["src/huge.py"]) == SOURCE_FILE_MAX_CHARS


def test_ranked_source_paths_are_python_only_and_bounded():
    tree = (["src/a.py", "src/b.py", "tests/test_a.py", "src/img.png",
             "README.md", "src/lib.go"]
            + [f"src/extra{i}.py" for i in range(20)])
    ranked = _ranked_source_paths(tree, SOURCE_FILE_LIMIT)
    assert len(ranked) == SOURCE_FILE_LIMIT
    assert all(p.endswith(".py") for p in ranked)
    # Non-Python source and non-source files never make the cut.
    assert "src/lib.go" not in ranked and "README.md" not in ranked
    # Non-test files are preferred over test files when both compete for
    # the same slot (the bound here is deliberately tight to force that).
    assert "src/a.py" in ranked and "src/b.py" in ranked


class RefRecordingBackend(StubBackend):
    """Records the ref every get_file call was made with, so the pinned-SHA
    fix in fetch_repo_facts can be verified directly rather than trusted."""

    def __init__(self, tree, commits, files):
        super().__init__(tree, commits, files)
        self.refs_seen: list[str] = []

    async def get_file(self, repo, path, ref=""):
        self.refs_seen.append(ref)
        return self._files.get(path)


async def test_readme_and_key_files_are_pinned_to_head_sha_not_the_branch():
    """Before this, README and key-file fetches used repo.default_branch --
    so if the branch moved between listing the tree and reading the file,
    the content could silently desync from the commit a permalink claims it
    came from. Every file read must use the exact commit this run saw."""
    backend = RefRecordingBackend(
        tree=["README.md", "pyproject.toml", "src/app.py"],
        commits=[{"sha": "pinned-sha", "author": "me", "message": "x", "date": ""}],
        files={"README.md": "# x", "pyproject.toml": "x = 1",
               "src/app.py": "def f():\n    pass\n"},
    )
    repo = RepoRef(name="demo", default_branch="main")
    facts = await fetch_repo_facts(backend, repo, "me", Config.load())
    assert facts.head_sha == "pinned-sha"
    assert backend.refs_seen and all(r == "pinned-sha" for r in backend.refs_seen)
    assert facts.source_files.get("src/app.py") == "def f():\n    pass\n"
