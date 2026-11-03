from rba.signals import RepoSignals, classify_tree, fetch_signals
from rba.state import RepoRef


def sig_for(paths):
    s = RepoSignals(repo="demo")
    classify_tree(paths, s)
    return s


def test_detects_python_and_js_test_layouts():
    assert sig_for(["tests/test_api.py"]).has_tests
    assert sig_for(["src/foo_test.go"]).has_tests
    assert sig_for(["app/__tests__/x.js"]).has_tests
    assert sig_for(["src/button.test.tsx"]).has_tests
    assert not sig_for(["src/latest.py", "contest/a.py"]).has_tests


def test_detects_ci_configs():
    assert sig_for([".github/workflows/ci.yml"]).has_ci
    assert sig_for([".gitlab-ci.yml"]).has_ci
    assert not sig_for(["workflows/ci.yml"]).has_ci


def test_separates_code_from_notebooks_and_archives():
    s = sig_for(["a.py", "b.ts", "notes.ipynb", "slides.pptx", "data.csv", "lec.pdf"])
    assert s.code_files == 2
    assert s.notebook_files == 1
    assert s.archive_files == 3


def test_course_archive_is_distinguishable_from_a_project():
    """The signal that stops lecture dumps being sold as engineering work."""
    archive = sig_for([f"week{i}/lecture.pdf" for i in range(20)]
                      + [f"week{i}/notes.ipynb" for i in range(20)])
    project = sig_for(["src/app.py", "src/db.py", "tests/test_app.py",
                       "pyproject.toml", ".github/workflows/ci.yml", "README.md"])
    assert archive.code_files == 0 and archive.archive_files == 20
    assert project.code_files == 3 and project.has_tests and project.has_ci
    assert project.dep_files == ["pyproject.toml"]


def test_top_dirs_ranked_by_file_count():
    s = sig_for(["src/a.py", "src/b.py", "src/c.py", "docs/x.md"])
    assert s.top_dirs[0] == "src"


def test_dependency_manifests_collected_once():
    s = sig_for(["package.json", "frontend/package.json", "go.mod"])
    assert sorted(s.dep_files) == ["go.mod", "package.json"]


def test_as_state_is_evidence_only():
    """Jev must receive facts, never our own verdict about the repo."""
    s = sig_for(["src/a.py", "README.md"])
    s.readme_chars = 500
    state = s.as_state(RepoRef(name="demo", language="Python", size_kb=100))
    assert state["source_code_files"] == 1
    assert state["readme_present"] is True
    # No pre-baked judgement leaks into the prompt.
    forbidden = {"claimable", "score", "good", "quality", "tier", "recommended"}
    assert not forbidden & set(state)


class StubBackend:
    def __init__(self, tree, commits, readme=None, fail=None):
        self._tree, self._commits, self._readme, self._fail = tree, commits, readme, fail

    async def get_tree(self, repo, ref=""):
        if self._fail == "tree":
            raise RuntimeError("boom")
        return self._tree

    async def list_commits(self, repo, limit=30):
        return self._commits

    async def get_file(self, repo, path, ref=""):
        return self._readme


async def test_fetch_signals_computes_author_share():
    backend = StubBackend(
        ["README.md", "src/a.py", "tests/test_a.py"],
        [{"author": "aniketqw", "date": "2026-10-26T00:00:00Z"},
         {"author": "aniketqw", "date": "2026-10-20T00:00:00Z"},
         {"author": "someone-else", "date": "2026-10-01T00:00:00Z"}],
        readme="# demo\nA real project.",
    )
    s = await fetch_signals(backend, RepoRef(name="demo"), owner="aniketqw")
    assert s.commit_count == 3
    assert s.author_commits == 2
    assert s.has_tests
    assert s.readme_path == "README.md"
    assert s.readme_chars == len("# demo\nA real project.")


async def test_fetch_signals_survives_a_broken_repo():
    """One unreachable repo must not abort a 93-repo sweep."""
    backend = StubBackend([], [], fail="tree")
    s = await fetch_signals(backend, RepoRef(name="broken"), owner="x")
    assert s.file_count == 0
    assert s.repo == "broken"
