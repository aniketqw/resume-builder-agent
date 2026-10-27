from rba.nodes.safety import _redact, safety_scan, scan_content, scan_paths
from rba.state import RepoFacts


def kinds(findings):
    return {f.kind for f in findings}


# -- path rules -----------------------------------------------------------
def test_committed_env_and_token_files_are_flagged():
    f = scan_paths("demo", [".env", "config/token.txt", "cookies.json",
                            "deploy/id_rsa", "certs/server.pem"])
    assert kinds(f) == {"env-file", "credential-file", "cookie-file", "private-key"}
    assert all(x.severity == "high" for x in f)


def test_env_examples_are_not_flagged():
    """.env.sample is the file we ask people to commit."""
    assert scan_paths("demo", [".env.sample", ".env.example", ".env.template"]) == []


def test_ordinary_files_are_not_flagged():
    assert scan_paths("demo", ["src/app.py", "README.md", "tests/test_x.py",
                               "docs/secrets-management.md"]) == []


def test_browser_profile_is_flagged():
    f = scan_paths("demo", ["profile/Default/Cookies"])
    assert kinds(f) == {"browser-profile"}


# -- content rules --------------------------------------------------------
def test_detects_real_credential_shapes():
    cases = {
        "groq-key": "GROQ_API_KEY=gsk_" + "a" * 48,
        "langsmith-key": "key: lsv2_pt_" + "b" * 32 + "_" + "c" * 10,
        "github-token": "token ghp_REDACTED_TEST_FIXTURE",
        "aws-access-key": "AKIA2B7QX9ZLMNPQRSTU",
        "anthropic-key": "sk-ant-" + "e" * 40,
        "private-key": "-----BEGIN RSA PRIVATE KEY-----",
    }
    for kind, text in cases.items():
        assert kinds(scan_content("r", "f", text)) == {kind}, kind


def test_aws_documentation_key_is_treated_as_a_placeholder():
    """AKIAIOSFODNN7EXAMPLE is AWS's own docs sample; flagging it is noise."""
    assert scan_content("r", "f", "AKIAIOSFODNN7EXAMPLE") == []


def test_anthropic_key_is_not_misreported_as_openai():
    """`sk-ant-...` also matches the broader `sk-...` shape; the specific wins."""
    f = scan_content("r", "f", "sk-ant-" + "e" * 40)
    assert kinds(f) == {"anthropic-key"}


def test_placeholders_are_ignored():
    for text in ['API_KEY="your-api-key-here"', 'password = "changeme123456"',
                 'token: "<YOUR_TOKEN_HERE>"', 'secret="xxxxxxxxxxxxxxxx"',
                 'api_key = "0000000000000000"']:
        assert scan_content("r", "f", text) == [], text


def test_prose_about_secrets_is_not_a_finding():
    text = ("Store your API key in the environment. Never commit a password "
            "or token to the repository.")
    assert scan_content("r", "README.md", text) == []


def test_repeated_character_runs_read_as_placeholders():
    """`ghp_xxxxxxxx...` is what people write in docs, not a real token."""
    assert scan_content("r", "f", "ghp_" + "x" * 36) == []


def test_finding_records_line_number_and_never_the_secret():
    key = "gsk_" + "z" * 48
    text = f"line one\nline two\nGROQ_API_KEY={key}\n"
    f = scan_content("r", "cfg.py", text)[0]
    assert f.path == "cfg.py:3"
    assert key not in f.detail, "the secret value must never be echoed back"
    assert "gsk_zz" in f.detail, "but enough shape to find it"


def test_redaction_keeps_it_unusable():
    key = "gsk_" + "q" * 48
    r = _redact(key)
    assert key not in r and len(r) < len(key)


def test_one_finding_per_kind_per_file():
    text = "\n".join(f"K{i}=gsk_{'a' * 48}" for i in range(5))
    assert len(scan_content("r", "f", text)) == 1


# -- node -----------------------------------------------------------------
async def test_safety_scan_covers_tree_readme_and_key_files():
    facts = RepoFacts(
        repo="leaky",
        tree=["src/app.py", ".env"],
        readme="Set GITHUB_TOKEN=ghp_REDACTED_TEST_FIXTURE",
        readme_path="README.md",
        key_files={"docker-compose.yml": "AWS_KEY: AKIA2B7QX9ZLMNPQRSTU"},
    )
    out = await safety_scan({"facts": {"leaky": facts}})
    assert kinds(out["safety"]) == {"env-file", "github-token", "aws-access-key"}
    assert {f.repo for f in out["safety"]} == {"leaky"}


async def test_clean_repo_produces_no_findings():
    facts = RepoFacts(repo="clean", tree=["src/app.py", ".env.sample"],
                      readme="# Clean project", readme_path="README.md")
    out = await safety_scan({"facts": {"clean": facts}})
    assert out["safety"] == []
