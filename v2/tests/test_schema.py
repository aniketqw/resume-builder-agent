"""Groq strict mode fails the request unless every object is fully specified."""

from pydantic import BaseModel, Field

from rba.schema import lint_strict_schema, response_format, to_strict_schema


class Evidence(BaseModel):
    path: str
    quote: str


class Claim(BaseModel):
    text: str
    evidence: list[Evidence]
    confidence: float | None = None


class RepoSummary(BaseModel):
    one_line: str
    tech: list[str]
    claims: list[Claim]
    has_tests: bool = False


def test_every_object_is_strict_compliant():
    assert lint_strict_schema(to_strict_schema(RepoSummary)) == []


def test_optional_fields_are_forced_into_required():
    """Pydantic omits optionals from `required`; Groq strict mode rejects that."""
    schema = to_strict_schema(RepoSummary)
    claim = schema["properties"]["claims"]["items"]
    assert set(claim["required"]) == {"text", "evidence", "confidence"}


def test_defaults_do_not_exempt_a_field():
    schema = to_strict_schema(RepoSummary)
    assert "has_tests" in schema["required"]


def test_nested_refs_are_inlined():
    schema = to_strict_schema(RepoSummary)
    assert "$defs" not in schema
    evidence = schema["properties"]["claims"]["items"]["properties"]["evidence"]["items"]
    assert evidence["properties"]["quote"]["type"] == "string"


def test_response_format_shape_matches_groq_api():
    rf = response_format(RepoSummary, name="repo_summary")
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["name"] == "repo_summary"
    assert rf["json_schema"]["strict"] is True


class StepLike(BaseModel):
    """Fields deliberately named after JSON-Schema annotations."""
    title: str
    default: str = "x"
    examples: list[str] = Field(default_factory=list)
    hours: float = 0.0


def test_fields_named_after_schema_annotations_survive():
    """Regression: `title`, `default` and `examples` were stripped as noisy
    annotations even when they were the model's own fields. The plan schema
    lost its `title` field, and every generated step then failed validation."""
    schema = to_strict_schema(StepLike)
    assert set(schema["properties"]) == {"title", "default", "examples", "hours"}
    assert set(schema["required"]) == {"title", "default", "examples", "hours"}
    assert lint_strict_schema(schema) == []


def test_annotations_are_still_stripped_from_schema_nodes():
    schema = to_strict_schema(StepLike)
    # The *annotation* on the whole object is gone, the *field* is not.
    assert "title" in schema["properties"]
    assert schema["properties"]["hours"].get("title") is None


def test_linter_catches_a_bad_schema():
    bad = {"type": "object", "properties": {"a": {"type": "string"}}, "required": []}
    problems = lint_strict_schema(bad)
    assert any("additionalProperties" in p for p in problems)
    assert any("required" in p for p in problems)


def test_linter_catches_unresolved_ref():
    assert lint_strict_schema({"$ref": "#/$defs/Thing"})
