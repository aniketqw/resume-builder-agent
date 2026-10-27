"""Strict JSON-schema helpers for Groq structured outputs.

Groq's strict mode has two hard requirements that are easy to violate and that
fail at request time rather than at author time:

  * every property must appear in ``required``
  * every object must set ``additionalProperties: false``

Pydantic's ``model_json_schema()`` satisfies neither by default (optional fields
are dropped from ``required``, and ``additionalProperties`` is simply absent).
``to_strict_schema`` fixes both, inlines ``$defs`` references, and
``lint_strict_schema`` is used by a unit test to keep us honest.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel


def _inline_refs(node: Any, defs: dict[str, Any]) -> Any:
    if isinstance(node, dict):
        if "$ref" in node:
            name = node["$ref"].rsplit("/", 1)[-1]
            target = defs.get(name)
            if target is None:
                raise ValueError(f"unresolved $ref: {node['$ref']}")
            merged = _inline_refs(dict(target), defs)
            for k, v in node.items():
                if k != "$ref":
                    merged[k] = v
            return merged
        return {k: _inline_refs(v, defs) for k, v in node.items() if k != "$defs"}
    if isinstance(node, list):
        return [_inline_refs(v, defs) for v in node]
    return node


def _strictify(node: Any) -> Any:
    if isinstance(node, dict):
        out = {k: _strictify(v) for k, v in node.items()}
        if out.get("type") == "object" and "properties" in out:
            out["additionalProperties"] = False
            out["required"] = list(out["properties"].keys())
        # Strict mode rejects these annotations on some nesting levels.
        for noisy in ("title", "default", "examples"):
            out.pop(noisy, None)
        return out
    if isinstance(node, list):
        return [_strictify(v) for v in node]
    return node


def to_strict_schema(model: type[BaseModel]) -> dict[str, Any]:
    raw = model.model_json_schema()
    defs = raw.get("$defs", {})
    return _strictify(_inline_refs(raw, defs))


def response_format(model: type[BaseModel], name: str | None = None) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name or model.__name__.lower(),
            "strict": True,
            "schema": to_strict_schema(model),
        },
    }


def lint_strict_schema(schema: Any, path: str = "$") -> list[str]:
    """Return a list of strict-mode violations. Empty list means compliant."""
    problems: list[str] = []
    if isinstance(schema, dict):
        if schema.get("type") == "object" and "properties" in schema:
            if schema.get("additionalProperties") is not False:
                problems.append(f"{path}: additionalProperties must be false")
            required = set(schema.get("required", []))
            props = set(schema["properties"])
            if props != required:
                problems.append(f"{path}: required must list every property; missing {sorted(props - required)}")
            for k, v in schema["properties"].items():
                problems += lint_strict_schema(v, f"{path}.{k}")
        if "$ref" in schema:
            problems.append(f"{path}: unresolved $ref (must be inlined)")
        for key in ("items", "anyOf", "oneOf", "allOf"):
            if key in schema:
                problems += lint_strict_schema(schema[key], f"{path}.{key}")
    elif isinstance(schema, list):
        for i, v in enumerate(schema):
            problems += lint_strict_schema(v, f"{path}[{i}]")
    return problems
