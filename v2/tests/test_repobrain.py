"""The stdlib-ast code structure adapter: the "repo brain" interface.

Proves the three things the pipeline actually depends on: it finds route
registrations and call edges correctly, a non-Python or unparsable file is
reported as a partial-source gap rather than silently dropped, and a
decorator or a nested function's calls are not misattributed to the
function they decorate/enclose.
"""
from __future__ import annotations

import json

from rba.repobrain import index, symbol_at_line

ROUTES_PY = '''
from fastapi import APIRouter
router = APIRouter()

def _resolve(session, body):
    return session.doc, body.target

@router.post("/api/score")
async def score_one(req):
    doc, target = _resolve(req.session, req.body)
    def inner():
        return helper_only_in_inner()
    inner()
    return score_many([doc], target)
'''

SCORING_PY = '''
def score_many(docs, target):
    return [_score(d, target) for d in docs]

def _score(doc, target):
    return 0.5
'''


def _demo_brain():
    return index({
        "app/routes.py": ROUTES_PY,
        "app/scoring.py": SCORING_PY,
        "README.md": "# hi",
        "broken.py": "def f(:\n  pass",
    })


def test_finds_route_registrations_and_their_handler():
    brain = _demo_brain()
    routes = brain.all_routes()
    assert len(routes) == 1
    assert routes[0].method == "POST"
    assert routes[0].pattern == "/api/score"
    assert routes[0].handler == "app/routes.py::score_one"


def test_non_python_and_unparsable_files_are_reported_not_dropped():
    brain = _demo_brain()
    gaps = dict(g.split(": ", 1) for g in brain.partial_source)
    assert "README.md" in gaps and "non-Python" in gaps["README.md"]
    assert "broken.py" in gaps and "SyntaxError" in gaps["broken.py"]


def test_to_dict_is_json_serialisable_and_carries_the_real_structure():
    """The Evidence tab's API response is `RepoBrain.to_dict()` -- if this
    isn't plain JSON-safe data, the endpoint that serves it is broken."""
    brain = _demo_brain()
    payload = brain.to_dict()
    encoded = json.dumps(payload)  # raises if anything isn't JSON-safe
    decoded = json.loads(encoded)

    assert set(decoded.keys()) == {"files", "partial_source"}
    routes_file = decoded["files"]["app/routes.py"]
    assert {"path", "symbols", "calls", "routes", "parse_error"} <= routes_file.keys()

    symbol_names = {s["name"] for s in routes_file["symbols"]}
    assert {"_resolve", "score_one", "inner"} <= symbol_names

    route = routes_file["routes"][0]
    assert route["method"] == "POST"
    assert route["pattern"] == "/api/score"
    assert route["handler"] == "app/routes.py::score_one"

    call_names = {c["callee"] for c in routes_file["calls"]}
    assert "_resolve" in call_names
    # broken.py is still a known (.py) file -- just one with no usable
    # symbols -- while README.md, not being Python, is never indexed at all.
    assert brain.files["broken.py"].symbols == []
    assert "README.md" not in brain.files
    assert {"app/routes.py", "app/scoring.py"} <= set(brain.files)


def test_callers_and_callees_cross_file():
    brain = _demo_brain()
    callers = brain.callers("_resolve")
    assert len(callers) == 1
    assert callers[0].caller == "app/routes.py::score_one"
    callees = {c.callee for c in brain.callees("app/routes.py::score_one")}
    assert callees == {"_resolve", "inner", "score_many"}


def test_decorator_call_is_not_attributed_as_a_callee():
    """@router.post(...) is applied at definition time, not called by the
    function it decorates -- it must never appear as score_one's callee."""
    brain = _demo_brain()
    callees = {c.callee for c in brain.callees("app/routes.py::score_one")}
    assert "post" not in callees


def test_nested_function_calls_are_not_double_counted():
    """A closure's own call must belong to the closure, not also to its
    enclosing function."""
    brain = _demo_brain()
    assert [c.callee for c in brain.callees("app/routes.py::inner")] == ["helper_only_in_inner"]
    assert "helper_only_in_inner" not in {
        c.callee for c in brain.callees("app/routes.py::score_one")}


def test_call_chain_crosses_files():
    brain = _demo_brain()
    chain = brain.call_chain("app/routes.py::score_one", "_score")
    assert chain == ["score_one", "score_many", "_score"]


def test_call_chain_returns_none_when_unreachable():
    brain = _demo_brain()
    assert brain.call_chain("app/routes.py::score_one", "nonexistent_function") is None


def test_symbol_at_line_picks_the_tightest_enclosing_range():
    brain = _demo_brain()
    sym = symbol_at_line(brain, "app/routes.py", 9)
    assert sym is not None and sym.name == "score_one"
    assert symbol_at_line(brain, "app/routes.py", 1) is None
