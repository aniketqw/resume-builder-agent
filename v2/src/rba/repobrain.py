"""The "repo brain": a small, swappable code-structure adapter.

This is the boundary the project's spec asked for -- symbol / callers / callees
/ call-chain lookup over a repository's source -- implemented with the stdlib
`ast` module: no clone, no server, no new dependency.

CodeGraphContext (Neo4j + tree-sitter) was evaluated for this role and ruled
out: it forces protobuf down to 3.20.3, which breaks the `onnxruntime` version
the ATS dense scorer needs (see v2/trace.md for the verified conflict). Should
it ever become installable again, it can implement this same `RepoBrain`
shape without the summarizer or the verifier changing at all.

Scope, honestly stated: this indexes Python only, and resolves calls by bare
name within the files actually fetched -- there is no import resolution, so
two functions with the same name in different files are not disambiguated.
That is a real limitation, not a silent one: every file that could not be
indexed (wrong language, or a parse error) is recorded in `partial_source`
rather than dropped.
"""
from __future__ import annotations

import ast
from collections import deque
from dataclasses import dataclass, field

#: Decorator attribute names that mark an HTTP route handler, e.g.
#: ``@router.post("/x")`` or ``@app.get("/x")``. Covers FastAPI/Flask-style
#: routing, which is what this project's own API uses.
_ROUTE_VERBS = {"get", "post", "put", "patch", "delete", "websocket"}


@dataclass
class Symbol:
    """A function, async function, or class, with its exact source location."""
    name: str
    kind: str          # "function" | "async function" | "class" | "method"
    path: str
    line_start: int
    line_end: int
    qualname: str = ""  # "<path>::<name>", unique within the indexed set

    def to_dict(self) -> dict:
        return {"name": self.name, "kind": self.kind, "path": self.path,
                "line_start": self.line_start, "line_end": self.line_end,
                "qualname": self.qualname}


@dataclass
class CallEdge:
    """``caller`` called something named ``callee`` at ``path``:``line``.

    ``callee`` is the bare name as written at the call site (e.g. the `y` in
    `x.y(...)`). It is NOT resolved against imports -- see the module
    docstring -- so a caller/callee lookup matches by name, honestly.
    """
    caller: str
    callee: str
    path: str
    line: int

    def to_dict(self) -> dict:
        return {"caller": self.caller, "callee": self.callee,
                "path": self.path, "line": self.line}


@dataclass
class RouteBinding:
    method: str
    pattern: str
    handler: str        # qualname of the decorated function
    path: str
    line: int

    def to_dict(self) -> dict:
        return {"method": self.method, "pattern": self.pattern,
                "handler": self.handler, "path": self.path, "line": self.line}


@dataclass
class FileIndex:
    path: str
    symbols: list[Symbol] = field(default_factory=list)
    calls: list[CallEdge] = field(default_factory=list)
    routes: list[RouteBinding] = field(default_factory=list)
    parse_error: str = ""

    def to_dict(self) -> dict:
        return {"path": self.path,
                "symbols": [s.to_dict() for s in self.symbols],
                "calls": [c.to_dict() for c in self.calls],
                "routes": [r.to_dict() for r in self.routes],
                "parse_error": self.parse_error}


@dataclass
class RepoBrain:
    """The adapter surface: symbol / callers / callees / call_chain lookup."""
    files: dict[str, FileIndex] = field(default_factory=dict)
    #: ``"<path>: <reason>"`` for every file that could not be indexed.
    #: Never silent -- a gap here is a fact, reported upstream as a
    #: needs-review item, not swallowed.
    partial_source: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"files": {path: idx.to_dict() for path, idx in self.files.items()},
                "partial_source": list(self.partial_source)}

    def all_symbols(self) -> list[Symbol]:
        return [s for idx in self.files.values() for s in idx.symbols]

    def all_calls(self) -> list[CallEdge]:
        return [c for idx in self.files.values() for c in idx.calls]

    def all_routes(self) -> list[RouteBinding]:
        return [r for idx in self.files.values() for r in idx.routes]

    def symbol(self, name: str) -> list[Symbol]:
        """Every symbol with this bare name (there may be more than one --
        no cross-file resolution is attempted)."""
        return [s for s in self.all_symbols() if s.name == name]

    def callers(self, name: str) -> list[CallEdge]:
        """Every call site whose callee bare-name matches ``name``."""
        return [c for c in self.all_calls() if c.callee == name]

    def callees(self, qualname: str) -> list[CallEdge]:
        """Every call made from inside the function identified by
        ``qualname`` (as produced by :func:`index`, ``"<path>::<name>"``)."""
        return [c for c in self.all_calls() if c.caller == qualname]

    def call_chain(self, start: str, goal: str, max_depth: int = 6) -> list[str] | None:
        """BFS over bare-name call edges from ``start`` (a qualname or a bare
        name) to a symbol named ``goal``. Returns the chain of names, or
        ``None`` if no path is found within ``max_depth`` hops.

        Matching is by bare name, consistent with :meth:`callees` -- this
        traces *plausible* call chains in an unresolved graph, not a proven
        single path. That limitation is inherent to not parsing imports, and
        it is the same limitation every lookup in this class carries.
        """
        start_name = start.rsplit("::", 1)[-1]
        if start_name == goal:
            return [start_name]
        seen = {start_name}
        queue: deque[list[str]] = deque([[start_name]])
        while queue:
            chain = queue.popleft()
            if len(chain) > max_depth:
                continue
            last = chain[-1]
            # A bare name may be ambiguous; follow every matching symbol's
            # qualname so callees() can find its edges.
            for sym in self.symbol(last):
                for edge in self.callees(sym.qualname):
                    if edge.callee in seen:
                        continue
                    nxt = [*chain, edge.callee]
                    if edge.callee == goal:
                        return nxt
                    seen.add(edge.callee)
                    queue.append(nxt)
        return None


def _route_from_decorator(dec: ast.expr) -> tuple[str, str] | None:
    """``@router.post("/x")`` -> ("POST", "/x"). Anything else -> None."""
    if not (isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute)):
        return None
    verb = dec.func.attr.lower()
    if verb not in _ROUTE_VERBS:
        return None
    pattern = ""
    if dec.args and isinstance(dec.args[0], ast.Constant) and isinstance(dec.args[0].value, str):
        pattern = dec.args[0].value
    return verb.upper(), pattern


def _callee_name(call: ast.Call) -> str | None:
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


class _OwnCalls(ast.NodeVisitor):
    """Calls made directly by one function, not by functions nested inside
    it. Without this, a closure's calls would be attributed to both the
    closure and its enclosing function -- double-counted, not just imprecise.
    """
    def __init__(self) -> None:
        self.calls: list[ast.Call] = []

    def visit_Call(self, node: ast.Call) -> None:
        self.calls.append(node)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        pass  # do not descend -- it gets its own pass from the outer walk

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        pass

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        pass


def index_file(path: str, source: str) -> FileIndex:
    """Index one Python file's text. Never raises -- a parse failure is
    recorded on the returned `FileIndex`, not thrown."""
    idx = FileIndex(path=path)
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError as exc:
        idx.parse_error = f"SyntaxError: {exc.msg} (line {exc.lineno})"
        return idx

    def_nodes = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    for node in ast.walk(tree):
        if not isinstance(node, def_nodes):
            continue
        kind = ("class" if isinstance(node, ast.ClassDef)
                else "async function" if isinstance(node, ast.AsyncFunctionDef)
                else "function")
        qualname = f"{path}::{node.name}"
        idx.symbols.append(Symbol(
            name=node.name, kind=kind, path=path,
            line_start=node.lineno, line_end=getattr(node, "end_lineno", node.lineno),
            qualname=qualname,
        ))
        if isinstance(node, ast.ClassDef):
            continue  # calls inside methods are attributed to the method, below
        for dec in node.decorator_list:
            route = _route_from_decorator(dec)
            if route:
                method, pattern = route
                idx.routes.append(RouteBinding(
                    method=method, pattern=pattern, handler=qualname,
                    path=path, line=node.lineno))
        # Only this function's own calls: decorators are applied at
        # definition time (not called by the function), and a nested
        # function's calls belong to it, not its enclosing scope.
        finder = _OwnCalls()
        for stmt in node.body:
            finder.visit(stmt)
        for call in finder.calls:
            name = _callee_name(call)
            if name:
                idx.calls.append(CallEdge(
                    caller=qualname, callee=name, path=path,
                    line=getattr(call, "lineno", node.lineno)))
    return idx


def index(files: dict[str, str]) -> RepoBrain:
    """Index every ``.py`` file in ``files`` (path -> source text).

    Non-Python files are recorded in ``partial_source`` and otherwise
    ignored -- there is no adapter for them yet, and saying so is better than
    pretending the index is complete.
    """
    brain = RepoBrain()
    for path, source in files.items():
        if not path.endswith(".py"):
            brain.partial_source.append(f"{path}: non-Python (no adapter)")
            continue
        file_index = index_file(path, source)
        brain.files[path] = file_index
        if file_index.parse_error:
            brain.partial_source.append(f"{path}: {file_index.parse_error}")
    return brain


def symbol_at_line(brain: RepoBrain, path: str, line: int) -> Symbol | None:
    """The innermost indexed symbol in ``path`` whose range contains ``line``,
    or ``None``. Used to turn a quote's line number into a sensible line
    range for a permalink, rather than a single bare line."""
    idx = brain.files.get(path)
    if idx is None:
        return None
    best: Symbol | None = None
    for sym in idx.symbols:
        if sym.line_start <= line <= sym.line_end:
            if best is None or (sym.line_end - sym.line_start) < (best.line_end - best.line_start):
                best = sym  # prefer the tightest enclosing range
    return best
