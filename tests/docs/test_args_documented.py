"""Every parameter of every public method is documented in its docstring."""

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "ts_shape"


def _undocumented():
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for cls in tree.body:
            if not isinstance(cls, ast.ClassDef) or cls.name.startswith("_"):
                continue
            for fn in cls.body:
                if not isinstance(fn, ast.FunctionDef):
                    continue
                if fn.name.startswith("_") and fn.name != "__init__":
                    continue
                if any(
                    isinstance(d, ast.Name) and d.id == "overload"
                    for d in fn.decorator_list
                ):
                    continue
                params = [
                    a.arg
                    for a in fn.args.args + fn.args.kwonlyargs
                    if a.arg not in ("self", "cls")
                ]
                doc = ast.get_docstring(fn) or ""
                if fn.name == "__init__" and not doc:
                    continue  # documented on the class
                missing = [
                    p for p in params if f"{p}:" not in doc and f"{p} (" not in doc
                ]
                if missing:
                    rel = path.relative_to(SRC.parent)
                    yield f"{rel}:{fn.lineno} {cls.name}.{fn.name}: {missing}"


def test_every_public_parameter_is_documented():
    missing = list(_undocumented())
    assert not missing, "document these parameters (Args:):\n" + "\n".join(missing)
