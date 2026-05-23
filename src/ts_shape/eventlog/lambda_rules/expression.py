"""AST-restricted Python expression compiler.

Lambda-rule triggers are written as short Python-syntax expressions
(``"(torque > 75) & (state == 'run')"``). To run them safely we parse
with :func:`ast.parse` in ``mode="eval"`` and walk the tree, rejecting
any node that is not on a small allowlist.

Operator gotcha: Python's bitwise ``&`` / ``|`` (the vectorized ones
pandas understands) bind **tighter** than comparison operators, so you
must wrap each comparison in parentheses:
``"(x > 1) & (y < 0)"`` — not ``"x > 1 & y < 0"``. This matches the
convention used by ``pandas.eval`` and ``DataFrame.query``.

Why an AST whitelist over ``pandas.eval`` or ``polars.expr``?

* Full control over error messages.
* Reliable Int64 / nullable-bool handling.
* Zero extra dependencies — stdlib :mod:`ast` only.
* The whitelist refuses dunders, attribute access, imports, comprehensions,
  and any function call outside of ``abs``, ``min``, ``max`` — so an
  LLM-proposed expression cannot exfiltrate or mutate state.
"""

from __future__ import annotations

import ast
from collections.abc import Callable

import pandas as pd


class UnsafeExpression(ValueError):
    """Raised when a trigger expression contains a disallowed AST node."""


_ALLOWED_FUNCS: frozenset[str] = frozenset({"abs", "min", "max"})

_ALLOWED_NODES: tuple[type, ...] = (
    ast.Expression,
    ast.BoolOp,
    ast.BinOp,
    ast.UnaryOp,
    ast.Compare,
    ast.Name,
    ast.Constant,
    ast.Load,
    ast.And,
    ast.Or,
    ast.Not,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Mod,
    ast.USub,
    ast.UAdd,
    ast.BitAnd,
    ast.BitOr,
    ast.BitXor,
    ast.Invert,
    ast.Call,
    ast.Tuple,
    ast.List,
    ast.In,
    ast.NotIn,
)


def _validate(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise UnsafeExpression(
                f"expression contains disallowed syntax: "
                f"{type(node).__name__} is not on the AST whitelist"
            )
        if isinstance(node, ast.Call):
            if (
                not isinstance(node.func, ast.Name)
                or node.func.id not in _ALLOWED_FUNCS
            ):
                raise UnsafeExpression(
                    "only abs / min / max calls are allowed in trigger expressions"
                )


def compile_expression(expression: str) -> Callable[[pd.DataFrame], pd.Series]:
    """Compile a trigger expression to a vectorized boolean mask function.

    The returned callable accepts a :class:`pandas.DataFrame` and returns
    a :class:`pandas.Series` of booleans (NaN → False) aligned with the
    frame's rows.
    """
    if not isinstance(expression, str) or not expression.strip():
        raise UnsafeExpression("expression must be a non-empty string")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise UnsafeExpression(f"could not parse expression: {exc}") from exc
    _validate(tree)
    code = compile(tree, filename="<lambda-rule>", mode="eval")

    def _evaluate(df: pd.DataFrame) -> pd.Series:
        # Build a name → Series mapping plus the three allow-listed builtins.
        local_ns: dict[str, object] = {col: df[col] for col in df.columns}
        global_ns: dict[str, object] = {
            "__builtins__": {},
            "abs": abs,
            "min": min,
            "max": max,
        }
        result = eval(code, global_ns, local_ns)  # noqa: S307 — sandboxed
        if not isinstance(result, pd.Series):
            # Constant expression: broadcast.
            result = pd.Series([bool(result)] * len(df), index=df.index)
        return result.fillna(False).astype(bool)

    return _evaluate
