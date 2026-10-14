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
* The whitelist refuses dunders, attribute access (except the
  ``.isin([...])`` method), imports, comprehensions, and any function call
  outside of ``abs``, ``min``, ``max`` — so an LLM-proposed expression
  cannot exfiltrate or mutate state.
* Python's ``and`` / ``or`` / ``not`` / ``in`` are not element-wise on
  columns; they are rejected at compile time with a hint to ``&``, ``|``,
  ``~`` and ``.isin(...)``.
"""

from __future__ import annotations

import ast
import functools
from collections.abc import Callable
from types import CodeType

import numpy as np
import pandas as pd

from ...utils.frames import require_columns


class UnsafeExpression(ValueError):
    """Raised when a trigger expression contains a disallowed AST node."""


_ALLOWED_FUNCS: frozenset[str] = frozenset({"abs", "min", "max"})
#: Methods callable on a column, e.g. ``state.isin(['run', 'setup'])``.
_ALLOWED_METHODS: frozenset[str] = frozenset({"isin"})

_ALLOWED_NODES: tuple[type, ...] = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Compare,
    ast.Name,
    ast.Constant,
    ast.Load,
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
    ast.Attribute,
    ast.Tuple,
    ast.List,
)

_HINT = (
    "trigger expressions are element-wise: use & (and), | (or), ~ (not) and "
    "column.isin([...]) (in), with each comparison in parentheses, e.g. "
    "\"(torque > 75) & ~(state.isin(['idle', 'off']))\""
)

_ARITHMETIC = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod)


def _reject_python_logic(node: ast.AST) -> None:
    if isinstance(node, ast.BoolOp):
        word = "and" if isinstance(node.op, ast.And) else "or"
        raise UnsafeExpression(f"{word!r} is not element-wise; {_HINT}")
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        raise UnsafeExpression(f"'not' is not element-wise; {_HINT}")
    if isinstance(node, ast.Compare):
        if any(isinstance(op, ast.In | ast.NotIn) for op in node.ops):
            raise UnsafeExpression(f"'in' / 'not in' is not element-wise; {_HINT}")
        if len(node.ops) > 1:
            raise UnsafeExpression(
                "chained comparison (e.g. 'a > 1 & b == 2' parses as "
                f"'a > (1 & b) == 2'); {_HINT}"
            )


def _validate(tree: ast.Expression) -> None:
    for node in ast.walk(tree):
        _reject_python_logic(node)
        if not isinstance(node, _ALLOWED_NODES):
            raise UnsafeExpression(
                f"expression contains disallowed syntax: "
                f"{type(node).__name__} is not on the AST whitelist"
            )
        if isinstance(node, ast.Attribute):
            if node.attr not in _ALLOWED_METHODS or node.attr.startswith("_"):
                raise UnsafeExpression(
                    f"attribute access {node.attr!r} is not allowed; only "
                    f"{sorted(_ALLOWED_METHODS)} methods can be called"
                )
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                if node.keywords or len(node.args) != 1:
                    raise UnsafeExpression(
                        f".{node.func.attr}() takes one list of values"
                    )
                continue
            if (
                not isinstance(node.func, ast.Name)
                or node.func.id not in _ALLOWED_FUNCS
            ):
                raise UnsafeExpression(
                    "only abs / min / max calls are allowed in trigger expressions"
                )
            if node.func.id in ("min", "max") and any(
                isinstance(n, ast.Name) for a in node.args for n in ast.walk(a)
            ):
                raise UnsafeExpression(
                    f"{node.func.id}() on a column is not element-wise; compare "
                    "each column instead, e.g. '(a > 5) | (b > 5)' for "
                    "'max(a, b) > 5'"
                )
    # Attributes are only allowed as the method of a call.
    callee = {id(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and id(node) not in callee:
            raise UnsafeExpression(
                f"attribute access {node.attr!r} is only allowed as a method call"
            )
    body = tree.body
    if (isinstance(body, ast.BinOp) and isinstance(body.op, _ARITHMETIC)) or (
        isinstance(body, ast.Constant) and not isinstance(body.value, bool)
    ):
        raise UnsafeExpression(
            "the expression must be a condition (True / False per row), e.g. "
            "'torque > 75'; arithmetic alone is not a trigger"
        )


def _column_names(tree: ast.Expression) -> list[str]:
    """Column names an expression reads (names that are not functions)."""
    funcs = {
        id(n.func)
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    names = [
        n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and id(n) not in funcs
    ]
    return list(dict.fromkeys(names))


def _as_mask(result: object, df: pd.DataFrame, expression: str) -> pd.Series:
    if not isinstance(result, pd.Series):
        if isinstance(result, bool | np.bool_):
            return pd.Series([bool(result)] * len(df), index=df.index)
        raise ValueError(
            f"trigger expression {expression!r} gives {type(result).__name__}, "
            "not a True / False value per row"
        )
    if pd.api.types.is_bool_dtype(result.dtype):
        return result.fillna(False).astype(bool)
    if result.dtype == object:
        present = result.dropna()
        if all(isinstance(v, bool | np.bool_) for v in present.tolist()):
            return result.where(result.notna(), False).astype(bool)
    raise ValueError(
        f"trigger expression {expression!r} gives {result.dtype} values, not "
        "True / False; write a condition such as 'torque > 75'"
    )


@functools.lru_cache(maxsize=512)
def _compile(expression: str) -> tuple[CodeType, tuple[str, ...]]:
    if not isinstance(expression, str) or not expression.strip():
        raise UnsafeExpression("expression must be a non-empty string")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise UnsafeExpression(f"could not parse expression: {exc}") from exc
    _validate(tree)
    code = compile(tree, filename="<lambda-rule>", mode="eval")
    return code, tuple(_column_names(tree))


def compile_expression(expression: str) -> Callable[[pd.DataFrame], pd.Series]:
    """Compile a trigger expression to a vectorized boolean mask function.

    Python's ``and`` / ``or`` / ``not`` / ``in``, chained comparisons and
    ``min`` / ``max`` over columns are rejected here, with a hint to the
    element-wise ``&``, ``|``, ``~`` and ``column.isin([...])``.

    Args:
        expression: The trigger, e.g. ``"(torque > 75) & (state == 'run')"``.

    Returns:
        A callable that takes a DataFrame and returns a boolean Series
        aligned with its rows (missing -> False).

    Raises:
        UnsafeExpression: For disallowed or non-element-wise syntax. The
            callable raises :class:`~ts_shape.errors.ColumnNotFoundError` for
            a name that is not a column and ``ValueError`` when the result is
            not boolean.
    """
    if not isinstance(expression, str):
        raise UnsafeExpression("expression must be a non-empty string")
    code, names = _compile(expression)

    def _evaluate(df: pd.DataFrame) -> pd.Series:
        require_columns(df, names, where=f"trigger expression {expression!r}")
        # Build a name → Series mapping plus the three allow-listed builtins.
        local_ns: dict[str, object] = {col: df[col] for col in names}
        global_ns: dict[str, object] = {
            "__builtins__": {},
            "abs": abs,
            "min": min,
            "max": max,
        }
        result = eval(code, global_ns, local_ns)  # noqa: S307 — sandboxed
        return _as_mask(result, df, expression)

    return _evaluate
