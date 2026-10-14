"""Deprecation helpers: rename parameters, methods and classes without breaking callers.

Every helper keeps the old spelling working and emits a
:class:`DeprecationWarning` that names the replacement, so a user's code
keeps running and their test suite tells them what to change. The old names
are removed in the next major release (see ``docs/naming.md``).

* :func:`renamed_kwargs` -- a keyword argument got a new name.
* :func:`deprecated_method` -- a method got a new name.
* :func:`deprecated_class` -- a class got a new name.
* :func:`keyword_only` -- optional arguments become keyword-only; passing
  them positionally still works but warns.

The wrapped callables report the *new* signature to :func:`inspect.signature`
(and so to the API reference and IDEs).
"""

from __future__ import annotations

import functools
import inspect
import warnings
from collections.abc import Callable
from typing import Any, TypeVar

__all__ = [
    "renamed_kwargs",
    "deprecated_method",
    "deprecated_class",
    "keyword_only",
]

F = TypeVar("F", bound=Callable[..., Any])
C = TypeVar("C", bound=type)

REMOVAL = "the next major release"


def _warn(message: str, stacklevel: int = 3) -> None:
    warnings.warn(
        f"{message}; the old name will be removed in {REMOVAL}",
        DeprecationWarning,
        stacklevel=stacklevel,
    )


def renamed_kwargs(**renames: str) -> Callable[[F], F]:
    """Accept old keyword names, mapped to their new names, with a warning.

    Example:
        >>> @renamed_kwargs(freq="window")
        ... def per_window(df, window="1h"): ...

    ``per_window(df, freq="1D")`` then works and warns; passing both the old
    and the new name raises :class:`TypeError`.
    """

    def decorate(func: F) -> F:
        name = func.__qualname__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            for old, new in renames.items():
                if old in kwargs:
                    if new in kwargs:
                        raise TypeError(
                            f"{name}() got both {old!r} (deprecated) and {new!r}"
                        )
                    _warn(f"{name}(): {old!r} is deprecated, use {new!r}")
                    kwargs[new] = kwargs.pop(old)
            return func(*args, **kwargs)

        wrapper.__deprecated_kwargs__ = dict(renames)  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    return decorate


def deprecated_method(new_name: str) -> Any:
    """A class attribute that forwards to ``new_name`` with a warning.

    Example:
        >>> class OEEEvents:
        ...     def oee(self): ...
        ...     calculate_oee = deprecated_method("oee")
    """

    class _Forward:
        def __set_name__(self, owner: type, name: str) -> None:
            self.old_name = name

        def __get__(self, obj: Any, owner: type | None = None) -> Any:
            target = getattr(owner if obj is None else obj, new_name)
            cls_name = (owner or type(obj)).__name__

            @functools.wraps(target)
            def forward(*args: Any, **kwargs: Any) -> Any:
                _warn(
                    f"{cls_name}.{self.old_name}() is deprecated, use " f".{new_name}()"
                )
                return target(*args, **kwargs)

            forward.__deprecated_alias_of__ = new_name  # type: ignore[attr-defined]
            return forward

    return _Forward()


def deprecated_class(new_cls: C, old_name: str) -> C:
    """A subclass of ``new_cls`` named ``old_name`` that warns when created.

    Instances are instances of ``new_cls``, so ``isinstance`` checks and the
    event-log taxonomy (which resolves aliases) keep working.
    """

    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        _warn(f"{old_name} is deprecated, use {new_cls.__name__}")
        new_cls.__init__(self, *args, **kwargs)  # type: ignore[misc]

    alias = type(
        old_name,
        (new_cls,),
        {
            "__init__": __init__,
            "__doc__": f"Deprecated alias of :class:`{new_cls.__name__}`.",
            "__module__": new_cls.__module__,
            "__deprecated_alias_of__": new_cls,
        },
    )
    return alias  # type: ignore[return-value]


def keyword_only(n_positional: int) -> Callable[[F], F]:
    """Make every parameter after the first ``n_positional`` keyword-only.

    ``n_positional`` counts ``self``. Callers that still pass the later
    parameters positionally keep working, with a warning. The reported
    signature shows the keyword-only marker.
    """

    def decorate(func: F) -> F:
        sig = inspect.signature(func)
        params = list(sig.parameters.values())
        names = [
            p.name
            for p in params
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        movable = names[n_positional:]
        new_params = [
            p.replace(kind=p.KEYWORD_ONLY) if p.name in movable else p for p in params
        ]
        # keyword-only parameters must come after positional ones
        new_params.sort(
            key=lambda p: {
                p.POSITIONAL_ONLY: 0,
                p.POSITIONAL_OR_KEYWORD: 1,
                p.VAR_POSITIONAL: 2,
                p.KEYWORD_ONLY: 3,
                p.VAR_KEYWORD: 4,
            }[p.kind]
        )
        name = func.__qualname__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if len(args) > n_positional:
                extra = movable[: len(args) - n_positional]
                _warn(
                    f"{name}(): passing {', '.join(extra)} positionally is "
                    "deprecated, pass them by keyword"
                )
            return func(*args, **kwargs)

        wrapper.__signature__ = sig.replace(parameters=new_params)  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    return decorate
