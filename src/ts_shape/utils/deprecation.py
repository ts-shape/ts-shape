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
* :func:`deprecated_function` -- a function got a new name.
* :func:`moved_module` -- a module moved; the old module forwards every name.
* :func:`moved_attribute` -- some names of a package moved to another module.

The wrapped callables report the *new* signature to :func:`inspect.signature`
(and so to the API reference and IDEs).

The deprecated objects are also marked the PEP 702 way (``__deprecated__``,
via :func:`warnings.deprecated` on Python 3.13+), without a second runtime
warning. Static type checkers only see a ``@deprecated`` decorator written in
the source, so the defining modules also declare each alias with
``typing_extensions.deprecated`` inside an ``if TYPE_CHECKING:`` block; mypy
(``--enable-error-code deprecated``), pyright and IDEs then flag its use.
"""

from __future__ import annotations

import functools
import importlib
import inspect
import sys
import warnings
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

__all__ = [
    "renamed_kwargs",
    "deprecated_method",
    "deprecated_class",
    "keyword_only",
    "caller_stacklevel",
    "deprecated_attribute",
    "deprecated_function",
    "moved_module",
    "moved_attribute",
]

F = TypeVar("F", bound=Callable[..., Any])
C = TypeVar("C", bound=type)

REMOVAL = "the next major release"

if TYPE_CHECKING:
    from typing_extensions import deprecated as _static_deprecated
else:

    def _static_deprecated(message: str) -> Callable[[F], F]:
        """``typing_extensions.deprecated`` for type checkers, a no-op at runtime."""
        return lambda func: func


if sys.version_info >= (3, 13):
    from warnings import deprecated as _pep702_deprecated
else:  # pragma: no cover - exercised on Python 3.11/3.12
    _pep702_deprecated = None


def _mark_deprecated(obj: Any, message: str) -> Any:
    """Mark ``obj`` deprecated for introspection (PEP 702), without warning.

    ``category=None`` makes :func:`warnings.deprecated` only set
    ``__deprecated__``; the helpers below emit their own warning (with the
    stack level pointing at the user's code), so there is no double warning.
    Before Python 3.13 the attribute is set directly, which is all the
    decorator does with ``category=None`` (no ``typing_extensions`` import).
    """
    if _pep702_deprecated is not None and callable(obj):
        return _pep702_deprecated(message, category=None)(obj)
    try:
        obj.__deprecated__ = message
    except (AttributeError, TypeError):  # pragma: no cover - read-only object
        pass
    return obj


_PACKAGE_DIR = Path(__file__).resolve().parent.parent


def _in_package(filename: str) -> bool:
    try:
        return Path(filename).resolve().is_relative_to(_PACKAGE_DIR)
    except (OSError, ValueError):
        return False


def caller_stacklevel() -> int:
    """``stacklevel`` that points a warning at the first frame outside ts_shape.

    Call it directly in the ``warnings.warn(...)`` call. Detector methods are
    wrapped (signal checks, keyword-only and renamed-keyword shims), so a
    fixed number would attribute the warning to a wrapper inside the package,
    and Python's default filters would then hide a ``DeprecationWarning``.
    """
    frame = sys._getframe(1)
    level = 1
    while frame.f_back is not None and _in_package(frame.f_code.co_filename):
        frame = frame.f_back
        level += 1
    return level


def _warn(message: str) -> None:
    warnings.warn(
        f"{message}; the old name will be removed in {REMOVAL}",
        DeprecationWarning,
        stacklevel=caller_stacklevel(),
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

            message = f"{cls_name}.{self.old_name}() is deprecated, use .{new_name}()"

            @functools.wraps(target)
            def forward(*args: Any, **kwargs: Any) -> Any:
                _warn(message)
                return target(*args, **kwargs)

            forward.__deprecated_alias_of__ = new_name  # type: ignore[attr-defined]
            return _mark_deprecated(forward, message)

    return _Forward()


def deprecated_class(new_cls: C, old_name: str) -> C:
    """A subclass of ``new_cls`` named ``old_name`` that warns when created.

    Instances are instances of ``new_cls``, so ``isinstance`` checks and the
    event-log taxonomy (which resolves aliases) keep working.
    """

    message = f"{old_name} is deprecated, use {new_cls.__name__}"

    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        _warn(message)
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
    return _mark_deprecated(alias, message)  # type: ignore[no-any-return]


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
        # Read by scripts/deprecation_report.py; no effect on behavior.
        wrapper.__keyword_only__ = tuple(movable)  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    return decorate


class deprecated_attribute:  # noqa: N801 - used like a decorator/property
    """A class attribute that forwards to ``new_name`` with a warning.

    Example:
        >>> class Loader:
        ...     start_timestamp = deprecated_attribute("start")

    Reading or setting ``loader.start_timestamp`` warns and uses
    ``loader.start``.
    """

    def __init__(self, new_name: str) -> None:
        self.new_name = new_name
        self.old_name = ""
        self.owner = ""

    def __set_name__(self, owner: type, name: str) -> None:
        self.old_name = name
        self.owner = owner.__name__
        # Seen when the attribute is looked up on the class (PEP 702).
        self.__deprecated__ = self._message()

    def _message(self) -> str:
        return (
            f"{self.owner}.{self.old_name} is deprecated, use "
            f"{self.owner}.{self.new_name}"
        )

    # Type checkers flag every read or write of an attribute declared with this
    # descriptor; the runtime warning names the replacement.
    @_static_deprecated("deprecated attribute; the warning names its replacement")
    def __get__(self, obj: Any, objtype: Any = None) -> Any:
        if obj is None:
            return self
        _warn(self._message())
        return getattr(obj, self.new_name)

    @_static_deprecated("deprecated attribute; the warning names its replacement")
    def __set__(self, obj: Any, value: Any) -> None:
        _warn(self._message())
        setattr(obj, self.new_name, value)


def deprecated_function(new_func: F, old_name: str) -> F:
    """A function named ``old_name`` that forwards to ``new_func`` with a warning."""

    message = f"{old_name}() is deprecated, use {new_func.__name__}()"

    @functools.wraps(new_func)
    def forward(*args: Any, **kwargs: Any) -> Any:
        _warn(message)
        return new_func(*args, **kwargs)

    forward.__name__ = forward.__qualname__ = old_name
    forward.__deprecated_alias_of__ = new_func  # type: ignore[attr-defined]
    return _mark_deprecated(forward, message)  # type: ignore[no-any-return]


def moved_module(
    old: str, new: str
) -> tuple[Callable[[str], Any], Callable[[], list[str]]]:
    """``__getattr__`` and ``__dir__`` for a module that moved to ``new``.

    The old module keeps only these two hooks: every name read from it is
    taken from ``new``, with a warning naming the new module. Importing the
    old module alone does not warn, and its ``__dir__`` lists nothing, so
    tools that walk the package (docs, tests) do not trigger the warning.

    Example (the whole old module)::

        from ts_shape.utils.deprecation import moved_module

        __getattr__, __dir__ = moved_module(
            "ts_shape.cycle_profile", "ts_shape.events.profiling.runner"
        )
    """

    def __getattr__(name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(f"module {old!r} has no attribute {name!r}")
        target = importlib.import_module(new)
        if not hasattr(target, name):
            raise AttributeError(f"module {old!r} has no attribute {name!r}")
        _warn(f"{old} moved to {new}; import {name} from {new}")
        return getattr(target, name)

    def __dir__() -> list[str]:
        return []

    return __getattr__, __dir__


def moved_attribute(module: str, name: str, moved: Mapping[str, str]) -> Any:
    """Resolve ``name``, which moved from package ``module`` to ``moved[name]``.

    For a package ``__getattr__`` whose other names stay::

        def __getattr__(name):
            return moved_attribute(__name__, name, _MOVED)
    """
    target = moved.get(name)
    if target is None:
        raise AttributeError(f"module {module!r} has no attribute {name!r}")
    _warn(f"{module}.{name} moved to {target}; import {name} from {target}")
    return getattr(importlib.import_module(target), name)
