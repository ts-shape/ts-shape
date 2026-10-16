import inspect
import warnings

import pytest

from ts_shape.utils.deprecation import (
    deprecated_attribute,
    deprecated_class,
    deprecated_method,
    keyword_only,
    renamed_kwargs,
)


@renamed_kwargs(freq="window")
def per_window(df, window="1h"):
    return df, window


def test_renamed_kwargs_maps_and_warns():
    with pytest.warns(DeprecationWarning, match="'freq' is deprecated, use 'window'"):
        assert per_window(1, freq="1D") == (1, "1D")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert per_window(1, window="2h") == (1, "2h")
    with pytest.raises(TypeError, match="both"):
        per_window(1, freq="1D", window="2h")
    assert list(inspect.signature(per_window).parameters) == ["df", "window"]


class New:
    def __init__(self, x, y=2):
        self.x, self.y = x, y

    def oee(self, k=1):
        return self.x * k

    calculate_oee = deprecated_method("oee")


Old = deprecated_class(New, "Old")


def test_deprecated_method_forwards():
    with pytest.warns(
        DeprecationWarning, match=r"New.calculate_oee\(\) is deprecated, use .oee\(\)"
    ):
        assert New(3).calculate_oee(k=2) == 6
    assert inspect.signature(New(3).calculate_oee) == inspect.signature(New(3).oee)


def test_deprecated_class_is_a_subclass_and_warns():
    with pytest.warns(DeprecationWarning, match="Old is deprecated, use New"):
        obj = Old(5, y=7)
    assert isinstance(obj, New) and obj.y == 7
    assert Old.__name__ == "Old"
    assert Old.__deprecated_alias_of__ is New


class K:
    @keyword_only(2)
    def method(self, a, b=1, c=2):
        return a, b, c


def test_keyword_only_signature_and_positional_shim():
    sig = inspect.signature(K.method)
    assert sig.parameters["b"].kind is inspect.Parameter.KEYWORD_ONLY
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert K().method(1, b=5) == (1, 5, 2)
    with pytest.warns(DeprecationWarning, match="passing b positionally"):
        assert K().method(1, 5) == (1, 5, 2)


class WithAttr:
    def __init__(self):
        self.start = 1

    start_timestamp = deprecated_attribute("start")


def test_deprecated_objects_are_marked_pep702_and_warn_once():
    """``__deprecated__`` is set for tools; the runtime warning is not doubled."""
    assert Old.__deprecated__ == "Old is deprecated, use New"
    assert New.__dict__.get("__deprecated__") is None
    assert New(3).calculate_oee.__deprecated__ == (
        "New.calculate_oee() is deprecated, use .oee()"
    )
    assert WithAttr.__dict__["start_timestamp"].__deprecated__ == (
        "WithAttr.start_timestamp is deprecated, use WithAttr.start"
    )
    for call in (
        lambda: Old(5),
        lambda: New(3).calculate_oee(),
        lambda: WithAttr().start_timestamp,
    ):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            call()
        assert [w.category for w in caught] == [DeprecationWarning]
