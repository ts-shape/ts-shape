"""Unit tests for the shared loader helpers in ``ts_shape.loader._utils``."""

import pytest

from ts_shape.errors import LoaderError
from ts_shape.loader._utils import (
    require_config,
    retry_call,
    validate_local_path,
)

# ---------------------------------------------------------------------------
# validate_local_path
# ---------------------------------------------------------------------------


def test_validate_local_path_returns_path_when_present(tmp_path):
    assert validate_local_path(tmp_path, must_be_dir=True) == tmp_path


def test_validate_local_path_raises_when_missing(tmp_path):
    missing = tmp_path / "nope"
    with pytest.raises(LoaderError, match="does not exist"):
        validate_local_path(missing)


def test_validate_local_path_raises_when_not_a_dir(tmp_path):
    f = tmp_path / "file.txt"
    f.write_text("x")
    with pytest.raises(LoaderError, match="not a directory"):
        validate_local_path(f, must_be_dir=True)


# ---------------------------------------------------------------------------
# require_config
# ---------------------------------------------------------------------------


def test_require_config_passes_when_all_keys_present():
    require_config({"a": 1, "b": 2}, ["a", "b"], name="cfg")


def test_require_config_lists_missing_keys():
    with pytest.raises(LoaderError, match=r"missing required key\(s\): \['b'\]"):
        require_config({"a": 1}, ["a", "b"], name="cfg")


# ---------------------------------------------------------------------------
# retry_call
# ---------------------------------------------------------------------------


def test_retry_call_returns_first_success():
    calls = []

    def ok():
        calls.append(1)
        return "value"

    assert retry_call(ok, sleep=lambda _: None) == "value"
    assert len(calls) == 1


def test_retry_call_retries_then_succeeds():
    attempts = {"n": 0}
    sleeps: list[float] = []

    def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise OSError("transient")
        return "ok"

    out = retry_call(
        flaky,
        attempts=3,
        initial_delay=0.1,
        backoff=2.0,
        sleep=sleeps.append,
    )
    assert out == "ok"
    assert attempts["n"] == 3
    # Backoff: waited before attempt 2 (0.1) and attempt 3 (0.2).
    assert sleeps == [0.1, 0.2]


def test_retry_call_raises_loader_error_after_exhaustion():
    def always_fail():
        raise OSError("boom")

    with pytest.raises(LoaderError, match="failed after 2 attempt"):
        retry_call(always_fail, attempts=2, sleep=lambda _: None)


def test_retry_call_excluded_exception_short_circuits():
    calls = {"n": 0}

    def missing():
        calls["n"] += 1
        raise FileNotFoundError("no data")

    with pytest.raises(FileNotFoundError):
        retry_call(
            missing,
            attempts=5,
            exclude=(FileNotFoundError,),
            sleep=lambda _: None,
        )
    # Excluded errors are not retried.
    assert calls["n"] == 1


def test_retry_call_does_not_retry_unlisted_exception():
    calls = {"n": 0}

    def bad():
        calls["n"] += 1
        raise ValueError("not transient")

    with pytest.raises(ValueError):
        retry_call(
            bad,
            attempts=5,
            retry_exceptions=(OSError,),
            sleep=lambda _: None,
        )
    assert calls["n"] == 1


def test_retry_call_rejects_invalid_attempts():
    with pytest.raises(ValueError, match="attempts must be >= 1"):
        retry_call(lambda: None, attempts=0)
