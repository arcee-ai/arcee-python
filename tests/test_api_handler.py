from time import time

import pytest

from arcee.api_handler import retry_call


def test_retry_call_no_args() -> None:
    @retry_call()
    def bad_func() -> None:
        raise Exception("fail!")

    with pytest.raises(Exception) as e:
        bad_func()
    assert str(e.value) == "fail!"


def test_retry_call_with_args() -> None:
    @retry_call(max_attempts=3, wait_sec=0.05)
    def bad_func() -> None:
        raise Exception("fail thrice!")

    t0 = time()
    with pytest.raises(Exception) as e:
        bad_func()
    t1 = time()
    assert (t1 - t0) < 0.5
    assert str(e.value) == "fail thrice!"


def test_retry_call_from_class() -> None:
    class Test:
        @retry_call(wait_sec=0.05)
        def tryit(self) -> None:
            raise Exception("foo")

    t = Test()
    with pytest.raises(Exception) as e:
        t.tryit()
    assert str(e.value) == "foo"
