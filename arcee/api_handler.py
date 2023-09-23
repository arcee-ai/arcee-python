from functools import partial, wraps
from time import sleep
from typing import Any, Callable, Literal, Optional, TypeVar, Union, cast

import requests
from typing_extensions import ParamSpec

from arcee import config
from arcee.schemas.routes import Route

T = TypeVar("T")
P = ParamSpec("P")


def retry_call(
    func: Optional[Callable[P, T]] = None, max_attempts: int = 2, wait_sec: Union[float, int] = 5
) -> Callable[P, T]:
    """Retry api call"""
    # Allows for using the decorator like @retry_call
    # without having to call it directly (if you don't want to change the default params)
    if func is None:
        return cast(Callable[P, T], partial(retry_call, max_attempts=max_attempts, wait_sec=wait_sec))
    assert wait_sec > 0, "wait_sec must be > 0"

    @wraps(func)
    def decorator(*args: P.args, **kwargs: P.kwargs) -> T:
        exception = ""
        for _ in range(max_attempts):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                exception = str(e)
                sleep(wait_sec)
        raise Exception(exception)

    return decorator


@retry_call
def make_request(
    request: Literal["post", "get"],
    route: Union[str, Route],
    body: Optional[dict[str, Any]] = None,
    params: Optional[dict[str, Any]] = None,
    headers: Optional[dict[str, Any]] = None,
) -> dict[str, str]:
    """Makes the request"""
    headers = headers or {}
    internal_headers = {"X-Token": f"{config.ARCEE_API_KEY}", "Content-Type": "application/json"}
    headers.update(**internal_headers)
    url = f"{config.ARCEE_API_URL}/{config.ARCEE_API_VERSION}/{route}"

    req_type = getattr(requests, request)
    response = req_type(url, json=body, params=params, headers=headers)
    if response.status_code not in (200, 201):
        raise Exception(f"Failed to make request. Response: {response.text}")
    return response.json()
