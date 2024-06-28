from functools import wraps
from time import sleep
from typing import Any, Callable, Dict, Literal, Optional, TypeVar, Union

import requests
from typing_extensions import ParamSpec

from arcee import __version__ as ARCEE_PY_VERSION
from arcee import config
from arcee.schemas.routes import Route

session_headers = {
    "User-Agent": f"arcee-py/{ARCEE_PY_VERSION}",
    "X-Token": f"{config.ARCEE_API_KEY}",
}
if config.ARCEE_ORG != "":
    session_headers["X-Arcee-Org"] = config.ARCEE_ORG

session = requests.Session()
session.headers.update(session_headers)

T = TypeVar("T")
P = ParamSpec("P")


def retry_call(*, max_attempts: int = 2, wait_sec: Union[float, int] = 5) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Retry api call"""
    assert wait_sec > 0, "wait_sec must be > 0"

    def retry_wrapper(func: Callable[P, T]) -> Callable[P, T]:
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

    return retry_wrapper


default_headers = {
    "Content-Type": "application/json",
}


# @retry_call()
def make_request(
    method: Literal["get", "post", "put", "patch", "delete", "head"],
    route: Union[str, Route],
    body: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """Makes the request"""
    arcee_api_url = config.ARCEE_API_URL.rstrip("/")
    url = f"{arcee_api_url}/api/{config.ARCEE_API_VERSION}/{route}"

    request_headers = {**default_headers, **headers} if headers else default_headers
    request = requests.Request(method.upper(), url, json=body, params=params)
    request.headers.update(request_headers)
    prepped = session.prepare_request(request)
    response = session.send(prepped, allow_redirects=True)

    if response.status_code not in (200, 201, 202):
        raise Exception(f"Failed to make request. Response: {response.text}")
    return response.json()


def nonjson_request(
    method: Literal["get", "post", "put", "patch", "delete", "head"],
    route: Union[str, Route],
    body: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, Any]] = None,
    stream: Optional[bool] = False,
) -> requests.Response:
    """Makes the request"""
    arcee_api_url = config.ARCEE_API_URL.rstrip("/")
    url = f"{arcee_api_url}/{config.ARCEE_API_VERSION}/{route}"

    request = requests.Request(method.upper(), url, json=body, params=params)
    if headers:
        request.headers.update(headers)
    prepped = session.prepare_request(request)
    response = session.send(prepped, allow_redirects=True, stream=stream)

    if response.status_code not in (200, 201, 202):
        raise Exception(f"Failed to make request. Response: {response.text}")
    return response
