from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.response_health_v1_health_get import ResponseHealthV1HealthGet
from ...types import Response


def _get_kwargs() -> dict[str, Any]:
    _kwargs: dict[str, Any] = {"method": "get", "url": "/v1/health"}
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> ResponseHealthV1HealthGet | None:
    if response.status_code == 200:
        response_200 = ResponseHealthV1HealthGet.from_dict(response.json())
        return response_200
    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Response[ResponseHealthV1HealthGet]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(*, client: AuthenticatedClient | Client) -> Response[ResponseHealthV1HealthGet]:
    kwargs = _get_kwargs()
    response = client.get_httpx_client().request(**kwargs)
    return _build_response(client=client, response=response)


def sync(*, client: AuthenticatedClient | Client) -> ResponseHealthV1HealthGet | None:
    return sync_detailed(client=client).parsed


async def asyncio_detailed(
    *, client: AuthenticatedClient | Client
) -> Response[ResponseHealthV1HealthGet]:
    kwargs = _get_kwargs()
    response = await client.get_async_httpx_client().request(**kwargs)
    return _build_response(client=client, response=response)


async def asyncio(*, client: AuthenticatedClient | Client) -> ResponseHealthV1HealthGet | None:
    return (await asyncio_detailed(client=client)).parsed
