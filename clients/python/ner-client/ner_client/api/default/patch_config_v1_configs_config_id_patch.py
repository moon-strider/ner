from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.ner_config_patch import NERConfigPatch
from ...models.ner_config_record import NERConfigRecord
from ...types import Response


def _get_kwargs(config_id: str, *, body: NERConfigPatch) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    _kwargs: dict[str, Any] = {
        "method": "patch",
        "url": "/v1/configs/{config_id}".format(config_id=quote(str(config_id), safe="")),
    }
    _kwargs["json"] = body.to_dict()
    headers["Content-Type"] = "application/json"
    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | NERConfigRecord | None:
    if response.status_code == 200:
        response_200 = NERConfigRecord.from_dict(response.json())
        return response_200
    if response.status_code == 422:
        response_422 = HTTPValidationError.from_dict(response.json())
        return response_422
    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Response[HTTPValidationError | NERConfigRecord]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    config_id: str, *, client: AuthenticatedClient | Client, body: NERConfigPatch
) -> Response[HTTPValidationError | NERConfigRecord]:
    kwargs = _get_kwargs(config_id=config_id, body=body)
    response = client.get_httpx_client().request(**kwargs)
    return _build_response(client=client, response=response)


def sync(
    config_id: str, *, client: AuthenticatedClient | Client, body: NERConfigPatch
) -> HTTPValidationError | NERConfigRecord | None:
    return sync_detailed(config_id=config_id, client=client, body=body).parsed


async def asyncio_detailed(
    config_id: str, *, client: AuthenticatedClient | Client, body: NERConfigPatch
) -> Response[HTTPValidationError | NERConfigRecord]:
    kwargs = _get_kwargs(config_id=config_id, body=body)
    response = await client.get_async_httpx_client().request(**kwargs)
    return _build_response(client=client, response=response)


async def asyncio(
    config_id: str, *, client: AuthenticatedClient | Client, body: NERConfigPatch
) -> HTTPValidationError | NERConfigRecord | None:
    return (await asyncio_detailed(config_id=config_id, client=client, body=body)).parsed
