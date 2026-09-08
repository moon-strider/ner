from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error_envelope import ErrorEnvelope
from ...models.ner_config_patch import NERConfigPatch
from ...models.ner_config_record import NERConfigRecord
from ...types import Response


def _get_kwargs(
    config_id: str,
    *,
    body: NERConfigPatch,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "patch",
        "url": "/v1/configs/{config_id}".format(
            config_id=quote(str(config_id), safe=""),
        ),
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> ErrorEnvelope | NERConfigRecord | None:
    if response.status_code == 200:
        response_200 = NERConfigRecord.from_dict(response.json())

        return response_200

    if response.status_code == 400:
        response_400 = ErrorEnvelope.from_dict(response.json())

        return response_400

    if response.status_code == 401:
        response_401 = ErrorEnvelope.from_dict(response.json())

        return response_401

    if response.status_code == 402:
        response_402 = ErrorEnvelope.from_dict(response.json())

        return response_402

    if response.status_code == 403:
        response_403 = ErrorEnvelope.from_dict(response.json())

        return response_403

    if response.status_code == 404:
        response_404 = ErrorEnvelope.from_dict(response.json())

        return response_404

    if response.status_code == 413:
        response_413 = ErrorEnvelope.from_dict(response.json())

        return response_413

    if response.status_code == 422:
        response_422 = ErrorEnvelope.from_dict(response.json())

        return response_422

    if response.status_code == 429:
        response_429 = ErrorEnvelope.from_dict(response.json())

        return response_429

    if response.status_code == 500:
        response_500 = ErrorEnvelope.from_dict(response.json())

        return response_500

    if response.status_code == 502:
        response_502 = ErrorEnvelope.from_dict(response.json())

        return response_502

    if response.status_code == 503:
        response_503 = ErrorEnvelope.from_dict(response.json())

        return response_503

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Response[ErrorEnvelope | NERConfigRecord]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    config_id: str,
    *,
    client: AuthenticatedClient,
    body: NERConfigPatch,
) -> Response[ErrorEnvelope | NERConfigRecord]:
    """Patch Config

    Args:
        config_id (str):
        body (NERConfigPatch):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ErrorEnvelope | NERConfigRecord]
    """

    kwargs = _get_kwargs(
        config_id=config_id,
        body=body,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    config_id: str,
    *,
    client: AuthenticatedClient,
    body: NERConfigPatch,
) -> ErrorEnvelope | NERConfigRecord | None:
    """Patch Config

    Args:
        config_id (str):
        body (NERConfigPatch):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ErrorEnvelope | NERConfigRecord
    """

    return sync_detailed(
        config_id=config_id,
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    config_id: str,
    *,
    client: AuthenticatedClient,
    body: NERConfigPatch,
) -> Response[ErrorEnvelope | NERConfigRecord]:
    """Patch Config

    Args:
        config_id (str):
        body (NERConfigPatch):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ErrorEnvelope | NERConfigRecord]
    """

    kwargs = _get_kwargs(
        config_id=config_id,
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    config_id: str,
    *,
    client: AuthenticatedClient,
    body: NERConfigPatch,
) -> ErrorEnvelope | NERConfigRecord | None:
    """Patch Config

    Args:
        config_id (str):
        body (NERConfigPatch):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ErrorEnvelope | NERConfigRecord
    """

    return (
        await asyncio_detailed(
            config_id=config_id,
            client=client,
            body=body,
        )
    ).parsed
