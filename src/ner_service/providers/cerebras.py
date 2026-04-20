from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, cast

from cerebras.cloud.sdk import (
    APIConnectionError as SDKAPIConnectionError,
)
from cerebras.cloud.sdk import (
    APIError as SDKAPIError,
)
from cerebras.cloud.sdk import (
    APIStatusError as SDKAPIStatusError,
)
from cerebras.cloud.sdk import (
    AsyncCerebras,
)
from cerebras.cloud.sdk import (
    AuthenticationError as SDKAuthenticationError,
)
from cerebras.cloud.sdk import (
    BadRequestError as SDKBadRequestError,
)
from cerebras.cloud.sdk import (
    PermissionDeniedError as SDKPermissionDeniedError,
)
from cerebras.cloud.sdk import (
    RateLimitError as SDKRateLimitError,
)
from pydantic import ValidationError

from ner_service.providers.base import (
    ProviderAuthError,
    ProviderBadRequestError,
    ProviderError,
    ProviderPermissionError,
    ProviderQuotaError,
    ProviderRateLimitError,
    ProviderUpstreamError,
)
from ner_service.schema_builder import build_response_format, build_system_prompt
from ner_service.schemas import EntityLabel, RawEntities, RawEntity


class CerebrasProvider:
    name: str = "cerebras"

    def __init__(
        self,
        api_key: str,
        model: str = "llama3.1-8b",
        timeout: float = 30.0,
        max_retries: int = 2,
    ) -> None:
        self.model = model
        self._client = AsyncCerebras(api_key=api_key, timeout=timeout, max_retries=max_retries)

    async def extract(
        self,
        text: str,
        labels: list[EntityLabel],
        *,
        require_offsets: bool,
        retries: int,
        max_tokens: int,
        reasoning_effort: str | None = None,
    ) -> RawEntities:
        system_prompt = build_system_prompt(labels, require_offsets=require_offsets)
        response_format = build_response_format(labels)
        allowed_labels = {label.name for label in labels}
        usage_total: dict[str, Any] = {}
        last_output: str | None = None
        last_error: str | None = None

        for attempt in range(1, retries + 1):
            try:
                completion = await self._create_completion(
                    text=text,
                    system_prompt=system_prompt,
                    response_format=response_format,
                    max_tokens=max_tokens,
                    reasoning_effort=reasoning_effort,
                    last_output=last_output,
                    last_error=last_error,
                )
            except SDKAuthenticationError as e:
                raise ProviderAuthError(str(e), details=_error_details(e)) from e
            except SDKRateLimitError as e:
                raise ProviderRateLimitError(
                    "provider rate limit exceeded",
                    details=_error_details(e),
                    headers=_error_headers(e),
                ) from e
            except SDKBadRequestError as e:
                raise ProviderBadRequestError(str(e), details=_error_details(e)) from e
            except SDKPermissionDeniedError as e:
                raise ProviderPermissionError(str(e), details=_error_details(e)) from e
            except SDKAPIStatusError as e:
                if e.status_code == 402:
                    raise ProviderQuotaError(
                        "provider credits or billing quota exhausted",
                        details=_error_details(e),
                    ) from e
                if e.status_code >= 500:
                    raise ProviderUpstreamError(str(e), details=_error_details(e)) from e
                raise ProviderBadRequestError(str(e), details=_error_details(e)) from e
            except SDKAPIConnectionError as e:
                raise ProviderUpstreamError(
                    f"connection error: {e}",
                    details=_error_details(e),
                ) from e
            except SDKAPIError as e:
                raise ProviderUpstreamError(str(e), details=_error_details(e)) from e

            completion_obj = cast(Any, completion)
            _merge_usage(usage_total, _extract_usage(completion_obj))
            content = completion_obj.choices[0].message.content
            last_output = content

            try:
                entities = _parse_raw_entities(content, allowed_labels)
            except ProviderError as e:
                last_error = str(e)
                if attempt == retries:
                    details = {"attempts": attempt, "last_error": last_error}
                    if usage_total:
                        details["usage"] = usage_total
                    raise ProviderError(
                        "model returned invalid structured output",
                        details=details,
                    ) from e
                continue

            return RawEntities(entities=entities, usage=usage_total or None)

        raise ProviderError("model returned invalid structured output")

    async def _create_completion(
        self,
        *,
        text: str,
        system_prompt: str,
        response_format: dict[str, Any],
        max_tokens: int,
        reasoning_effort: str | None,
        last_output: str | None,
        last_error: str | None,
    ) -> Any:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": text}]
        if last_error is not None:
            messages.extend(
                [
                    {"role": "assistant", "content": last_output or ""},
                    {
                        "role": "user",
                        "content": (
                            "The previous output was invalid. "
                            "Return a corrected JSON object only.\n"
                            f"Error: {last_error}"
                        ),
                    },
                ]
            )

        params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "response_format": response_format,
            "temperature": 0,
            "max_completion_tokens": max_tokens,
        }
        if reasoning_effort is not None:
            params["reasoning_effort"] = reasoning_effort

        return await self._client.chat.completions.create(**params)

    async def aclose(self) -> None:
        close = getattr(self._client, "close", None)
        if close is None:
            return
        result = close()
        if hasattr(result, "__await__"):
            await result


def _parse_raw_entities(content: str | None, allowed_labels: set[str]) -> list[RawEntity]:
    if not content:
        raise ProviderError("empty completion content")

    try:
        payload = json.loads(content)
    except json.JSONDecodeError as e:
        raise ProviderError(f"invalid JSON from model: {e}") from e

    try:
        raw = RawEntities.model_validate(payload)
    except ValidationError as e:
        raise ProviderError(f"schema mismatch from model: {e}") from e

    invalid_labels = sorted(
        {entity.label for entity in raw.entities if entity.label not in allowed_labels}
    )
    if invalid_labels:
        raise ProviderError(f"schema mismatch from model: unsupported labels {invalid_labels}")
    return raw.entities


def _error_details(error: SDKAPIError) -> dict[str, Any]:
    details: dict[str, Any] = {"message": str(error)}
    status_code = getattr(error, "status_code", None)
    if isinstance(status_code, int):
        details["status_code"] = status_code
    body = getattr(error, "body", None)
    if body is not None:
        details["body"] = _json_safe(body)
    headers = _error_headers(error)
    if headers:
        details["headers"] = headers
    return details


def _error_headers(error: SDKAPIError) -> dict[str, str]:
    response = getattr(error, "response", None)
    headers_obj = getattr(response, "headers", None)
    if headers_obj is None:
        return {}
    headers: dict[str, str] = {}
    for key, value in headers_obj.items():
        lower = key.lower()
        if lower == "retry-after" or lower.startswith("x-ratelimit-"):
            headers[key] = value
    return headers


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def _merge_usage(total: dict[str, Any], usage: dict[str, Any] | None) -> None:
    if usage is None:
        return
    for key, value in usage.items():
        if isinstance(value, (int, float)):
            total[key] = total.get(key, 0) + value
        elif isinstance(value, dict):
            existing = total.setdefault(key, {})
            if isinstance(existing, dict):
                _merge_usage(existing, value)
        elif key not in total:
            total[key] = value


def _extract_usage(completion: Any) -> dict[str, Any] | None:
    usage = getattr(completion, "usage", None)
    if usage is None:
        return None
    dump = getattr(usage, "model_dump", None)
    if callable(dump):
        return dict(dump())
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }
