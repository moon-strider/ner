from __future__ import annotations

import json
from collections.abc import Mapping
from functools import partial
from typing import Any, cast

import httpx
from pydantic import ValidationError

from ner_service.circuit_breaker import CircuitBreaker, CircuitBreakerOpen
from ner_service.config_store import PreparedNERConfig
from ner_service.metrics import MetricsCollector
from ner_service.providers.base import (
    ProviderAuthError,
    ProviderBadRequestError,
    ProviderError,
    ProviderPermissionError,
    ProviderQuotaError,
    ProviderRateLimitError,
    ProviderUpstreamError,
)
from ner_service.rate_limiter import RateLimiter
from ner_service.schemas import RawEntities, RawEntity


def _build_messages(
    system_prompt: str,
    text: str,
    last_output: str | None,
    last_error: str | None,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]
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
    return messages


def _build_request_body(
    model: str,
    messages: list[dict[str, Any]],
    response_format: dict[str, Any],
    max_tokens: int,
    reasoning_effort: str | None,
    temperature: float,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "response_format": response_format,
        "temperature": temperature,
        "max_completion_tokens": max_tokens,
    }
    if reasoning_effort is not None:
        body["reasoning_effort"] = reasoning_effort
    return body


def _extract_usage(completion: dict[str, Any]) -> dict[str, Any] | None:
    usage = completion.get("usage")
    if usage is None:
        return None
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }


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


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(v) for v in value]
    return str(value)


def _raise_provider_error_from_response(
    status_code: int,
    body: dict[str, Any],
    headers: dict[str, str] | None = None,
) -> None:
    error = body.get("error", {})
    message = error.get("message", "provider error") if isinstance(error, dict) else str(body)
    details = {"status_code": status_code, "body": _json_safe(body)}
    rate_limit_headers = {}
    if headers:
        for key, value in headers.items():
            lower = key.lower()
            if lower == "retry-after" or lower.startswith("x-ratelimit-"):
                rate_limit_headers[key] = value
    if status_code == 401:
        raise ProviderAuthError(message, details=details)
    if status_code == 429:
        raise ProviderRateLimitError(
            "provider rate limit exceeded", details=details, headers=rate_limit_headers or None
        )
    if status_code == 402:
        raise ProviderQuotaError("provider credits or billing quota exhausted", details=details)
    if status_code == 403:
        raise ProviderPermissionError(message, details=details)
    if status_code == 400:
        raise ProviderBadRequestError(message, details=details)
    if status_code >= 500:
        raise ProviderUpstreamError(message, details=details)
    raise ProviderError(message, details=details)


def _should_count_for_circuit_breaker(exc: Exception) -> bool:
    return isinstance(exc, ProviderUpstreamError)


class OpenAICompatibleProvider:
    name: str = "openai_compatible"

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout: float = 30.0,
        max_retries: int = 2,
        provider_name: str = "openai_compatible",
        circuit_breaker: CircuitBreaker | None = None,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        self.model = model
        self.name = provider_name
        self._timeout = timeout
        self._max_retries = max_retries
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._client: httpx.AsyncClient | None = None
        self._circuit = circuit_breaker or CircuitBreaker()
        self._rate_limiter = rate_limiter

    @property
    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            limits = httpx.Limits(max_connections=100, max_keepalive_connections=20)
            transport = httpx.AsyncHTTPTransport(retries=self._max_retries, limits=limits)
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._timeout),
                transport=transport,
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
        return self._client

    async def extract(
        self,
        text: str,
        *,
        prepared: PreparedNERConfig,
        system_prompt: str,
    ) -> RawEntities:
        config = prepared.config
        usage_total: dict[str, Any] = {}
        last_output: str | None = None
        last_error: str | None = None

        for attempt in range(1, config.retries + 1):
            messages = _build_messages(system_prompt, text, last_output, last_error)
            body = _build_request_body(
                model=config.model,
                messages=messages,
                response_format=prepared.response_format,
                max_tokens=config.max_tokens,
                reasoning_effort=config.reasoning_effort,
                temperature=0.0,
            )
            request = partial(self._post_with_status_check, body, config.model)

            try:
                response = await self._circuit.call(
                    request,
                    is_failure=_should_count_for_circuit_breaker,
                )
            except CircuitBreakerOpen as e:
                MetricsCollector().record_circuit_breaker(
                    provider=self.name,
                    model=config.model,
                    state="open",
                )
                raise ProviderUpstreamError(
                    "circuit breaker open",
                    details={
                        "status_code": 503,
                        "circuit_breaker": {
                            "state": "open",
                            "reason": str(e),
                        },
                    },
                ) from e

            if response.status_code >= 400:
                try:
                    err_body = response.json()
                except Exception:
                    err_body = {"raw": response.text}
                _raise_provider_error_from_response(
                    response.status_code,
                    err_body,
                    dict(response.headers),
                )

            try:
                completion = response.json()
            except ValueError as e:
                raise ProviderError(f"invalid JSON from provider response: {e}") from e
            if not isinstance(completion, dict):
                raise ProviderError("provider response must be a JSON object")
            usage_total = _merge_usage(usage_total, _extract_usage(completion))
            choices = completion.get("choices", [])
            if not choices:
                last_error = "empty choices array"
                if attempt == config.retries:
                    raise ProviderError("model returned empty choices")
                continue

            content = cast(str, choices[0].get("message", {}).get("content") or "")
            last_output = content

            try:
                entities = _parse_raw_entities(content, prepared.allowed_labels)
            except ProviderError as e:
                last_error = str(e)
                if attempt == config.retries:
                    details = {"attempts": attempt, "last_error": last_error}
                    if usage_total:
                        details["usage"] = usage_total
                    raise ProviderError(
                        "model returned invalid structured output",
                        details=details,
                    ) from e
                continue

            return RawEntities(entities=entities, usage=usage_total or None, attempts=attempt)

        raise ProviderError("model returned invalid structured output")

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _post_with_status_check(self, body: dict[str, Any], model: str) -> httpx.Response:
        if self._rate_limiter is None:
            return await self._do_post_with_status_check(body, model)
        await self._rate_limiter.acquire(self.name)
        try:
            return await self._do_post_with_status_check(body, model)
        finally:
            self._rate_limiter.release(self.name)

    async def _do_post_with_status_check(self, body: dict[str, Any], model: str) -> httpx.Response:
        try:
            response = await self._http.post(
                f"{self._base_url}/chat/completions",
                json=body,
            )
        except httpx.TimeoutException as e:
            raise ProviderUpstreamError(f"upstream timeout: {e}") from e
        except httpx.ConnectError as e:
            raise ProviderUpstreamError(f"connection error: {e}") from e
        except httpx.HTTPError as e:
            raise ProviderUpstreamError(f"upstream transport error: {e}") from e

        if response.status_code >= 500:
            try:
                err_body = response.json()
            except Exception:
                err_body = {"raw": response.text}
            MetricsCollector().record_circuit_breaker(
                provider=self.name,
                model=model,
                state="failure",
            )
            _raise_provider_error_from_response(
                response.status_code,
                err_body,
                dict(response.headers),
            )
        return response


def _merge_usage(total: dict[str, Any], usage: dict[str, Any] | None) -> dict[str, Any]:
    if usage is None:
        return total
    result = dict(total)
    for key, value in usage.items():
        if isinstance(value, int | float):
            result[key] = result.get(key, 0) + value
        elif isinstance(value, dict):
            existing = result.setdefault(key, {})
            if isinstance(existing, dict):
                _merge_usage(existing, value)
        elif key not in result:
            result[key] = value
    return result
