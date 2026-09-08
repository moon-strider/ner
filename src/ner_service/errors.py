"""Public errors deliberately exclude provider bodies, prompts, and exception internals."""

from __future__ import annotations

from typing import Any

from ner_service.config_store import ConfigNotFoundError, PromptTemplateError
from ner_service.providers.base import (
    ProviderAuthError,
    ProviderBadRequestError,
    ProviderError,
    ProviderPermissionError,
    ProviderQuotaError,
    ProviderRateLimitError,
    ProviderUpstreamError,
)

_PROVIDER_ERRORS = {
    ProviderAuthError: (502, "provider_auth_failed", "provider authentication failed"),
    ProviderRateLimitError: (429, "provider_rate_limited", "provider rate limit exceeded"),
    ProviderQuotaError: (
        402,
        "provider_quota_exhausted",
        "provider credits or billing quota exhausted",
    ),
    ProviderPermissionError: (403, "provider_permission_denied", "provider permission denied"),
    ProviderBadRequestError: (400, "provider_bad_request", "provider rejected the request"),
    ProviderUpstreamError: (502, "provider_upstream_error", "provider is unavailable"),
}


def public_error(exc: Exception) -> tuple[int, str, str]:
    for kind, result in _PROVIDER_ERRORS.items():
        if isinstance(exc, kind):
            return result
    if isinstance(exc, ProviderError):
        return 502, "provider_error", "provider returned invalid structured output"
    if isinstance(exc, ConfigNotFoundError):
        return 404, "config_not_found", "config not found"
    if isinstance(exc, PromptTemplateError):
        return 422, "prompt_template_error", str(exc)
    if isinstance(exc, ValueError):
        return 422, "invalid_request", str(exc)
    return 500, "internal_error", "internal server error"


def provider_details(exc: ProviderError) -> dict[str, Any]:
    return {
        key: value
        for key in ("status_code", "attempts")
        if type(value := exc.details.get(key)) is int
    }


def provider_headers(exc: ProviderError) -> dict[str, str]:
    return {
        key.lower(): value
        for key, value in exc.headers.items()
        if (key.lower() == "retry-after" or key.lower().startswith("x-ratelimit-"))
        and value.isascii()
        and all(32 <= ord(char) < 127 for char in value)
    }
