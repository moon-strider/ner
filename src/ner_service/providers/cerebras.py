from __future__ import annotations

import json
from typing import Any, cast

from cerebras.cloud.sdk import (
    APIConnectionError as SDKAPIConnectionError,
)
from cerebras.cloud.sdk import (
    APIError as SDKAPIError,
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
    RateLimitError as SDKRateLimitError,
)

from ner_service.providers.base import (
    ProviderAuthError,
    ProviderBadRequestError,
    ProviderError,
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

    async def extract(self, text: str, labels: list[EntityLabel]) -> RawEntities:
        system_prompt = build_system_prompt(labels)
        response_format = build_response_format(labels)

        try:
            completion = await self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
                response_format=response_format,
                temperature=0,
            )
        except SDKAuthenticationError as e:
            raise ProviderAuthError(str(e)) from e
        except SDKRateLimitError as e:
            raise ProviderRateLimitError(str(e)) from e
        except SDKBadRequestError as e:
            raise ProviderBadRequestError(str(e)) from e
        except SDKAPIConnectionError as e:
            raise ProviderUpstreamError(f"connection error: {e}") from e
        except SDKAPIError as e:
            raise ProviderUpstreamError(str(e)) from e

        completion_obj = cast(Any, completion)
        content = completion_obj.choices[0].message.content
        if not content:
            raise ProviderError("empty completion content")

        try:
            payload = json.loads(content)
        except json.JSONDecodeError as e:
            raise ProviderError(f"invalid JSON from model: {e}") from e

        usage = _extract_usage(completion_obj)
        entities = [RawEntity.model_validate(item) for item in payload.get("entities", [])]
        return RawEntities(entities=entities, usage=usage)

    async def aclose(self) -> None:
        close = getattr(self._client, "close", None)
        if close is None:
            return
        result = close()
        if hasattr(result, "__await__"):
            await result


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
