from __future__ import annotations

import asyncio
import json
import time

from pydantic import ValidationError

from ner_service.cache import ResultCache
from ner_service.config import RuntimeLimits, TokenPricing
from ner_service.config_store import (
    ConfigStore,
    PreparedNERConfig,
    prepare_config,
    render_system_prompt,
)
from ner_service.metrics import MetricsCollector
from ner_service.offsets import attach_offsets, canonicalize_entities
from ner_service.providers.base import NerProvider
from ner_service.schemas import (
    ExtractRequest,
    ExtractResponse,
    NERConfig,
    NERConfigPatch,
    NERConfigRecord,
)
from ner_service.stores import ConfigStoreBackend


class NerService:
    def __init__(
        self,
        provider: NerProvider,
        *,
        default_model: str = "llama3.1-8b",
        max_tokens: int = 1024,
        limits: RuntimeLimits | None = None,
        cache: ResultCache | None = None,
        config_store: ConfigStoreBackend | None = None,
        token_pricing: dict[str, TokenPricing] | None = None,
    ) -> None:
        self._provider = provider
        self._default_model = default_model
        self._max_tokens = max_tokens
        self._limits = limits or RuntimeLimits()
        self._configs = ConfigStore(config_store)
        self._config_lock = asyncio.Lock()
        self._cache = cache
        self._token_pricing = token_pricing or {}
        self._metrics = MetricsCollector()

    @property
    def provider(self) -> NerProvider:
        return self._provider

    async def create_config(self, config: NERConfig) -> NERConfigRecord:
        return await self._configs.create(self._prepare_runtime_config(config))

    async def list_configs(self) -> list[NERConfigRecord]:
        return await self._configs.list()

    async def get_config(self, config_id: str) -> NERConfigRecord:
        self._validate_config_id(config_id)
        prepared = await self._configs.get(config_id)
        return NERConfigRecord(id=config_id, config=prepared.config)

    async def put_config(self, config_id: str, config: NERConfig) -> NERConfigRecord:
        async with self._config_lock:
            self._validate_config_id(config_id)
            return await self._configs.put(config_id, self._prepare_runtime_config(config))

    async def patch_config(self, config_id: str, patch: NERConfigPatch) -> NERConfigRecord:
        async with self._config_lock:
            self._validate_config_id(config_id)
            current = (await self._configs.get(config_id)).config
            data = current.model_dump()
            data.update(patch.model_dump(exclude_unset=True))
            try:
                config = NERConfig.model_validate(data)
            except ValidationError as e:
                raise ValueError("patch contains invalid config values") from e
            return await self._configs.put(config_id, self._prepare_runtime_config(config))

    async def delete_config(self, config_id: str) -> None:
        async with self._config_lock:
            self._validate_config_id(config_id)
            await self._configs.delete(config_id)

    async def ready(self) -> dict[str, object]:
        return {
            "status": "ready",
            "provider": self._provider.name,
            "model": self._provider.model,
            "config_store": await self._configs.healthcheck(),
        }

    async def extract(self, request: ExtractRequest) -> ExtractResponse:
        started = time.perf_counter()
        model = self._default_model
        try:
            response = await self._extract(request)
            model = response.model
        except Exception as exc:
            self._metrics.record_attempt(
                self._provider.name, model, (time.perf_counter() - started) * 1000, False
            )
            self._metrics.record_error(self._provider.name, type(exc).__name__)
            raise
        self._metrics.record_attempt(
            self._provider.name, model, (time.perf_counter() - started) * 1000, True
        )
        return response

    async def _extract(self, request: ExtractRequest) -> ExtractResponse:
        self._validate_request(request)
        prepared = await self._resolve_config(request)
        system_prompt = render_system_prompt(prepared, request.prompt_payload)
        if len(system_prompt) > self._limits.max_rendered_prompt_length:
            raise ValueError("rendered system prompt exceeds configured limit")
        config_key = self._cache_key(prepared, system_prompt)
        cached = self._cache.get(request.text, config_key) if self._cache is not None else None
        if cached is not None:
            self._metrics.record_cache("hit")
            response = ExtractResponse.model_validate(cached)
            return response.model_copy(update={"usage": None, "attempts": 0, "cache_hit": True})
        self._metrics.record_cache("miss")
        raw = await self._provider.extract(
            request.text,
            prepared=prepared,
            system_prompt=system_prompt,
        )
        config = prepared.config
        entities = (
            attach_offsets(
                request.text,
                raw.entities,
                case_sensitive=config.case_sensitive,
            )
            if config.require_offsets
            else canonicalize_entities(
                request.text,
                raw.entities,
                case_sensitive=config.case_sensitive,
            )
        )
        response = ExtractResponse(
            entities=entities,
            model=config.model,
            provider=self._provider.name,
            usage=raw.usage,
            attempts=raw.attempts,
            warnings=(
                [
                    f"Dropped {len(raw.entities) - len(entities)} unmatched, duplicate, "
                    "or overlapping entities."
                ]
                if len(entities) < len(raw.entities)
                else []
            ),
        )
        self._metrics.record_structured_output_retries(
            provider=self._provider.name,
            model=config.model,
            retries=raw.attempts - 1,
        )
        self._metrics.record_tokens(self._provider.name, config.model, raw.usage)
        self._record_estimated_cost(config.model, raw.usage)
        if self._cache is not None:
            self._cache.set(
                request.text,
                config_key,
                response.model_dump(mode="json"),
            )
        return response

    async def aclose(self) -> None:
        try:
            await self._provider.aclose()
        finally:
            await self._configs.aclose()

    async def _resolve_config(self, request: ExtractRequest) -> PreparedNERConfig:
        if request.config_id is not None:
            self._validate_config_id(request.config_id)
            prepared = await self._configs.get(request.config_id)
            self._validate_config(prepared.config)
            return prepared
        assert request.config is not None
        return prepare_config(self._prepare_runtime_config(request.config))

    def _prepare_runtime_config(self, config: NERConfig) -> NERConfig:
        config = self._apply_runtime_defaults(config)
        self._validate_config(config)
        return config

    def _cache_key(self, prepared: PreparedNERConfig, system_prompt: str) -> str:
        payload = {
            "provider": self._provider.name,
            "config": prepared.config.model_dump(mode="json"),
            "prompt": system_prompt,
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def _apply_runtime_defaults(self, config: NERConfig) -> NERConfig:
        updates: dict[str, object] = {}
        if "model" not in config.model_fields_set:
            updates["model"] = self._default_model
        if "max_tokens" not in config.model_fields_set:
            updates["max_tokens"] = self._max_tokens
        if not updates:
            return config
        return config.model_copy(update=updates)

    def _validate_request(self, request: ExtractRequest) -> None:
        if len(request.text) > self._limits.max_text_length:
            raise ValueError(f"text length must be <= {self._limits.max_text_length}")
        if request.config_id is not None:
            self._validate_config_id(request.config_id)

    def _validate_config(self, config: NERConfig) -> None:
        if self._limits.allowed_models and config.model not in self._limits.allowed_models:
            raise ValueError("model is not in ALLOWED_MODELS")
        if config.retries > self._limits.max_attempts:
            raise ValueError(f"retries must be <= {self._limits.max_attempts}")
        if config.max_tokens > self._limits.max_output_tokens:
            raise ValueError(f"max_tokens must be <= {self._limits.max_output_tokens}")
        if len(config.few_shot_examples) > self._limits.max_few_shot_examples:
            raise ValueError("too many few-shot examples")
        for example in config.few_shot_examples:
            if len(example.text) > self._limits.max_text_length:
                raise ValueError("few-shot text exceeds configured limit")
        if len(config.labels) > self._limits.max_labels:
            raise ValueError(f"labels length must be <= {self._limits.max_labels}")
        for label in config.labels:
            if len(label.description) > self._limits.max_label_description_length:
                raise ValueError(
                    "label description length must be "
                    f"<= {self._limits.max_label_description_length}"
                )
        if (
            config.system_prompt is not None
            and len(config.system_prompt) > self._limits.max_system_prompt_length
        ):
            raise ValueError(
                f"system_prompt length must be <= {self._limits.max_system_prompt_length}"
            )

    def _validate_config_id(self, config_id: str) -> None:
        if len(config_id) > self._limits.max_config_id_length:
            raise ValueError(f"config_id length must be <= {self._limits.max_config_id_length}")

    def _record_estimated_cost(self, model: str, usage: dict[str, object] | None) -> None:
        if not usage:
            return
        pricing = self._token_pricing.get(model)
        if pricing is None:
            return
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        prompt = prompt_tokens if isinstance(prompt_tokens, int | float) else 0
        completion = completion_tokens if isinstance(completion_tokens, int | float) else 0
        cost = (
            prompt * pricing.input_per_million + completion * pricing.output_per_million
        ) / 1_000_000
        self._metrics.record_estimated_cost(self._provider.name, model, cost)
