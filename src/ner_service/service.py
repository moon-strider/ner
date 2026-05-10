from __future__ import annotations

import json

from pydantic import ValidationError

from ner_service.cache import ResultCache
from ner_service.config import RuntimeLimits
from ner_service.config_store import (
    ConfigStore,
    PreparedNERConfig,
    prepare_config,
    render_system_prompt,
)
from ner_service.offsets import canonicalize_entities
from ner_service.offsets_trie import attach_offsets_trie
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
    ) -> None:
        self._provider = provider
        self._default_model = default_model
        self._max_tokens = max_tokens
        self._limits = limits or RuntimeLimits()
        self._configs = ConfigStore(config_store)
        self._cache = cache

    @property
    def provider(self) -> NerProvider:
        return self._provider

    async def create_config(self, config: NERConfig) -> NERConfigRecord:
        return await self._configs.create(self._prepare_runtime_config(config))

    async def list_configs(self) -> list[NERConfigRecord]:
        return await self._configs.list()

    async def get_config(self, config_id: str) -> NERConfigRecord:
        prepared = await self._configs.get(config_id)
        return NERConfigRecord(id=config_id, config=prepared.config)

    async def put_config(self, config_id: str, config: NERConfig) -> NERConfigRecord:
        self._validate_config_id(config_id)
        return await self._configs.put(config_id, self._prepare_runtime_config(config))

    async def patch_config(self, config_id: str, patch: NERConfigPatch) -> NERConfigRecord:
        self._validate_config_id(config_id)
        current = (await self._configs.get(config_id)).config
        data = current.model_dump()
        data.update(patch.model_dump(exclude_unset=True))
        try:
            config = NERConfig.model_validate(data)
        except ValidationError as e:
            raise ValueError(str(e)) from e
        return await self._configs.put(config_id, self._prepare_runtime_config(config))

    async def delete_config(self, config_id: str) -> None:
        self._validate_config_id(config_id)
        await self._configs.delete(config_id)

    async def extract(self, request: ExtractRequest) -> ExtractResponse:
        self._validate_request(request)
        prepared = await self._resolve_config(request)
        config_key = self._cache_key(prepared)
        cached = self._cache.get(request.text, config_key) if self._cache is not None else None
        if cached is not None:
            return ExtractResponse.model_validate(cached)
        system_prompt = render_system_prompt(prepared, request.prompt_payload)
        raw = await self._provider.extract(
            request.text,
            prepared=prepared,
            system_prompt=system_prompt,
        )
        config = prepared.config
        entities = (
            attach_offsets_trie(
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
        )
        if self._cache is not None:
            self._cache.set(
                request.text,
                config_key,
                response.model_dump(mode="json"),
            )
        return response

    async def aclose(self) -> None:
        await self._provider.aclose()

    async def _resolve_config(self, request: ExtractRequest) -> PreparedNERConfig:
        if request.config_id is not None:
            self._validate_config_id(request.config_id)
            return await self._configs.get(request.config_id)
        assert request.config is not None
        return prepare_config(self._prepare_runtime_config(request.config))

    def _prepare_runtime_config(self, config: NERConfig) -> NERConfig:
        config = self._apply_runtime_defaults(config)
        self._validate_config(config)
        return config

    def _cache_key(self, prepared: PreparedNERConfig) -> str:
        payload = prepared.config.model_dump(mode="json")
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
        if request.config is not None:
            self._validate_config(request.config)

    def _validate_config(self, config: NERConfig) -> None:
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
