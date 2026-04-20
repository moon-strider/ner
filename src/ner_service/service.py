from __future__ import annotations

from ner_service.config_store import (
    ConfigStore,
    PreparedNERConfig,
    prepare_config,
    render_system_prompt,
)
from ner_service.offsets import attach_offsets, canonicalize_entities
from ner_service.providers.base import NerProvider
from ner_service.schemas import (
    ExtractRequest,
    ExtractResponse,
    NERConfig,
    NERConfigPatch,
    NERConfigRecord,
)


class NerService:
    def __init__(
        self,
        provider: NerProvider,
        *,
        default_model: str = "llama3.1-8b",
        max_tokens: int = 1024,
    ) -> None:
        self._provider = provider
        self._default_model = default_model
        self._max_tokens = max_tokens
        self._configs = ConfigStore()

    @property
    def provider(self) -> NerProvider:
        return self._provider

    def create_config(self, config: NERConfig) -> NERConfigRecord:
        return self._configs.create(self._apply_runtime_defaults(config))

    def list_configs(self) -> list[NERConfigRecord]:
        return self._configs.list()

    def get_config(self, config_id: str) -> NERConfigRecord:
        prepared = self._configs.get(config_id)
        return NERConfigRecord(id=config_id, config=prepared.config)

    def put_config(self, config_id: str, config: NERConfig) -> NERConfigRecord:
        return self._configs.put(config_id, self._apply_runtime_defaults(config))

    def patch_config(self, config_id: str, patch: NERConfigPatch) -> NERConfigRecord:
        return self._configs.patch(config_id, patch)

    def delete_config(self, config_id: str) -> None:
        self._configs.delete(config_id)

    async def extract(self, request: ExtractRequest) -> ExtractResponse:
        prepared = self._resolve_config(request)
        system_prompt = render_system_prompt(prepared, request.prompt_payload)
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
        return ExtractResponse(
            entities=entities,
            model=config.model,
            provider=self._provider.name,
            usage=raw.usage,
        )

    async def aclose(self) -> None:
        await self._provider.aclose()

    def _resolve_config(self, request: ExtractRequest) -> PreparedNERConfig:
        if request.config_id is not None:
            return self._configs.get(request.config_id)
        assert request.config is not None
        return prepare_config(self._apply_runtime_defaults(request.config))

    def _apply_runtime_defaults(self, config: NERConfig) -> NERConfig:
        updates: dict[str, object] = {}
        if "model" not in config.model_fields_set:
            updates["model"] = self._default_model
        if "max_tokens" not in config.model_fields_set:
            updates["max_tokens"] = self._max_tokens
        if not updates:
            return config
        return config.model_copy(update=updates)
