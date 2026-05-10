from __future__ import annotations

import json
from dataclasses import dataclass

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


@dataclass(frozen=True)
class RuntimeLimits:
    max_text_length: int = 32_000
    max_labels: int = 50
    max_system_prompt_length: int = 20_000
    max_label_description_length: int = 500
    max_config_id_length: int = 128


@dataclass(frozen=True)
class TokenPricing:
    input_per_million: float
    output_per_million: float


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    ner_provider: str = Field(default="cerebras")
    ner_model: str = Field(default="llama3.1-8b")
    request_timeout_s: float = Field(default=30.0, gt=0.0)
    transport_retries: int = Field(default=2, ge=0)

    cerebras_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    openrouter_api_key: SecretStr | None = None
    vllm_api_key: str = Field(default="not-needed")

    cerebras_base_url: str | None = None
    openai_base_url: str | None = None
    openrouter_base_url: str | None = None
    vllm_base_url: str | None = None

    max_tokens: int = Field(default=1024, gt=0)
    otel_endpoint: str | None = None
    rate_limit_rps: float = Field(default=100.0, gt=0.0)
    rate_limit_burst: int = Field(default=200, gt=0)
    provider_concurrency_limit: int = Field(default=50, gt=0)
    max_text_length: int = Field(default=32_000, gt=0)
    max_labels: int = Field(default=50, gt=0)
    max_system_prompt_length: int = Field(default=20_000, gt=0)
    max_label_description_length: int = Field(default=500, gt=0)
    max_config_id_length: int = Field(default=128, gt=0)
    config_db_path: str = Field(default="configs.db", min_length=1)
    cache_enabled: bool = True
    cache_ttl_seconds: int = Field(default=600, gt=0)
    cache_max_size: int = Field(default=10_000, gt=0)
    circuit_breaker_failure_threshold: int = Field(default=5, gt=0)
    circuit_breaker_recovery_timeout_s: float = Field(default=30.0, gt=0.0)
    circuit_breaker_half_open_max_calls: int = Field(default=1, gt=0)
    batch_concurrency: int = Field(default=10, gt=0)
    token_pricing_json: str | None = None

    def runtime_limits(self) -> RuntimeLimits:
        return RuntimeLimits(
            max_text_length=self.max_text_length,
            max_labels=self.max_labels,
            max_system_prompt_length=self.max_system_prompt_length,
            max_label_description_length=self.max_label_description_length,
            max_config_id_length=self.max_config_id_length,
        )

    def token_pricing(self) -> dict[str, TokenPricing]:
        if not self.token_pricing_json:
            return {}
        try:
            raw = json.loads(self.token_pricing_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"TOKEN_PRICING_JSON must be valid JSON: {e}") from e
        if not isinstance(raw, dict):
            raise ValueError("TOKEN_PRICING_JSON must be a JSON object")
        pricing: dict[str, TokenPricing] = {}
        for model, value in raw.items():
            if not isinstance(model, str) or not isinstance(value, dict):
                raise ValueError("TOKEN_PRICING_JSON entries must map model names to objects")
            input_price = value.get("input_per_million")
            output_price = value.get("output_per_million")
            if not isinstance(input_price, int | float) or not isinstance(
                output_price, int | float
            ):
                raise ValueError(
                    "pricing entries require numeric input_per_million and output_per_million"
                )
            if input_price < 0 or output_price < 0:
                raise ValueError("pricing values must be >= 0")
            pricing[model] = TokenPricing(float(input_price), float(output_price))
        return pricing


def get_settings() -> Settings:
    return Settings()
