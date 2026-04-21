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


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    cerebras_api_key: SecretStr | None = None
    ner_provider: str = Field(default="cerebras")
    ner_model: str = Field(default="llama3.1-8b")
    request_timeout_s: float = Field(default=30.0, gt=0.0)
    transport_retries: int = Field(default=2, ge=0)
    max_tokens: int = Field(default=1024, gt=0)
    max_text_length: int = Field(default=32_000, gt=0)
    max_labels: int = Field(default=50, gt=0)
    max_system_prompt_length: int = Field(default=20_000, gt=0)
    max_label_description_length: int = Field(default=500, gt=0)
    max_config_id_length: int = Field(default=128, gt=0)

    def runtime_limits(self) -> RuntimeLimits:
        return RuntimeLimits(
            max_text_length=self.max_text_length,
            max_labels=self.max_labels,
            max_system_prompt_length=self.max_system_prompt_length,
            max_label_description_length=self.max_label_description_length,
            max_config_id_length=self.max_config_id_length,
        )


def get_settings() -> Settings:
    return Settings()
