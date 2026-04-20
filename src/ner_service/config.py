from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    max_text_length: int = Field(default=32_000, gt=0)
    max_labels: int = Field(default=50, gt=0)


def get_settings() -> Settings:
    return Settings()
