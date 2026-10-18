from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Reads settings from the .env file (or environment variables)."""

    app_name: str = "ShopVoice AI"
    environment: str = "development"

    # Optional at startup so /health still works without a key.
    # We check it when the AI feature is actually used.
    openai_api_key: SecretStr | None = None
    openai_timeout_seconds: float = 20.0

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()