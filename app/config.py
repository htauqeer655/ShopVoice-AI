from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Reads settings from the .env file (or environment variables)."""

    app_name: str = "ShopVoice AI"
    environment: str = "development"

    # Optional at startup so /health still works without a key.
    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-4.1-mini"
    openai_timeout_seconds: float = 20.0
    openai_max_retries: int = 1    
    business_name: str = "DemoShop India"
    max_question_chars: int = 1000

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()