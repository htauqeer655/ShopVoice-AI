from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


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

    # Knowledge base / RAG
    openai_embedding_model: str = "text-embedding-3-small"
    knowledge_base_dir: Path = PROJECT_ROOT / "data" / "knowledge_base"
    chroma_dir: Path = PROJECT_ROOT / "data" / "chroma_db"
    rag_chunk_size: int = 800
    rag_chunk_overlap: int = 100
    rag_top_k: int = 4
    rag_max_distance: float = 0.6

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()