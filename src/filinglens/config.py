"""Application settings, loaded from environment variables and `.env`.

All configuration goes through `get_settings()`; nothing else reads `os.environ` directly.
"""

import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_EMAIL_RE = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")


class Settings(BaseSettings):
    # env_parse_none_str="": an empty value in .env (e.g. `OPENAI_API_KEY=`) means "not set"
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", env_parse_none_str=""
    )

    # SEC EDGAR (required by the EDGAR client; optional so DB/eval commands work without it)
    sec_user_agent: str | None = None

    # Database
    database_url: str = "postgresql+psycopg://filinglens:filinglens@localhost:5432/filinglens"

    # LLM
    llm_provider: Literal["openai", "anthropic"] = "anthropic"
    llm_model: str = "claude-sonnet-5-5"
    # 0 for reproducible evals; set empty for models that reject a temperature (reasoning models)
    llm_temperature: float | None = 0.0
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None

    # Embeddings
    embedding_model: str = "text-embedding-3-small"

    # Observability
    langfuse_public_key: str | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_host: str = "https://cloud.langfuse.com"

    # Local paths
    data_dir: Path = Path("data")

    @field_validator("sec_user_agent")
    @classmethod
    def _user_agent_has_email(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not _EMAIL_RE.search(value):
            raise ValueError("SEC_USER_AGENT must contain a name and a contact email")
        return value.strip()


@lru_cache
def get_settings() -> Settings:
    return Settings()
