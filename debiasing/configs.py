from functools import lru_cache

from pydantic import ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_FILE = ".env"
DEV_ENV_FILE = "dev.env"

OPENAI_COMPLETION_MODEL = "gpt-4o-mini"
# Ref: https://platform.openai.com/docs/guides/embeddings
OPENAI_EMB_MODEL = "text-embedding-3-small"
ANTROPHIC_COMPLETION_MODEL = "claude-3-5-sonnet-20241022"

# Ref: https://fastapi.tiangolo.com/advanced/settings/#read-settings-from-env
# Ref: https://docs.pydantic.dev/latest/concepts/pydantic_settings/
class Settings(BaseSettings):

    # OpenAI configs
    OPENAI_CHAT_ENDPOINT: str = "https://localhost:3000"
    OPENAI_EMB_ENDPOINT: str = "https://localhost:3000"
    OPENAI_API_KEY: str | None = None
    OPENAI_COMPLETION_MODEL: str = OPENAI_COMPLETION_MODEL
    OPENAI_EMB_MODEL: str = OPENAI_EMB_MODEL

    # Antrophic configs
    ANTROPHIC_CHAT_ENDPOINT: str = "https://localhost:3000"
    ANTROPHIC_API_KEY: str | None = None
    ANTROPHIC_COMPLETION_MODEL: str = ANTROPHIC_COMPLETION_MODEL

    # General LLM configs
    LLM_TIMEOUT: int = 15
    TEMPERATURE: float = 0.0
    MAX_TOKENS: int = 400

    model_config: ConfigDict = SettingsConfigDict(
        # env files are overwritten in reverse order of preference.
        # Ref: https://docs.pydantic.dev/latest/concepts/pydantic_settings/#dotenv-env-support
        env_file=(DEV_ENV_FILE, ENV_FILE),
        env_file_encoding="utf-8",
        extra="allow",
    )


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
