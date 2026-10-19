from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProvider = Literal["ollama", "openai", "gemini", "anthropic"]
EmbedProvider = Literal["ollama", "openai", "gemini"]

# Qwen3-Embedding is trained with an instruction prefix on the query side only.
QWEN3_QUERY_INSTRUCTION = (
    "Instruct: Given a question, retrieve passages from company documents that answer it\nQuery:"
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://postgres:postgres@localhost:5432/rag_assistant"
    table_name: str = "chunks"

    llm_provider: LLMProvider = "ollama"
    llm_model: str = "qwen3.5:9b"
    embed_provider: EmbedProvider = "ollama"
    embed_model: str = "qwen3-embedding:0.6b"
    embed_dim: int = 1024
    embed_query_instruction: str | None = QWEN3_QUERY_INSTRUCTION

    ollama_base_url: str = "http://localhost:11434"
    # Ollama reloads the model whenever num_ctx changes, so match whatever else uses it.
    ollama_context_window: int = 8192
    openai_api_key: SecretStr | None = None
    google_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None

    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 5
    similarity_cutoff: float = 0.4
    max_upload_mb: int = 20


@lru_cache
def get_settings() -> Settings:
    return Settings()
