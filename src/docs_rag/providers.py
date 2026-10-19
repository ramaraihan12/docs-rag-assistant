"""Build the LLM and embedding model from settings.

Provider SDKs are imported inside each branch, so startup only pays for the one in use.
"""

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms import LLM
from pydantic import SecretStr

from docs_rag.config import Settings


def _secret(value: SecretStr | None) -> str | None:
    return value.get_secret_value() if value else None


def build_llm(settings: Settings) -> LLM:
    match settings.llm_provider:
        case "ollama":
            from llama_index.llms.ollama import Ollama

            return Ollama(
                model=settings.llm_model,
                base_url=settings.ollama_base_url,
                request_timeout=180.0,
                context_window=8192,
                temperature=0.1,
                # Reasoning traces are slow on a small GPU and we never show them.
                thinking=False,
            )
        case "openai":
            from llama_index.llms.openai import OpenAI

            return OpenAI(model=settings.llm_model, api_key=_secret(settings.openai_api_key))
        case "gemini":
            from llama_index.llms.google_genai import GoogleGenAI

            return GoogleGenAI(model=settings.llm_model, api_key=_secret(settings.google_api_key))
        case "anthropic":
            from llama_index.llms.anthropic import Anthropic

            return Anthropic(
                model=settings.llm_model,
                api_key=_secret(settings.anthropic_api_key),
                max_tokens=1024,
            )


def build_embed_model(settings: Settings) -> BaseEmbedding:
    match settings.embed_provider:
        case "ollama":
            from llama_index.embeddings.ollama import OllamaEmbedding

            return OllamaEmbedding(
                model_name=settings.embed_model,
                base_url=settings.ollama_base_url,
                query_instruction=settings.embed_query_instruction or None,
            )
        case "openai":
            from llama_index.embeddings.openai import OpenAIEmbedding

            return OpenAIEmbedding(
                model=settings.embed_model,
                dimensions=settings.embed_dim,
                api_key=_secret(settings.openai_api_key),
            )
        case "gemini":
            from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

            return GoogleGenAIEmbedding(
                model_name=settings.embed_model,
                api_key=_secret(settings.google_api_key),
                embedding_config={"output_dimensionality": settings.embed_dim},
            )
