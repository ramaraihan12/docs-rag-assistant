"""One-line explanations for model provider failures, shared by the CLI and the API."""

import re

import httpx
import ollama

# The Ollama client wraps connect errors in ConnectionError but lets timeouts through as
# plain httpx exceptions. A model that was never pulled comes back as a 404 ResponseError.
PROVIDER_ERRORS = (
    ConnectionError,
    httpx.ConnectError,
    httpx.TimeoutException,
    ollama.ResponseError,
)

_MISSING_MODEL = re.compile(r"""model ["']([^"']+)["'] not found""")


def provider_error_message(exc: Exception) -> str:
    if isinstance(exc, ollama.ResponseError):
        if missing := _MISSING_MODEL.search(exc.error):
            name = missing.group(1)
            return f"Ollama doesn't have the model {name}. Pull it with: ollama pull {name}"
        return f"Ollama returned an error: {exc.error}"
    if isinstance(exc, httpx.TimeoutException):
        return "The model provider took too long to answer."
    return f"Can't reach the model provider: {exc}"
