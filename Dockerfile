FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.10.4 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Dependencies first so code changes don't reinstall them.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project

COPY src ./src
COPY samples ./samples
RUN uv sync --locked --no-dev --no-editable

RUN useradd --create-home app
USER app

EXPOSE 8000
CMD ["docs-rag", "serve", "--host", "0.0.0.0", "--port", "8000"]
