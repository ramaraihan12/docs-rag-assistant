"""Entrypoint for hosts that import an `app` object. Vercel finds this file by its name."""

from docs_rag.api import create_app

app = create_app()
