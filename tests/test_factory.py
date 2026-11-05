from sqlalchemy import make_url

from docs_rag.factory import _async_url


def test_async_url_translates_libpq_options():
    url = make_url("postgresql://u:p@host/db?sslmode=require&channel_binding=require")

    async_url = make_url(_async_url(url))

    assert async_url.drivername == "postgresql+asyncpg"
    assert dict(async_url.query) == {"ssl": "require"}


def test_async_url_leaves_a_plain_url_alone():
    url = make_url("postgresql://postgres:postgres@localhost:5432/rag_assistant")

    assert _async_url(url) == (
        "postgresql+asyncpg://postgres:postgres@localhost:5432/rag_assistant"
    )
