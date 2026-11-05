from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms import LLM
from llama_index.vector_stores.postgres import PGVectorStore
from sqlalchemy import URL, create_engine, make_url
from sqlalchemy.pool import NullPool

from docs_rag.catalog import Catalog
from docs_rag.config import Settings
from docs_rag.knowledge import KnowledgeBase
from docs_rag.providers import build_embed_model, build_llm
from docs_rag.quota import Quota


def _sync_url(url: URL) -> str:
    return url.set(drivername="postgresql+psycopg2").render_as_string(hide_password=False)


def _async_url(url: URL) -> str:
    """asyncpg only knows `ssl`, and raises on libpq's `sslmode` and `channel_binding`."""
    query = dict(url.query)
    query.pop("channel_binding", None)
    if "sslmode" in query:
        query["ssl"] = query.pop("sslmode")
    url = url.set(drivername="postgresql+asyncpg", query=query)
    return url.render_as_string(hide_password=False)


def _engine_options(settings: Settings) -> dict:
    # A serverless instance gets frozen between requests, which leaves pooled sockets dead.
    # The demo connects through Neon's pooler, so it doesn't need a pool of its own.
    return {"poolclass": NullPool} if settings.demo_mode else {}


def build_knowledge_base(
    settings: Settings,
    table_name: str | None = None,
    *,
    embed_model: BaseEmbedding | None = None,
    llm: LLM | None = None,
) -> KnowledgeBase:
    """Wire the real thing: Postgres + pgvector and the configured model providers.

    `embed_model` and `llm` replace the configured providers; the integration tests use
    that to run against a real database without a model server.
    """
    table = table_name or settings.table_name
    url = make_url(settings.database_url)
    sync_url = _sync_url(url)
    engine_options = _engine_options(settings)

    vector_store = PGVectorStore.from_params(
        connection_string=sync_url,
        async_connection_string=_async_url(url),
        table_name=table,
        embed_dim=settings.embed_dim,
        use_jsonb=True,
        hnsw_kwargs={
            "hnsw_m": 16,
            "hnsw_ef_construction": 64,
            "hnsw_ef_search": 40,
            "hnsw_dist_method": "vector_cosine_ops",
        },
        create_engine_kwargs=engine_options,
    )
    catalog = Catalog(
        create_engine(sync_url, pool_pre_ping=True, **engine_options), f"{table}_files"
    )

    return KnowledgeBase(
        vector_store=vector_store,
        catalog=catalog,
        embed_model=embed_model or build_embed_model(settings),
        llm=llm or build_llm(settings),
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        top_k=settings.top_k,
        similarity_cutoff=settings.similarity_cutoff,
        read_only=settings.demo_mode,
    )


def build_quota(settings: Settings) -> Quota:
    url = make_url(settings.database_url)
    return Quota(
        create_engine(_sync_url(url), **_engine_options(settings)),
        f"{settings.table_name}_quota",
        per_minute=settings.demo_questions_per_minute,
        per_day=settings.demo_questions_per_day,
        all_per_day=settings.demo_questions_all_per_day,
    )
