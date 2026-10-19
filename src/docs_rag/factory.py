from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms import LLM
from llama_index.vector_stores.postgres import PGVectorStore
from sqlalchemy import create_engine, make_url

from docs_rag.catalog import Catalog
from docs_rag.config import Settings
from docs_rag.knowledge import KnowledgeBase
from docs_rag.providers import build_embed_model, build_llm


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
    sync_url = url.set(drivername="postgresql+psycopg2").render_as_string(hide_password=False)
    async_url = url.set(drivername="postgresql+asyncpg").render_as_string(hide_password=False)

    vector_store = PGVectorStore.from_params(
        connection_string=sync_url,
        async_connection_string=async_url,
        table_name=table,
        embed_dim=settings.embed_dim,
        use_jsonb=True,
        hnsw_kwargs={
            "hnsw_m": 16,
            "hnsw_ef_construction": 64,
            "hnsw_ef_search": 40,
            "hnsw_dist_method": "vector_cosine_ops",
        },
    )
    catalog = Catalog(create_engine(sync_url, pool_pre_ping=True), f"{table}_files")

    return KnowledgeBase(
        vector_store=vector_store,
        catalog=catalog,
        embed_model=embed_model or build_embed_model(settings),
        llm=llm or build_llm(settings),
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        top_k=settings.top_k,
        similarity_cutoff=settings.similarity_cutoff,
    )
