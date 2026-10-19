"""Runs the real Postgres path (pgvector, asyncpg, the catalog table).

Needs TEST_DATABASE_URL pointing at a database with the pgvector extension available.
Models are still faked, so no Ollama or API keys are required.
"""

import os
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from llama_index.core.llms import MockLLM
from sqlalchemy import create_engine, make_url, text
from sqlalchemy.exc import OperationalError

from docs_rag.config import Settings
from docs_rag.factory import build_knowledge_base
from docs_rag.knowledge import KnowledgeBase
from docs_rag.models import AnswerDelta, SourcesFound

DATABASE_URL = os.environ.get("TEST_DATABASE_URL")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not DATABASE_URL, reason="TEST_DATABASE_URL is not set"),
]


@pytest.fixture
def pg_kb(keyword_embedding) -> Iterator[KnowledgeBase]:
    table = f"test_{uuid.uuid4().hex[:8]}"
    settings = Settings(
        database_url=DATABASE_URL,
        embed_dim=keyword_embedding.dim,
        top_k=3,
        similarity_cutoff=0.2,
    )
    try:
        kb = build_knowledge_base(
            settings, table, embed_model=keyword_embedding, llm=MockLLM(max_tokens=4)
        )
    except OperationalError as exc:
        pytest.skip(f"Postgres not reachable: {exc.orig}")

    yield kb

    engine = create_engine(make_url(DATABASE_URL).set(drivername="postgresql+psycopg2"))
    with engine.begin() as conn:
        conn.execute(text(f'DROP TABLE IF EXISTS "data_{table}", "{table}_files"'))
    engine.dispose()


def test_ingest_and_retrieve_with_pages(pg_kb: KnowledgeBase, handbook: Path):
    result = pg_kb.ingest_file(handbook)

    sources = pg_kb.retrieve("how many working days of annual leave per year")

    assert result.pages == 7
    assert sources[0].page == 3
    assert [d.file_name for d in pg_kb.documents()] == [handbook.name]


async def test_async_answer_path(pg_kb: KnowledgeBase, handbook: Path):
    pg_kb.ingest_file(handbook)

    events = [e async for e in pg_kb.astream_answer("when is salary paid each month")]

    assert isinstance(events[0], SourcesFound)
    assert events[0].sources[0].page == 4
    assert "".join(e.text for e in events if isinstance(e, AnswerDelta)) == "text text text text"


def test_reingest_replaces_and_remove_deletes(pg_kb: KnowledgeBase, tmp_path: Path):
    path = tmp_path / "parking.md"
    path.write_text("# Parking\n\nVisitors park in bay seven.\n", encoding="utf-8")
    pg_kb.ingest_file(path)
    path.write_text("# Parking\n\nVisitors park behind the warehouse.\n", encoding="utf-8")
    pg_kb.ingest_file(path)

    texts = [s.text for s in pg_kb.retrieve("where do visitors park", apply_cutoff=False)]
    assert len(texts) == 1
    assert "warehouse" in texts[0]

    assert pg_kb.remove("parking.md")
    assert pg_kb.retrieve("where do visitors park", apply_cutoff=False) == []
