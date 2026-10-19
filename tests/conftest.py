import hashlib
import math
import re
from collections.abc import Callable
from pathlib import Path

import pytest
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms import LLM, MockLLM
from llama_index.core.vector_stores import SimpleVectorStore
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from docs_rag.catalog import Catalog
from docs_rag.knowledge import KnowledgeBase

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


class KeywordEmbedding(BaseEmbedding):
    """Hashed bag-of-words vectors.

    MockEmbedding returns the same vector for every input, so every chunk scores 1.0 and
    the similarity cutoff can't be tested with it. Word overlap is crude but deterministic
    and offline, which is all these tests need.
    """

    dim: int = 256

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            if len(word) > 2:
                bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim
                vec[bucket] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._vector(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._vector(text)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._vector(query)


def memory_catalog() -> Catalog:
    engine = create_engine(
        "sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False}
    )
    return Catalog(engine, "chunks_files")


def make_kb(llm: LLM | None = None) -> KnowledgeBase:
    """An in-memory knowledge base: no Postgres, no model server."""
    return KnowledgeBase(
        vector_store=SimpleVectorStore(),
        catalog=memory_catalog(),
        embed_model=KeywordEmbedding(),
        llm=llm or MockLLM(max_tokens=8),
        chunk_size=256,
        chunk_overlap=32,
        top_k=3,
        similarity_cutoff=0.2,
    )


@pytest.fixture
def kb() -> KnowledgeBase:
    return make_kb()


@pytest.fixture
def kb_with_llm() -> Callable[[LLM], KnowledgeBase]:
    return make_kb


@pytest.fixture
def handbook() -> Path:
    return SAMPLES / "kabut-labs-employee-handbook.pdf"
