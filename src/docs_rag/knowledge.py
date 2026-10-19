import hashlib
from collections.abc import AsyncIterator
from pathlib import Path

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.indices.vector_store import VectorIndexRetriever
from llama_index.core.llms import LLM
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    MetadataFilter,
    MetadataFilters,
)

from docs_rag.catalog import Catalog, DocumentRecord
from docs_rag.loaders import load_file
from docs_rag.models import AnswerDelta, AnswerEvent, IngestResult, Source, SourcesFound
from docs_rag.prompts import build_messages, no_answer_message


class KnowledgeBase:
    def __init__(
        self,
        vector_store: BasePydanticVectorStore,
        catalog: Catalog,
        embed_model: BaseEmbedding,
        llm: LLM,
        *,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        top_k: int = 5,
        similarity_cutoff: float = 0.45,
    ) -> None:
        self._vector_store = vector_store
        self._catalog = catalog
        self._llm = llm
        self._index = VectorStoreIndex(
            nodes=[],
            storage_context=StorageContext.from_defaults(vector_store=vector_store),
            embed_model=embed_model,
        )
        self._splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        # Not index.as_retriever(): that pins the retriever to the node ids known at
        # creation time, so chunks ingested later would never be found.
        self._retriever = VectorIndexRetriever(self._index, similarity_top_k=top_k)
        self._cutoff = SimilarityPostprocessor(similarity_cutoff=similarity_cutoff)

    def documents(self) -> list[DocumentRecord]:
        return self._catalog.list()

    def ingest_file(self, path: Path, file_name: str | None = None) -> IngestResult:
        name = file_name or path.name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()

        existing = self._catalog.get(name)
        if existing and existing.sha256 == digest:
            return IngestResult(name, existing.pages, existing.chunks, skipped=True)

        docs = load_file(path, file_name=name)

        nodes = self._splitter.get_nodes_from_documents(docs)
        page_numbers = [doc.metadata["page"] for doc in docs if doc.metadata["page"] is not None]
        pages = max(page_numbers) if page_numbers else None

        self._delete_chunks(name)
        self._index.insert_nodes(nodes)
        self._catalog.save(name, digest, pages, len(nodes))
        return IngestResult(name, pages, len(nodes))

    def remove(self, file_name: str) -> bool:
        self._delete_chunks(file_name)
        return self._catalog.remove(file_name)

    def retrieve(self, question: str, *, apply_cutoff: bool = True) -> list[Source]:
        return self._to_sources(self._retriever.retrieve(question), apply_cutoff)

    async def aretrieve(self, question: str, *, apply_cutoff: bool = True) -> list[Source]:
        return self._to_sources(await self._retriever.aretrieve(question), apply_cutoff)

    async def astream_answer(self, question: str) -> AsyncIterator[AnswerEvent]:
        """Yield the sources first, then the answer text as it streams in.

        When nothing clears the similarity cutoff the LLM is never called; a fixed
        "not in your documents" reply is cheaper and can't hallucinate.
        """
        sources = await self.aretrieve(question)
        yield SourcesFound(sources)

        if not sources:
            yield AnswerDelta(no_answer_message(question))
            return

        stream = await self._llm.astream_chat(build_messages(question, sources))
        async for chunk in stream:
            if chunk.delta:
                yield AnswerDelta(chunk.delta)

    def _delete_chunks(self, file_name: str) -> None:
        self._vector_store.delete_nodes(
            filters=MetadataFilters(filters=[MetadataFilter(key="file_name", value=file_name)])
        )

    def _to_sources(self, nodes: list[NodeWithScore], apply_cutoff: bool) -> list[Source]:
        if apply_cutoff:
            nodes = self._cutoff.postprocess_nodes(nodes)
        return [
            Source(
                number=number,
                file_name=item.node.metadata["file_name"],
                page=item.node.metadata.get("page"),
                section=item.node.metadata.get("section"),
                score=round(item.score or 0.0, 4),
                text=item.node.get_content(),
            )
            for number, item in enumerate(nodes, start=1)
        ]
