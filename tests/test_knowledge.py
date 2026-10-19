import shutil
from pathlib import Path

import pytest

from docs_rag.knowledge import KnowledgeBase
from docs_rag.loaders import EmptyDocument
from docs_rag.models import AnswerDelta, SourcesFound
from docs_rag.prompts import NO_ANSWER


def test_ingest_records_pages_and_chunks(kb: KnowledgeBase, handbook: Path):
    result = kb.ingest_file(handbook)

    assert result.pages == 7
    assert result.chunks >= 7
    assert not result.skipped
    assert [d.file_name for d in kb.documents()] == [handbook.name]


def test_reingesting_unchanged_file_is_skipped(kb: KnowledgeBase, handbook: Path):
    first = kb.ingest_file(handbook)
    second = kb.ingest_file(handbook)

    assert second.skipped
    assert second.chunks == first.chunks


def test_changed_file_replaces_old_chunks(kb: KnowledgeBase, tmp_path: Path):
    path = tmp_path / "policy.md"
    path.write_text("# Parking\n\nVisitors park in bay seven.\n", encoding="utf-8")
    kb.ingest_file(path)

    path.write_text("# Parking\n\nVisitors park behind the warehouse.\n", encoding="utf-8")
    kb.ingest_file(path)

    texts = [s.text for s in kb.retrieve("where do visitors park", apply_cutoff=False)]
    assert any("warehouse" in t for t in texts)
    assert not any("bay seven" in t for t in texts)


def test_failed_reingest_keeps_the_previous_version(
    flaky_kb: KnowledgeBase, failing_embedding, tmp_path: Path
):
    path = tmp_path / "policy.md"
    path.write_text("# Parking\n\nVisitors park in bay seven.\n", encoding="utf-8")
    first = flaky_kb.ingest_file(path)

    path.write_text("# Parking\n\nVisitors park behind the warehouse.\n", encoding="utf-8")
    failing_embedding.error = ConnectionError("ollama is down")
    with pytest.raises(ConnectionError):
        flaky_kb.ingest_file(path)
    failing_embedding.error = None

    texts = [s.text for s in flaky_kb.retrieve("where do visitors park", apply_cutoff=False)]
    assert any("bay seven" in t for t in texts)
    assert flaky_kb.documents()[0].chunks == first.chunks

    retry = flaky_kb.ingest_file(path)

    assert not retry.skipped
    texts = [s.text for s in flaky_kb.retrieve("where do visitors park", apply_cutoff=False)]
    assert any("warehouse" in t for t in texts)


def test_retrieve_returns_numbered_sources_with_pages(kb: KnowledgeBase, handbook: Path):
    kb.ingest_file(handbook)

    sources = kb.retrieve("how many working days of annual leave per year")

    assert [s.number for s in sources] == list(range(1, len(sources) + 1))
    assert sources[0].file_name == handbook.name
    assert sources[0].page == 3
    assert sources[0].location == "page 3"


def test_similarity_cutoff_filters_unrelated_questions(kb: KnowledgeBase, handbook: Path):
    kb.ingest_file(handbook)

    assert kb.retrieve("zebra quantum volcano recipe") == []
    assert kb.retrieve("zebra quantum volcano recipe", apply_cutoff=False) != []


def test_remove_deletes_chunks_and_catalog_row(kb: KnowledgeBase, handbook: Path):
    kb.ingest_file(handbook)

    assert kb.remove(handbook.name)
    assert kb.documents() == []
    assert kb.retrieve("annual leave", apply_cutoff=False) == []
    assert not kb.remove(handbook.name)


def test_file_without_text_is_rejected(kb: KnowledgeBase, tmp_path: Path):
    path = tmp_path / "blank.md"
    path.write_text("\n\n", encoding="utf-8")

    with pytest.raises(EmptyDocument):
        kb.ingest_file(path)


def test_ingest_under_a_different_name(kb: KnowledgeBase, handbook: Path, tmp_path: Path):
    copy = tmp_path / "upload-tmp.pdf"
    shutil.copy(handbook, copy)

    result = kb.ingest_file(copy, file_name="handbook.pdf")

    assert result.file_name == "handbook.pdf"
    assert kb.retrieve("annual leave")[0].file_name == "handbook.pdf"


async def collect(kb: KnowledgeBase, question: str) -> tuple[list, str]:
    events = [event async for event in kb.astream_answer(question)]
    sources = events[0].sources if isinstance(events[0], SourcesFound) else []
    text = "".join(e.text for e in events if isinstance(e, AnswerDelta))
    return sources, text


async def test_answer_streams_after_sources(kb: KnowledgeBase, handbook: Path):
    kb.ingest_file(handbook)

    sources, text = await collect(kb, "when is salary paid each month")

    assert sources
    assert text == "text text text text text text text text"


@pytest.mark.parametrize(
    ("question", "lang"),
    [("zebra volcano recipes?", "en"), ("apa resep zebra?", "id")],
)
async def test_no_sources_means_no_llm_call(
    kb: KnowledgeBase, handbook: Path, question: str, lang: str
):
    kb.ingest_file(handbook)

    sources, text = await collect(kb, question)

    assert sources == []
    assert text == NO_ANSWER[lang]
