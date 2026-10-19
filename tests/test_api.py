import json
from collections.abc import Iterator
from pathlib import Path

import httpx
import ollama
import pytest
from fastapi.testclient import TestClient
from llama_index.core.llms import ChatMessage, ChatResponse, MockLLM

from docs_rag.api import create_app
from docs_rag.config import Settings
from docs_rag.knowledge import KnowledgeBase


@pytest.fixture
def client(kb: KnowledgeBase) -> Iterator[TestClient]:
    app = create_app(kb=kb, settings=Settings(max_upload_mb=1))
    with TestClient(app) as client:
        yield client


def parse_sse(body: str) -> list[tuple[str, object]]:
    events = []
    for block in body.strip().split("\n\n"):
        name, data = "message", None
        for line in block.splitlines():
            if line.startswith("event: "):
                name = line.removeprefix("event: ")
            elif line.startswith("data: "):
                data = json.loads(line.removeprefix("data: "))
        if data is not None:
            events.append((name, data))
    return events


def upload(client: TestClient, path: Path, name: str | None = None):
    with path.open("rb") as fh:
        return client.post("/api/documents", files={"file": (name or path.name, fh)})


def test_index_page_is_served(client: TestClient):
    res = client.get("/")

    assert res.status_code == 200
    assert "Docs RAG" in res.text


def test_upload_then_list(client: TestClient, handbook: Path):
    res = upload(client, handbook)

    assert res.status_code == 201
    assert res.json()["pages"] == 7
    docs = client.get("/api/documents").json()
    assert [d["file_name"] for d in docs] == [handbook.name]


def test_upload_rejects_unsupported_type(client: TestClient, tmp_path: Path):
    path = tmp_path / "data.csv"
    path.write_text("a,b\n1,2\n")

    res = upload(client, path)

    assert res.status_code == 415


def test_upload_rejects_large_files(client: TestClient, tmp_path: Path):
    path = tmp_path / "big.txt"
    path.write_text("word " * 300_000)

    res = upload(client, path)

    assert res.status_code == 413


def test_upload_reports_files_without_text(client: TestClient, tmp_path: Path):
    path = tmp_path / "empty.md"
    path.write_text("   \n")

    res = upload(client, path)

    assert res.status_code == 422
    assert "no text" in res.json()["detail"]


def test_upload_strips_directories_from_file_name(client: TestClient, handbook: Path):
    res = upload(client, handbook, name="../../etc/handbook.pdf")

    assert res.status_code == 201
    assert res.json()["file_name"] == "handbook.pdf"


def test_delete_document(client: TestClient, handbook: Path):
    upload(client, handbook)

    assert client.delete(f"/api/documents/{handbook.name}").status_code == 204
    assert client.get("/api/documents").json() == []
    assert client.delete(f"/api/documents/{handbook.name}").status_code == 404


def test_ask_streams_sources_tokens_then_done(client: TestClient, handbook: Path):
    upload(client, handbook)

    res = client.post("/api/ask", json={"question": "how many days of annual leave"})

    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(res.text)
    names = [name for name, _ in events]
    assert names[0] == "sources"
    assert names[-1] == "done"
    assert set(names[1:-1]) == {"token"}

    first = events[0][1][0]
    assert first["number"] == 1
    assert first["file_name"] == handbook.name
    assert first["location"] == "page 3"
    assert "14 working days" in first["text"]


def test_ask_reports_cited_sources(kb_with_llm, handbook: Path):
    # Without max_tokens MockLLM echoes the prompt back, and the prompt lists [1] to [3].
    kb = kb_with_llm(MockLLM())
    with TestClient(create_app(kb=kb, settings=Settings())) as client:
        upload(client, handbook)
        events = parse_sse(client.post("/api/ask", json={"question": "annual leave"}).text)

    assert events[-1] == ("done", {"cited": [1, 2, 3]})


def test_ask_ignores_citations_without_a_source(kb_with_llm, handbook: Path):
    class MadeUpCitationLLM(MockLLM):
        async def astream_chat(self, messages, **kwargs):
            async def gen():
                text = "Fourteen working days [1][9]."
                yield ChatResponse(message=ChatMessage(role="assistant", content=text), delta=text)

            return gen()

    kb = kb_with_llm(MadeUpCitationLLM())
    with TestClient(create_app(kb=kb, settings=Settings())) as client:
        upload(client, handbook)
        events = parse_sse(client.post("/api/ask", json={"question": "annual leave"}).text)

    assert len(events[0][1]) < 9
    assert events[-1] == ("done", {"cited": [1]})


def test_ask_without_matches_says_so(client: TestClient, handbook: Path):
    upload(client, handbook)

    events = parse_sse(client.post("/api/ask", json={"question": "zebra volcano?"}).text)

    assert events[0] == ("sources", [])
    assert "couldn't find" in "".join(data for name, data in events if name == "token")


def test_ask_turns_llm_failure_into_error_event(kb_with_llm, handbook: Path):
    class BrokenLLM(MockLLM):
        async def astream_chat(self, messages, **kwargs):
            raise ConnectionError("ollama is down")

    kb = kb_with_llm(BrokenLLM())
    with TestClient(create_app(kb=kb, settings=Settings())) as client:
        upload(client, handbook)
        events = parse_sse(client.post("/api/ask", json={"question": "annual leave"}).text)

    assert events[0][0] == "sources"
    assert events[-1][0] == "error"


def test_ask_names_a_missing_model(kb_with_llm, handbook: Path):
    class NotPulledLLM(MockLLM):
        async def astream_chat(self, messages, **kwargs):
            raise ollama.ResponseError("model 'qwen3.5:9b' not found", 404)

    kb = kb_with_llm(NotPulledLLM())
    with TestClient(create_app(kb=kb, settings=Settings())) as client:
        upload(client, handbook)
        events = parse_sse(client.post("/api/ask", json={"question": "annual leave"}).text)

    assert events[-1] == (
        "error",
        "Ollama doesn't have the model qwen3.5:9b. Pull it with: ollama pull qwen3.5:9b",
    )


def test_upload_reports_a_missing_embedding_model(
    flaky_kb: KnowledgeBase, failing_embedding, handbook: Path
):
    failing_embedding.error = ollama.ResponseError(
        '{"error":"model \\"nomic-embed-text\\" not found, try pulling it first"}', 404
    )
    with TestClient(create_app(kb=flaky_kb, settings=Settings())) as client:
        res = upload(client, handbook)

    assert res.status_code == 502
    assert res.json()["detail"].endswith("ollama pull nomic-embed-text")


def test_upload_reports_a_provider_timeout(
    flaky_kb: KnowledgeBase, failing_embedding, handbook: Path
):
    failing_embedding.error = httpx.ReadTimeout("timed out")
    with TestClient(create_app(kb=flaky_kb, settings=Settings())) as client:
        res = upload(client, handbook)

    assert res.status_code == 504
    assert "too long" in res.json()["detail"]


def test_ask_validates_question(client: TestClient):
    assert client.post("/api/ask", json={"question": ""}).status_code == 422
