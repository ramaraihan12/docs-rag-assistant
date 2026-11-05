import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from llama_index.core.llms import MockLLM
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

from docs_rag.api import DEMO_MAX_BODY_BYTES, DEMO_MODEL_BUSY, create_app
from docs_rag.config import Settings
from docs_rag.knowledge import KnowledgeBase
from docs_rag.quota import Quota
from tests.conftest import make_kb
from tests.test_api import parse_sse


@pytest.fixture
def engine():
    return create_engine(
        "sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False}
    )


@pytest.fixture
def quota(engine) -> Quota:
    return Quota(engine, "chunks_quota", per_minute=3, per_day=10, all_per_day=10)


@pytest.fixture
def demo_kb(handbook: Path) -> KnowledgeBase:
    kb = make_kb()
    kb.ingest_file(handbook)
    return kb


@pytest.fixture
def client(demo_kb: KnowledgeBase, quota: Quota) -> Iterator[TestClient]:
    settings = Settings(demo_mode=True, demo_max_question_chars=100)
    with TestClient(create_app(kb=demo_kb, settings=settings, quota=quota)) as client:
        yield client


def ask(client: TestClient, question: str = "annual leave", ip: str = "1.1.1.1"):
    return client.post("/api/ask", json={"question": question}, headers={"x-real-ip": ip})


def hits(engine) -> int:
    with engine.connect() as conn:
        return conn.execute(text("SELECT COUNT(*) FROM chunks_quota")).scalar()


def test_page_says_it_is_a_demo(client: TestClient):
    page = client.get("/").text

    assert 'data-demo data-max-question="100"' in page
    assert "Public demo" in page


def test_documents_can_still_be_listed(client: TestClient, handbook: Path):
    docs = client.get("/api/documents").json()

    assert [d["file_name"] for d in docs] == [handbook.name]


def test_uploads_are_refused(client: TestClient, demo_kb: KnowledgeBase, handbook: Path):
    with handbook.open("rb") as fh:
        res = client.post("/api/documents", files={"file": ("new.pdf", fh)})

    assert res.status_code == 403
    assert "read-only" in res.json()["detail"]
    assert [d.file_name for d in demo_kb.documents()] == [handbook.name]


def test_deletes_are_refused(client: TestClient, demo_kb: KnowledgeBase, handbook: Path):
    res = client.delete(f"/api/documents/{handbook.name}")

    assert res.status_code == 403
    assert [d.file_name for d in demo_kb.documents()] == [handbook.name]


def test_ask_answers_within_the_limit(client: TestClient):
    res = ask(client)

    assert res.status_code == 200
    assert parse_sse(res.text)[0][0] == "sources"


def test_ask_is_refused_over_the_limit_with_a_real_status(client: TestClient):
    for _ in range(3):
        assert ask(client).status_code == 200

    res = ask(client)

    assert res.status_code == 429
    assert int(res.headers["retry-after"]) > 0
    assert "Try again" in res.json()["detail"]


def test_the_limit_is_per_client(client: TestClient):
    for _ in range(3):
        ask(client, ip="1.1.1.1")

    assert ask(client, ip="1.1.1.1").status_code == 429
    assert ask(client, ip="2.2.2.2").status_code == 200


def test_a_long_question_is_refused_before_it_counts(client: TestClient, engine):
    res = ask(client, "x" * 101)

    assert res.status_code == 413
    assert "100 characters" in res.json()["detail"]
    assert hits(engine) == 0


def test_a_large_body_is_refused_before_it_counts(client: TestClient, engine):
    res = client.post(
        "/api/ask",
        content=b'{"question": "' + b"x" * DEMO_MAX_BODY_BYTES + b'"}',
        headers={"content-type": "application/json"},
    )

    assert res.status_code == 413
    assert hits(engine) == 0


@pytest.mark.parametrize("body", [b"not json", b"{}", b'{"question": ""}'])
def test_a_malformed_body_doesnt_spend_the_shared_budget(client: TestClient, engine, body: bytes):
    res = client.post("/api/ask", content=body, headers={"content-type": "application/json"})

    assert res.status_code == 422
    assert hits(engine) == 0


def test_no_answer_doesnt_suggest_uploading(handbook: Path, quota: Quota):
    kb = make_kb(read_only=True)
    kb.ingest_file(handbook)
    with TestClient(create_app(kb=kb, settings=Settings(demo_mode=True), quota=quota)) as client:
        events = parse_sse(ask(client, "zebra volcano?").text)

    answer = "".join(data for name, data in events if name == "token")
    assert "couldn't find" in answer
    assert "upload" not in answer


class BusyError(Exception):
    def __init__(self, code: int) -> None:
        self.code = code


class BusyLLM(MockLLM):
    async def astream_chat(self, messages, **kwargs):
        raise BusyError(self.max_tokens)


@pytest.mark.parametrize("code", [429, 503])
def test_a_busy_model_gets_a_readable_message(handbook: Path, quota: Quota, code: int):
    kb = make_kb(BusyLLM(max_tokens=code))
    kb.ingest_file(handbook)
    with TestClient(create_app(kb=kb, settings=Settings(demo_mode=True), quota=quota)) as client:
        events = parse_sse(ask(client).text)

    assert events[-1] == ("error", DEMO_MODEL_BUSY)


def test_the_demo_runs_without_the_file_parsers():
    # They aren't in the Vercel bundle, so importing one there takes the whole site down.
    code = """
import sys
from docs_rag.api import create_app
from docs_rag.config import Settings
create_app(settings=Settings(demo_mode=True))
loaded = {"docx", "pypdf", "lxml", "docs_rag.loaders"} & set(sys.modules)
sys.exit(f"imported {sorted(loaded)}" if loaded else 0)
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr


class TestFlagOff:
    @pytest.fixture
    def client(self, kb: KnowledgeBase) -> Iterator[TestClient]:
        with TestClient(create_app(kb=kb, settings=Settings())) as client:
            yield client

    def test_page_is_unchanged(self, client: TestClient):
        assert "data-demo" not in client.get("/").text.split(">", 2)[1]

    def test_uploads_and_deletes_work(self, client: TestClient, handbook: Path):
        with handbook.open("rb") as fh:
            assert client.post("/api/documents", files={"file": fh}).status_code == 201
        assert client.delete(f"/api/documents/{handbook.name}").status_code == 204

    def test_a_busy_model_keeps_the_generic_message(self, handbook: Path):
        kb = make_kb(BusyLLM(max_tokens=503))
        kb.ingest_file(handbook)
        with TestClient(create_app(kb=kb, settings=Settings())) as client:
            events = parse_sse(ask(client).text)

        assert events[-1][0] == "error"
        assert "Something went wrong" in events[-1][1]

    def test_ask_has_no_limit_and_keeps_its_length_cap(self, client: TestClient):
        for _ in range(8):
            assert ask(client, "x" * 2000).status_code == 200
        assert ask(client, "x" * 2001).status_code == 422
