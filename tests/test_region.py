from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from docs_rag.api import create_app
from docs_rag.config import Settings
from docs_rag.knowledge import KnowledgeBase
from docs_rag.quota import Quota
from tests.conftest import make_kb

COUNTRY = "x-vercel-ip-country"


class Counter:
    asked = 0


@pytest.fixture
def kb(handbook: Path) -> KnowledgeBase:
    kb = make_kb()
    kb.ingest_file(handbook)
    return kb


@pytest.fixture
def counter(kb: KnowledgeBase, monkeypatch: pytest.MonkeyPatch) -> Counter:
    """Counts the questions that get as far as the model."""
    counter = Counter()
    real = kb.astream_answer

    async def counting(question: str):
        counter.asked += 1
        async for event in real(question):
            yield event

    monkeypatch.setattr(kb, "astream_answer", counting)
    return counter


@pytest.fixture
def quota() -> Quota:
    engine = create_engine(
        "sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False}
    )
    return Quota(engine, "chunks_quota", per_minute=3, per_day=10, all_per_day=10)


@pytest.fixture
def client(kb: KnowledgeBase, counter: Counter, quota: Quota) -> Iterator[TestClient]:
    settings = Settings(_env_file=None, demo_mode=True)
    with TestClient(create_app(kb=kb, settings=settings, quota=quota)) as client:
        yield client


def ask(client: TestClient, country: str | None):
    headers = {COUNTRY: country} if country else {}
    return client.post("/api/ask", json={"question": "annual leave"}, headers=headers)


@pytest.mark.parametrize("country", ["DE", "FR", "GB", "CH", "NO", "IS", "RE", "de"])
def test_a_blocked_country_gets_451_and_no_model_call(
    client: TestClient, counter: Counter, country: str
):
    res = ask(client, country)

    assert res.status_code == 451
    assert "EEA, Switzerland or the UK" in res.json()["detail"]
    assert counter.asked == 0


def test_a_blocked_country_doesnt_spend_quota(client: TestClient):
    for _ in range(5):
        assert ask(client, "NL").status_code == 451

    assert ask(client, "ID").status_code == 200


def test_an_allowed_country_can_ask(client: TestClient, counter: Counter):
    res = ask(client, "ID")

    assert res.status_code == 200
    assert counter.asked == 1


def test_a_missing_header_is_allowed(client: TestClient, counter: Counter):
    res = ask(client, None)

    assert res.status_code == 200
    assert counter.asked == 1


def test_the_page_marks_itself_for_a_blocked_country(client: TestClient):
    blocked = client.get("/", headers={COUNTRY: "SE"})
    allowed = client.get("/", headers={COUNTRY: "ID"})

    assert "data-blocked" in blocked.text
    assert blocked.headers["cache-control"] == "private, no-store"
    assert "data-blocked" not in allowed.text


def test_listing_documents_still_works_when_blocked(client: TestClient):
    assert client.get("/api/documents", headers={COUNTRY: "DE"}).status_code == 200


def test_uploads_are_gated_outside_demo_mode(kb: KnowledgeBase, handbook: Path):
    with (
        TestClient(create_app(kb=kb, settings=Settings(_env_file=None))) as client,
        handbook.open("rb") as fh,
    ):
        res = client.post(
            "/api/documents", files={"file": ("new.pdf", fh)}, headers={COUNTRY: "DE"}
        )

    assert res.status_code == 451
