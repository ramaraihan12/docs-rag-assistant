import logging
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import httpx
import pytest
from pydantic import SecretStr

from docs_rag import tracing
from docs_rag.config import Settings


def settings(**kw) -> Settings:
    base = {"langfuse_public_key": SecretStr("pk"), "langfuse_secret_key": SecretStr("sk")}
    return Settings(**{**base, "trace_sample_rate": 1.0, **kw})


def run_send(s: Settings, **kw):
    args = {
        "question": "How many leave days?",
        "answer": "Twelve days [1].",
        "cited": [1],
        "file_names": ["kabut-labs-employee-handbook.pdf", "mine.pdf"],
        "source_count": 2,
        "error": False,
        "start": datetime.now(UTC) - timedelta(seconds=1),
    }
    return tracing.send_trace(s, **{**args, **kw})


def test_redaction_order_and_caps():
    text = "mail a.b@example.com call +62 812-3456-7890 or 081234567890 id 1234567 tok " + "a" * 30
    out = tracing.redact(text, 500)
    assert "[email]" in out and out.count("[phone]") == 2
    assert "[number]" in out and "[token]" in out
    assert "example" not in out and "7890" not in out
    assert tracing.redact("x " * 600, 500) == ("x " * 600)[:500]
    assert len(tracing.redact("word " * 1000, 2000)) == 2000


def test_short_numbers_survive():
    assert tracing.redact("12 days in 2026, page 3", 500) == "12 days in 2026, page 3"


def test_sampling_is_deterministic_and_normalised():
    assert tracing.is_sampled("Hello", 1.0)
    assert not tracing.is_sampled("Hello", 0.0)
    assert tracing.is_sampled("  HELLO ", 0.3) == tracing.is_sampled("hello", 0.3)
    hits = sum(tracing.is_sampled(f"q{i}", 0.25) for i in range(2000))
    assert 380 < hits < 620


def test_flags():
    f = tracing.flags_for
    assert f(error=False, source_count=0, answer="x", seconds=1) == ["no_sources"]
    assert f(error=False, source_count=2, answer="no cite", seconds=1) == ["no_citation"]
    assert f(error=False, source_count=2, answer="ok [1]", seconds=1) == []
    assert f(error=True, source_count=2, answer="", seconds=16) == ["error", "slow"]


@pytest.mark.asyncio
async def test_payload_shape(monkeypatch):
    post = AsyncMock()
    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    monkeypatch.setenv("VERCEL_GIT_COMMIT_SHA", "abc123")
    await run_send(settings(), answer="No cite, mail me@x.io", source_count=2)

    (url,), kwargs = post.call_args
    assert url == "https://cloud.langfuse.com/api/public/ingestion"
    assert kwargs["auth"] == ("pk", "sk")
    trace, gen = kwargs["json"]["batch"]
    assert trace["type"] == "trace-create" and gen["type"] == "generation-create"
    assert trace["id"] != gen["id"]
    tb, gb = trace["body"], gen["body"]
    assert tb["name"] == "docs-rag.ask" and tb["release"] == "abc123"
    assert tb["tags"] == ["app:docs-rag", "flag:no_citation"]
    assert tb["metadata"]["flags"] == ["no_citation"] and tb["metadata"]["sample_rate"] == 1.0
    assert gb["traceId"] == tb["id"] and gb["level"] == "WARNING"
    assert gb["output"] == tb["output"] and "[email]" in tb["output"]["answer"]
    assert tb["output"]["sources"] == ["kabut-labs-employee-handbook.pdf", "[uploaded]"]
    assert "text" not in str(tb["output"]["sources"])


@pytest.mark.asyncio
async def test_levels(monkeypatch):
    post = AsyncMock()
    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    await run_send(settings())
    assert post.call_args.kwargs["json"]["batch"][1]["body"]["level"] == "DEFAULT"
    await run_send(settings(), error=True)
    assert post.call_args.kwargs["json"]["batch"][1]["body"]["level"] == "ERROR"


@pytest.mark.asyncio
async def test_off_without_keys(monkeypatch):
    post = AsyncMock()
    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    await run_send(
        Settings(trace_sample_rate=1.0, langfuse_public_key=None, langfuse_secret_key=None)
    )
    await run_send(settings(langfuse_secret_key=None))
    post.assert_not_called()


@pytest.mark.asyncio
async def test_unsampled_sends_nothing(monkeypatch):
    post = AsyncMock()
    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    await run_send(settings(trace_sample_rate=0.0))
    post.assert_not_called()


@pytest.mark.asyncio
async def test_exceptions_swallowed(monkeypatch, caplog):
    monkeypatch.setattr(
        httpx.AsyncClient, "post", AsyncMock(side_effect=httpx.ReadTimeout("boom pk"))
    )
    with caplog.at_level(logging.WARNING):
        await run_send(settings())
    assert "ReadTimeout" in caplog.text and "pk" not in caplog.text.replace("ReadTimeout", "")


def test_ask_sends_after_answer(monkeypatch, handbook):
    from fastapi.testclient import TestClient

    from docs_rag.api import create_app
    from tests.conftest import make_kb

    kb = make_kb()
    kb.ingest_file(handbook)
    post = AsyncMock()
    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    with TestClient(create_app(kb=kb, settings=settings())) as client:
        resp = client.post("/api/ask", json={"question": "annual leave"})
    assert "event: done" in resp.text
    assert post.await_count == 1
