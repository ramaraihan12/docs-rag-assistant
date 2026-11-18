import json
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
    return Settings(_env_file=None, **{**base, "trace_sample_rate": 1.0, **kw})


def ok_post(status: int = 200) -> AsyncMock:
    return AsyncMock(return_value=httpx.Response(status))


def attrs(kwargs) -> dict:
    span = kwargs["json"]["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
    return {a["key"]: a["value"] for a in span["attributes"]}


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


@pytest.mark.parametrize("key", ["3f9a8b7c1234567d9e0f1a2b3c4d5e6f", "a1b2c3d4e5" * 4])
def test_hex_key_with_digit_run_is_one_token(key):
    assert tracing.redact(f"key {key} end", 500) == "key [token] end"


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
    assert f(error=False, source_count=0, cited=[], seconds=1) == ["no_sources"]
    assert f(error=False, source_count=2, cited=[], seconds=1) == ["no_citation"]
    assert f(error=False, source_count=2, cited=[1], seconds=1) == []
    assert f(error=True, source_count=2, cited=[], seconds=16) == ["error", "slow"]


def test_phone_rule():
    r = tracing.redact
    assert r("call +62 812-3456-7890", 500) == "call [phone]"
    assert r("call 0812" + "‐" + "3456" + "–" + "7890", 500) == "call [phone]"  # noqa: RUF001
    assert r("HP0812-3456-7890", 500) == "HP[phone]"
    assert r("WA+62 812 3456 7890", 500) == "WA[phone]"
    assert r("Telp.0812-3456-7890", 500) == "Telp.[phone]"
    assert r("Rp 1.050.000.000", 500) == "Rp 1.050.000.000"
    assert r("Rp 1,050,000,000", 500) == "Rp 1,050,000,000"
    assert r("2026-11-18 10:30", 500) == "2026-11-18 10:30"
    assert "[phone]" in r("call (021) 555-1234", 500)
    assert r("on 2026-11-18 10:30 sharp", 500) == "on 2026-11-18 10:30 sharp"
    assert r("total Rp 1.500.000.000 now", 500) == "total Rp 1.500.000.000 now"
    assert r("total 1 500 000 000", 500) == "total 1 500 000 000"


def test_redacts_slice_before_cut():
    # A phone straddling the cap is redacted whole, never left as a partial number.
    text = "x" * 497 + " 0812-3456-7890"
    out = tracing.redact(text, 500)
    assert "0812" not in out and "78" not in out


def test_invalid_sample_rate_falls_back(monkeypatch):
    monkeypatch.setenv("TRACE_SAMPLE_RATE", "lots")
    assert Settings(_env_file=None).trace_sample_rate == 0.25
    assert Settings(_env_file=None, trace_sample_rate=7).trace_sample_rate == 0.25
    assert Settings(_env_file=None, trace_sample_rate=0.5).trace_sample_rate == 0.5


@pytest.mark.asyncio
async def test_total_deadline(monkeypatch, caplog):
    import asyncio

    monkeypatch.setattr(tracing, "TIMEOUT_SECONDS", 0.05)

    async def slow(*a, **k):
        await asyncio.sleep(5)

    monkeypatch.setattr(httpx.AsyncClient, "post", slow)
    with caplog.at_level(logging.WARNING):
        await run_send(settings())
    assert "TimeoutError" in caplog.text


@pytest.mark.asyncio
async def test_payload_shape(monkeypatch):
    post = ok_post()
    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    monkeypatch.setenv("VERCEL_GIT_COMMIT_SHA", "abc123")
    await run_send(settings(), answer="No cite, mail me@x.io", cited=[], source_count=2)

    (url,), kwargs = post.call_args
    assert url == "https://cloud.langfuse.com/api/public/otel/v1/traces"
    assert kwargs["auth"] == ("pk", "sk")
    assert kwargs["headers"] == {"x-langfuse-ingestion-version": "4"}
    json.dumps(kwargs["json"])
    spans = kwargs["json"]["resourceSpans"][0]["scopeSpans"][0]["spans"]
    assert len(spans) == 1
    span = spans[0]
    assert len(span["traceId"]) == 32 and len(span["spanId"]) == 16
    assert int(span["endTimeUnixNano"]) > int(span["startTimeUnixNano"])
    a = attrs(kwargs)
    assert a["langfuse.trace.name"] == {"stringValue": "docs-rag.ask"}
    assert a["langfuse.environment"] == {"stringValue": "production"}
    assert a["langfuse.release"] == {"stringValue": "abc123"}
    assert a["langfuse.observation.type"] == {"stringValue": "generation"}
    assert a["langfuse.observation.level"] == {"stringValue": "WARNING"}
    tags = [t["stringValue"] for t in a["langfuse.trace.tags"]["arrayValue"]["values"]]
    assert tags == ["app:docs-rag", "flag:no_citation"]
    output = json.loads(a["langfuse.observation.output"]["stringValue"])
    assert "[email]" in output["answer"]
    assert output["sources"] == ["kabut-labs-employee-handbook.pdf", "[uploaded]"]
    assert json.loads(a["langfuse.observation.input"]["stringValue"]) == {
        "question": "How many leave days?"
    }
    assert json.loads(a["langfuse.observation.metadata.flags"]["stringValue"]) == ["no_citation"]
    assert a["langfuse.observation.metadata.sample_rate"] == {"stringValue": "1.0"}


@pytest.mark.asyncio
async def test_levels(monkeypatch):
    post = ok_post()
    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    await run_send(settings())
    assert attrs(post.call_args.kwargs)["langfuse.observation.level"] == {"stringValue": "DEFAULT"}
    await run_send(settings(), error=True)
    assert attrs(post.call_args.kwargs)["langfuse.observation.level"] == {"stringValue": "ERROR"}


@pytest.mark.asyncio
async def test_rejected_response_logs_status_only(monkeypatch, caplog):
    body = httpx.Response(400, text="secret answer text")
    monkeypatch.setattr(httpx.AsyncClient, "post", AsyncMock(return_value=body))
    with caplog.at_level(logging.WARNING):
        await run_send(settings())
    assert "400" in caplog.text and "secret" not in caplog.text


@pytest.mark.asyncio
async def test_ok_response_logs_nothing(monkeypatch, caplog):
    monkeypatch.setattr(httpx.AsyncClient, "post", ok_post(200))
    with caplog.at_level(logging.WARNING):
        await run_send(settings())
    assert caplog.text == ""


@pytest.mark.asyncio
async def test_off_without_keys(monkeypatch):
    post = ok_post()
    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    await run_send(
        Settings(
            _env_file=None,
            trace_sample_rate=1.0,
            langfuse_public_key=None,
            langfuse_secret_key=None,
        )
    )
    await run_send(settings(langfuse_secret_key=None))
    post.assert_not_called()


@pytest.mark.asyncio
async def test_unsampled_sends_nothing(monkeypatch):
    post = ok_post()
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
    post = ok_post()
    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    with TestClient(create_app(kb=kb, settings=settings())) as client:
        resp = client.post("/api/ask", json={"question": "annual leave"})
    assert "event: done" in resp.text
    assert post.await_count == 1


def test_ask_survives_task_failure(monkeypatch, handbook):
    import asyncio

    from fastapi.testclient import TestClient

    from docs_rag.api import create_app
    from tests.conftest import make_kb

    kb = make_kb()
    kb.ingest_file(handbook)

    def boom(coro, **kw):
        coro.close()
        raise RuntimeError("no loop")

    monkeypatch.setattr(asyncio, "create_task", boom)
    with TestClient(create_app(kb=kb, settings=settings())) as client:
        resp = client.post("/api/ask", json={"question": "annual leave"})
    assert "event: done" in resp.text
