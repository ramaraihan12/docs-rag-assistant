"""Gemini's free tier answers 503 "high demand" every so often.

These run the real SDK and the LlamaIndex wrapper against a local server, because the retry
lives inside the SDK and a mock LLM would skip right past it.
"""

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from llama_index.core.llms import ChatMessage

from docs_rag.config import Settings
from docs_rag.providers import build_llm

MODEL = "stub-model"
CHUNK = {"candidates": [{"content": {"role": "model", "parts": [{"text": "Fourteen days."}]}}]}
HIGH_DEMAND = {"error": {"code": 503, "message": "high demand", "status": "UNAVAILABLE"}}


class Gemini:
    """Answers the streaming endpoint with `failures` 503s, then a normal stream."""

    def __init__(self, failures: int, status: int = 503) -> None:
        self.failures = failures
        self.status = status
        self.posts = 0


@pytest.fixture
def stub(monkeypatch: pytest.MonkeyPatch) -> Iterator[Gemini]:
    gemini = Gemini(failures=0)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:
            pass

        def send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            # models.get, which the client calls at startup outside demo mode
            body = json.dumps({"name": f"models/{MODEL}", "inputTokenLimit": 1000})
            self.send(200, body.encode(), "application/json")

        def do_POST(self) -> None:
            self.rfile.read(int(self.headers["Content-Length"]))
            gemini.posts += 1
            if gemini.posts <= gemini.failures:
                self.send(gemini.status, json.dumps(HIGH_DEMAND).encode(), "application/json")
            else:
                self.send(200, f"data: {json.dumps(CHUNK)}\n\n".encode(), "text/event-stream")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", f"http://127.0.0.1:{server.server_port}")
    yield gemini
    server.shutdown()
    server.server_close()


def gemini_llm(*, demo: bool):
    settings = Settings(
        _env_file=None,
        llm_provider="gemini",
        llm_model=MODEL,
        google_api_key="test-key",
        demo_mode=demo,
    )
    return build_llm(settings)


async def answer(llm) -> str:
    stream = await llm.astream_chat([ChatMessage(role="user", content="annual leave?")])
    return "".join([chunk.delta async for chunk in stream if chunk.delta])


async def test_demo_rides_out_a_503(stub: Gemini):
    stub.failures = 1

    assert await answer(gemini_llm(demo=True)) == "Fourteen days."
    assert stub.posts == 2


async def test_demo_gives_up_after_a_few_tries(stub: Gemini):
    stub.failures = 99

    with pytest.raises(Exception, match="503"):
        await answer(gemini_llm(demo=True))

    assert stub.posts == 3


async def test_demo_doesnt_retry_a_used_up_quota(stub: Gemini):
    stub.failures = 1
    stub.status = 429

    with pytest.raises(Exception, match="429"):
        await answer(gemini_llm(demo=True))

    assert stub.posts == 1


async def test_a_503_is_not_retried_with_the_flag_off(stub: Gemini):
    stub.failures = 1

    with pytest.raises(Exception, match="503"):
        await answer(gemini_llm(demo=False))

    assert stub.posts == 1
