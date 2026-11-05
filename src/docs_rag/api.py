import logging
import re
import shutil
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import httpx
import ollama
from fastapi import Depends, FastAPI, HTTPException, Request, Response, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.sse import EventSourceResponse, ServerSentEvent
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy.exc import OperationalError

from docs_rag.catalog import DocumentRecord
from docs_rag.config import Settings, get_settings
from docs_rag.errors import PROVIDER_ERRORS, provider_error_message
from docs_rag.knowledge import KnowledgeBase
from docs_rag.models import AnswerDelta, IngestResult, SourcesFound
from docs_rag.quota import Quota, client_ip

STATIC_DIR = Path(__file__).parent / "static"

# A question of the longest allowed length plus the JSON around it fits with room to spare.
DEMO_MAX_BODY_BYTES = 8 * 1024
DEMO_READ_ONLY = "This public demo is read-only, so uploads and deletes are turned off."
DEMO_MODEL_BUSY = "Gemini's free tier is busy right now. Try again in a minute."

log = logging.getLogger(__name__)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


def _kb(request: Request) -> KnowledgeBase:
    return request.app.state.kb


KB = Annotated[KnowledgeBase, Depends(_kb)]


def create_app(
    kb: KnowledgeBase | None = None,
    settings: Settings | None = None,
    quota: Quota | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    max_upload_bytes = settings.max_upload_mb * 1024 * 1024

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if kb is not None:
            app.state.kb = kb
        else:
            from docs_rag.factory import build_knowledge_base

            app.state.kb = build_knowledge_base(settings)
        if settings.demo_mode:
            from docs_rag.factory import build_quota

            app.state.quota = quota or build_quota(settings)
        yield

    app = FastAPI(title="docs-rag-assistant", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.exception_handler(OperationalError)
    async def database_down(request: Request, exc: OperationalError) -> JSONResponse:
        log.error("database error: %s", exc.orig)
        return JSONResponse({"detail": "The database is not reachable."}, status_code=503)

    @app.exception_handler(ConnectionError)
    @app.exception_handler(httpx.ConnectError)
    @app.exception_handler(httpx.TimeoutException)
    @app.exception_handler(ollama.ResponseError)
    async def provider_failed(request: Request, exc: Exception) -> JSONResponse:
        log.error("model provider error: %r", exc)
        if isinstance(exc, ollama.ResponseError):
            code = status.HTTP_502_BAD_GATEWAY
        elif isinstance(exc, httpx.TimeoutException):
            code = status.HTTP_504_GATEWAY_TIMEOUT
        else:
            code = status.HTTP_503_SERVICE_UNAVAILABLE
        return JSONResponse({"detail": provider_error_message(exc)}, status_code=code)

    async def demo_limits(request: Request) -> None:
        """Runs before the event stream opens, so a refusal can still be a real status code."""
        body = await request.body()
        if len(body) > DEMO_MAX_BODY_BYTES:
            raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, "That request is too large.")

        # FastAPI rejects a malformed body right after this. It costs no model call, so it
        # shouldn't spend the shared budget either.
        try:
            question = AskRequest.model_validate_json(body).question
        except ValidationError:
            return

        if len(question) > settings.demo_max_question_chars:
            raise HTTPException(
                status.HTTP_413_CONTENT_TOO_LARGE,
                f"Questions can be up to {settings.demo_max_question_chars} characters here.",
            )

        peer = request.client.host if request.client else None
        ip = client_ip(request.headers, peer)
        refusal = await run_in_threadpool(request.app.state.quota.check, ip)
        if refusal:
            raise HTTPException(
                status.HTTP_429_TOO_MANY_REQUESTS,
                refusal.message,
                headers={"Retry-After": str(refusal.retry_after)},
            )

    @app.get("/", include_in_schema=False)
    def index() -> Response:
        if not settings.demo_mode:
            return FileResponse(STATIC_DIR / "index.html")
        html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        attrs = f'data-demo data-max-question="{settings.demo_max_question_chars}"'
        return HTMLResponse(html.replace("<html", f"<html {attrs}", 1))

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/documents")
    def list_documents(kb: KB) -> list[DocumentRecord]:
        return kb.documents()

    if settings.demo_mode:
        # No file parameter here, so an upload is refused without parsing the body.
        @app.post("/api/documents", include_in_schema=False)
        @app.delete("/api/documents/{file_name}", include_in_schema=False)
        def read_only() -> None:
            raise HTTPException(status.HTTP_403_FORBIDDEN, DEMO_READ_ONLY)

    else:
        from docs_rag.loaders import SUPPORTED_SUFFIXES, DocumentError

        @app.post("/api/documents", status_code=status.HTTP_201_CREATED)
        def upload_document(file: UploadFile, kb: KB) -> IngestResult:
            name = Path(file.filename or "").name
            if Path(name).suffix.lower() not in SUPPORTED_SUFFIXES:
                allowed = ", ".join(sorted(SUPPORTED_SUFFIXES))
                raise HTTPException(
                    status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, f"Supported file types: {allowed}"
                )

            # Starlette counts the bytes while it spools the upload, so this can be checked
            # before copying the file anywhere.
            if file.size is not None and file.size > max_upload_bytes:
                raise HTTPException(
                    status.HTTP_413_CONTENT_TOO_LARGE,
                    f"Files can be up to {settings.max_upload_mb} MB",
                )

            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / name
                with path.open("wb") as out:
                    shutil.copyfileobj(file.file, out)
                try:
                    return kb.ingest_file(path)
                except DocumentError as exc:
                    raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc

        @app.delete("/api/documents/{file_name}", status_code=status.HTTP_204_NO_CONTENT)
        def delete_document(file_name: str, kb: KB) -> None:
            if not kb.remove(file_name):
                raise HTTPException(status.HTTP_404_NOT_FOUND, f"{file_name} is not indexed")

    @app.post(
        "/api/ask",
        response_class=EventSourceResponse,
        dependencies=[Depends(demo_limits)] if settings.demo_mode else [],
    )
    async def ask(body: AskRequest, kb: KB) -> AsyncIterator[ServerSentEvent]:
        """Stream an answer as server-sent events.

        Events, in order: `sources` (the numbered chunks), any number of `token`, then
        `done` with the source numbers the answer actually cited. If the model or the
        database fails mid-way, an `error` event replaces `done`.
        """
        answer: list[str] = []
        source_count = 0
        try:
            async for event in kb.astream_answer(body.question):
                match event:
                    case SourcesFound(sources):
                        source_count = len(sources)
                        data = [{**asdict(s), "location": s.location} for s in sources]
                        yield ServerSentEvent(event="sources", data=data)
                    case AnswerDelta(text):
                        answer.append(text)
                        yield ServerSentEvent(event="token", data=text)
        except PROVIDER_ERRORS as exc:
            # Headers are already sent, so the client can only learn about it in-band.
            log.error("answering failed: %r", exc)
            yield ServerSentEvent(event="error", data=provider_error_message(exc))
            return
        except Exception as exc:
            log.exception("answering failed")
            # google-genai's errors carry the HTTP status as `code`.
            if settings.demo_mode and getattr(exc, "code", None) in (429, 503):
                yield ServerSentEvent(event="error", data=DEMO_MODEL_BUSY)
                return
            yield ServerSentEvent(
                event="error",
                data="Something went wrong while answering. The server log has the details.",
            )
            return

        # Small models sometimes cite a number that was never in the prompt.
        numbers = {int(n) for n in re.findall(r"\[(\d+)\]", "".join(answer))}
        cited = sorted(n for n in numbers if 1 <= n <= source_count)
        yield ServerSentEvent(event="done", data={"cited": cited})

    return app
