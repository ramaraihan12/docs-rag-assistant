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
from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.sse import EventSourceResponse, ServerSentEvent
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy.exc import OperationalError

from docs_rag.catalog import DocumentRecord
from docs_rag.config import Settings, get_settings
from docs_rag.errors import PROVIDER_ERRORS, provider_error_message
from docs_rag.knowledge import KnowledgeBase
from docs_rag.loaders import SUPPORTED_SUFFIXES, DocumentError
from docs_rag.models import AnswerDelta, IngestResult, SourcesFound

STATIC_DIR = Path(__file__).parent / "static"

log = logging.getLogger(__name__)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


def _kb(request: Request) -> KnowledgeBase:
    return request.app.state.kb


KB = Annotated[KnowledgeBase, Depends(_kb)]


def create_app(kb: KnowledgeBase | None = None, settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    max_upload_bytes = settings.max_upload_mb * 1024 * 1024

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if kb is not None:
            app.state.kb = kb
        else:
            from docs_rag.factory import build_knowledge_base

            app.state.kb = build_knowledge_base(settings)
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

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/documents")
    def list_documents(kb: KB) -> list[DocumentRecord]:
        return kb.documents()

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

    @app.post("/api/ask", response_class=EventSourceResponse)
    async def ask(body: AskRequest, kb: KB) -> AsyncIterator[ServerSentEvent]:
        """Stream an answer as server-sent events.

        Events, in order: `sources` (the numbered chunks), any number of `token`, then
        `done` with the source numbers the answer actually cited. If the model or the
        database fails mid-way, an `error` event replaces `done`.
        """
        answer: list[str] = []
        try:
            async for event in kb.astream_answer(body.question):
                match event:
                    case SourcesFound(sources):
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
        except Exception:
            log.exception("answering failed")
            yield ServerSentEvent(
                event="error",
                data="Something went wrong while answering. The server log has the details.",
            )
            return

        cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", "".join(answer))})
        yield ServerSentEvent(event="done", data={"cited": cited})

    return app
