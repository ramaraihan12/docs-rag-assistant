"""Sampled, redacted tracing of /api/ask to Langfuse Cloud.

Off unless both Langfuse keys are set. One POST to the ingestion endpoint, sent after the
answer has been delivered, and every failure is swallowed.
"""

import hashlib
import logging
import os
import re
import uuid
from datetime import UTC, datetime

import httpx

from docs_rag.config import Settings

log = logging.getLogger(__name__)

APP = "docs-rag"
TIMEOUT_SECONDS = 2.0
SLOW_SECONDS = 15.0
QUESTION_CAP = 500
ANSWER_CAP = 2000
BUNDLED_FILES = frozenset(
    {
        "kabut-labs-employee-handbook.pdf",
        "kebijakan-perjalanan-dinas.md",
        "tandon-sense-t2-manual.docx",
    }
)

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_PHONE = re.compile(r"\+?\d[\d\s().-]{6,}\d")
_NUMBER = re.compile(r"\d{6,}")
_TOKEN = re.compile(r"[A-Za-z0-9_-]{24,}")


def _phone(match: re.Match[str]) -> str:
    digits = sum(c.isdigit() for c in match.group())
    return "[phone]" if digits >= 9 else match.group()


def redact(text: str, cap: int) -> str:
    text = _EMAIL.sub("[email]", text)
    text = _PHONE.sub(_phone, text)
    text = _NUMBER.sub("[number]", text)
    text = _TOKEN.sub("[token]", text)
    return text[:cap]


def is_sampled(question: str, rate: float) -> bool:
    digest = hashlib.sha256(question.strip().lower().encode()).hexdigest()
    return int(digest[:8], 16) / 2**32 < rate


def enabled(settings: Settings) -> bool:
    return bool(settings.langfuse_public_key and settings.langfuse_secret_key)


def flags_for(*, error: bool, source_count: int, answer: str, seconds: float) -> list[str]:
    flags = []
    if error:
        flags.append("error")
    elif source_count == 0:
        flags.append("no_sources")
    elif not re.search(r"\[\d+\]", answer):
        flags.append("no_citation")
    if seconds > SLOW_SECONDS:
        flags.append("slow")
    return flags


def _now(moment: datetime) -> str:
    return moment.isoformat().replace("+00:00", "Z")


def build_batch(
    *,
    question: str,
    answer: str,
    cited: list[int],
    file_names: list[str],
    flags: list[str],
    model: str,
    sample_rate: float,
    start: datetime,
    end: datetime,
) -> dict:
    output = {
        "answer": redact(answer, ANSWER_CAP),
        "cited": cited,
        "sources": [n if n in BUNDLED_FILES else "[uploaded]" for n in file_names],
    }
    question = redact(question, QUESTION_CAP)
    latency_ms = int((end - start).total_seconds() * 1000)
    metadata = {
        "app": APP,
        "model": model,
        "latency_ms": latency_ms,
        "flags": flags,
        "sample_rate": sample_rate,
    }
    level = "ERROR" if "error" in flags else "WARNING" if flags else "DEFAULT"
    trace_id = str(uuid.uuid4())
    return {
        "batch": [
            {
                "id": str(uuid.uuid4()),
                "type": "trace-create",
                "timestamp": _now(end),
                "body": {
                    "id": trace_id,
                    "timestamp": _now(start),
                    "name": f"{APP}.ask",
                    "environment": "production",
                    "release": os.environ.get("VERCEL_GIT_COMMIT_SHA") or "unknown",
                    "input": {"question": question},
                    "output": output,
                    "tags": [f"app:{APP}", *(f"flag:{f}" for f in flags)],
                    "metadata": metadata,
                },
            },
            {
                "id": str(uuid.uuid4()),
                "type": "generation-create",
                "timestamp": _now(end),
                "body": {
                    "id": str(uuid.uuid4()),
                    "traceId": trace_id,
                    "name": "answer",
                    "environment": "production",
                    "startTime": _now(start),
                    "endTime": _now(end),
                    "model": model,
                    "input": {"question": question},
                    "output": output,
                    "level": level,
                    "metadata": metadata,
                },
            },
        ]
    }


async def send_trace(
    settings: Settings,
    *,
    question: str,
    answer: str,
    cited: list[int],
    file_names: list[str],
    source_count: int,
    error: bool,
    start: datetime,
) -> None:
    """Send one trace if tracing is on and this question is in the sample. Never raises."""
    try:
        if not enabled(settings) or not is_sampled(question, settings.trace_sample_rate):
            return
        end = datetime.now(UTC)
        flags = flags_for(
            error=error,
            source_count=source_count,
            answer=answer,
            seconds=(end - start).total_seconds(),
        )
        payload = build_batch(
            question=question,
            answer=answer,
            cited=cited,
            file_names=file_names,
            flags=flags,
            model=settings.llm_model,
            sample_rate=settings.trace_sample_rate,
            start=start,
            end=end,
        )
        auth = (
            settings.langfuse_public_key.get_secret_value(),
            settings.langfuse_secret_key.get_secret_value(),
        )
        url = settings.langfuse_host.rstrip("/") + "/api/public/ingestion"
        async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
            await client.post(url, json=payload, auth=auth)
    except Exception as exc:
        log.warning("trace not sent: %s", type(exc).__name__)
