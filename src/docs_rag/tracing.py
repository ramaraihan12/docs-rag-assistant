"""Sampled, redacted tracing of /api/ask to Langfuse Cloud.

Off unless both Langfuse keys are set. One OTLP/HTTP POST (the legacy ingestion API is gone on
Langfuse Cloud), sent after the answer has been delivered, and every failure is swallowed.
"""

import asyncio
import hashlib
import json
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
# Only numbers that start with + or 0 count as phones, so dates, times and long amounts that
# begin with another digit are left alone. U+2010 to U+2015 are the unicode hyphens.
_PHONE = re.compile(r"(?<![\d+])(?<!\d[.,])[+0]\d[\d\s().\u2010-\u2015-]{6,}\d")
_NUMBER = re.compile(r"\d{6,}")
_TOKEN = re.compile(r"[A-Za-z0-9_-]{24,}")


def _phone(match: re.Match[str]) -> str:
    digits = sum(c.isdigit() for c in match.group())
    return "[phone]" if digits >= 9 else match.group()


def redact(text: str, cap: int) -> str:
    # Tokens go before phones and numbers: a hex key with a run of digits would otherwise be
    # split by the number rule and leave pieces the token rule can no longer match.
    # Redact a bounded slice, then cut: the cut can't leave half a number behind and the regexes
    # never run over a huge string.
    text = text[: cap * 2]
    text = _EMAIL.sub("[email]", text)
    text = _TOKEN.sub("[token]", text)
    text = _PHONE.sub(_phone, text)
    text = _NUMBER.sub("[number]", text)
    return text[:cap]


def is_sampled(question: str, rate: float) -> bool:
    digest = hashlib.sha256(question.strip().lower().encode()).hexdigest()
    return int(digest[:8], 16) / 2**32 < rate


def enabled(settings: Settings) -> bool:
    return bool(settings.langfuse_public_key and settings.langfuse_secret_key)


def flags_for(*, error: bool, source_count: int, cited: list[int], seconds: float) -> list[str]:
    flags = []
    if error:
        flags.append("error")
    elif source_count == 0:
        flags.append("no_sources")
    elif not cited:
        flags.append("no_citation")
    if seconds > SLOW_SECONDS:
        flags.append("slow")
    return flags


def _nanos(moment: datetime) -> str:
    return str(int(moment.timestamp() * 1000) * 1_000_000)


def _text(key: str, value: str) -> dict:
    return {"key": key, "value": {"stringValue": value}}


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
    """One OTLP/JSON export holding a single root generation span for the ask."""
    output = {
        "answer": redact(answer, ANSWER_CAP),
        "cited": cited,
        "sources": [n if n in BUNDLED_FILES else "[uploaded]" for n in file_names],
    }
    latency_ms = int((end - start).total_seconds() * 1000)
    level = "ERROR" if "error" in flags else "WARNING" if flags else "DEFAULT"
    attributes = [
        _text("langfuse.observation.type", "generation"),
        _text("langfuse.trace.name", f"{APP}.ask"),
        _text("langfuse.environment", "production"),
        _text("langfuse.release", os.environ.get("VERCEL_GIT_COMMIT_SHA") or "unknown"),
        {
            "key": "langfuse.trace.tags",
            "value": {
                "arrayValue": {
                    "values": [
                        {"stringValue": t} for t in [f"app:{APP}", *(f"flag:{f}" for f in flags)]
                    ]
                }
            },
        },
        _text("langfuse.observation.level", level),
        _text("langfuse.observation.model.name", model),
        _text(
            "langfuse.observation.input", json.dumps({"question": redact(question, QUESTION_CAP)})
        ),
        _text("langfuse.observation.output", json.dumps(output)),
        _text("langfuse.observation.metadata.app", APP),
        _text("langfuse.observation.metadata.model", model),
        _text("langfuse.observation.metadata.latency_ms", str(latency_ms)),
        _text("langfuse.observation.metadata.flags", json.dumps(flags)),
        _text("langfuse.observation.metadata.sample_rate", str(sample_rate)),
    ]
    span = {
        "traceId": uuid.uuid4().hex,
        "spanId": uuid.uuid4().hex[:16],
        "name": "answer",
        "kind": 1,
        "startTimeUnixNano": _nanos(start),
        "endTimeUnixNano": _nanos(end),
        "attributes": attributes,
        "status": {"code": 2 if "error" in flags else 1},
    }
    return {
        "resourceSpans": [
            {
                "resource": {"attributes": [_text("service.name", APP)]},
                "scopeSpans": [{"scope": {"name": APP}, "spans": [span]}],
            }
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
            cited=cited,
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
        url = settings.langfuse_host.rstrip("/") + "/api/public/otel/v1/traces"

        async def post() -> httpx.Response:
            async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
                return await client.post(
                    url, json=payload, auth=auth, headers={"x-langfuse-ingestion-version": "4"}
                )

        # The httpx timeout is per phase; this is the real total deadline.
        resp = await asyncio.wait_for(post(), TIMEOUT_SECONDS)
        if not 200 <= resp.status_code < 300:
            # Status only: the body could echo the payload.
            log.warning("trace rejected: status %s", resp.status_code)
    except Exception as exc:
        log.warning("trace not sent: %s", type(exc).__name__)
