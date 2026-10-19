import re
from collections.abc import Sequence

from llama_index.core.llms import ChatMessage, MessageRole

from docs_rag.models import Source

SYSTEM_PROMPT = """You answer questions about the user's own documents, using only the numbered \
sources in the message.

Rules:
- Cite every fact with the number of the source it came from in square brackets, like [2]. \
Use more than one, like [1][3], when a sentence relies on several sources.
- If the sources do not answer the question, say that the documents don't cover it. \
Do not fill gaps with outside knowledge.
- Reply in the same language as the question, even when the sources are in another language.
- Keep it short: a few sentences or a small list."""

NO_ANSWER = {
    "en": (
        "I couldn't find anything about that in your documents, so I'd rather not guess. "
        "Try rephrasing, or upload the document that covers it."
    ),
    "id": (
        "Saya tidak menemukan informasi itu di dokumen Anda, jadi saya tidak mau menebak. "
        "Coba ubah pertanyaannya, atau unggah dokumen yang membahasnya."
    ),
}

# Short function words that almost never show up in English questions.
_INDONESIAN_WORDS = {
    "apa", "apakah", "bagaimana", "berapa", "kapan", "siapa", "mengapa", "kenapa",
    "dimana", "mana", "yang", "saya", "aku", "kami", "untuk", "dengan", "tidak",
    "bisa", "boleh", "harus", "cara", "jika", "kalau", "ada", "dan", "atau", "dari",
    "ke", "di", "itu", "ini", "sudah", "belum", "berapakah", "bagaimanakah",
}  # fmt: skip


def looks_indonesian(text: str) -> bool:
    return any(word in _INDONESIAN_WORDS for word in re.findall(r"[a-z]+", text.lower()))


def no_answer_message(question: str) -> str:
    return NO_ANSWER["id" if looks_indonesian(question) else "en"]


def format_sources(sources: Sequence[Source]) -> str:
    blocks = []
    for source in sources:
        header = f"[{source.number}] {source.file_name}"
        if source.location:
            header += f", {source.location}"
        blocks.append(f"{header}\n{source.text}")
    return "\n\n".join(blocks)


def build_messages(question: str, sources: Sequence[Source]) -> list[ChatMessage]:
    user = f"Sources:\n\n{format_sources(sources)}\n\nQuestion: {question}"
    return [
        ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT),
        ChatMessage(role=MessageRole.USER, content=user),
    ]
