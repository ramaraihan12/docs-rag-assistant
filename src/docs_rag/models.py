from dataclasses import dataclass


@dataclass(frozen=True)
class Source:
    """A retrieved chunk, numbered the way the answer cites it."""

    number: int
    file_name: str
    page: int | None
    section: str | None
    score: float
    text: str

    @property
    def location(self) -> str:
        if self.page is not None:
            return f"page {self.page}"
        return self.section or ""


@dataclass(frozen=True)
class IngestResult:
    file_name: str
    pages: int | None
    chunks: int
    skipped: bool = False


@dataclass(frozen=True)
class SourcesFound:
    sources: list[Source]


@dataclass(frozen=True)
class AnswerDelta:
    text: str


AnswerEvent = SourcesFound | AnswerDelta
