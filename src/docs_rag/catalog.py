"""Small table of ingested files.

The vector table only knows about chunks. This keeps one row per file so the UI can
list documents and so re-ingesting an unchanged file can be skipped by hash.
"""

from dataclasses import asdict, dataclass
from datetime import UTC, datetime

from sqlalchemy import (
    Column,
    DateTime,
    Engine,
    Integer,
    MetaData,
    String,
    Table,
    delete,
    insert,
    select,
)


@dataclass(frozen=True)
class DocumentRecord:
    file_name: str
    sha256: str
    pages: int | None
    chunks: int
    ingested_at: datetime


class Catalog:
    def __init__(self, engine: Engine, table_name: str) -> None:
        self._engine = engine
        metadata = MetaData()
        self._table = Table(
            table_name,
            metadata,
            Column("file_name", String(255), primary_key=True),
            Column("sha256", String(64), nullable=False),
            Column("pages", Integer, nullable=True),
            Column("chunks", Integer, nullable=False),
            Column("ingested_at", DateTime(timezone=True), nullable=False),
        )
        metadata.create_all(engine)

    def get(self, file_name: str) -> DocumentRecord | None:
        t = self._table
        with self._engine.connect() as conn:
            row = conn.execute(select(t).where(t.c.file_name == file_name)).first()
        return DocumentRecord(**row._mapping) if row else None

    def list(self) -> list[DocumentRecord]:
        t = self._table
        with self._engine.connect() as conn:
            rows = conn.execute(select(t).order_by(t.c.file_name)).all()
        return [DocumentRecord(**row._mapping) for row in rows]

    def save(self, file_name: str, sha256: str, pages: int | None, chunks: int) -> DocumentRecord:
        t = self._table
        record = DocumentRecord(file_name, sha256, pages, chunks, datetime.now(UTC))
        with self._engine.begin() as conn:
            conn.execute(delete(t).where(t.c.file_name == file_name))
            conn.execute(insert(t).values(**asdict(record)))
        return record

    def remove(self, file_name: str) -> bool:
        t = self._table
        with self._engine.begin() as conn:
            result = conn.execute(delete(t).where(t.c.file_name == file_name))
        return result.rowcount > 0
