"""Turn files into LlamaIndex documents that remember where their text came from.

Every document carries `file_name`, `page` (1-based, or None when the format has no
pages) and `section` (the nearest heading, when we can tell). The splitter copies that
metadata onto each chunk, which is what lets an answer cite "manual.docx, page 4".
"""

import re
from collections.abc import Iterator
from pathlib import Path

from docx import Document as open_docx
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph
from llama_index.core import Document
from pypdf import PdfReader

SUPPORTED_SUFFIXES = {".pdf", ".md", ".markdown", ".txt", ".docx"}

_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


class UnsupportedFileType(ValueError):
    pass


def load_file(path: Path, file_name: str | None = None) -> list[Document]:
    """Load one file. `file_name` overrides the name stored in metadata (used for uploads)."""
    name = file_name or path.name
    suffix = Path(name).suffix.lower()
    match suffix:
        case ".pdf":
            parts = _pdf_pages(path)
        case ".docx":
            parts = _docx_parts(path)
        case ".md" | ".markdown":
            parts = _markdown_sections(path.read_text(encoding="utf-8"))
        case ".txt":
            parts = iter([(None, None, path.read_text(encoding="utf-8"))])
        case _:
            raise UnsupportedFileType(f"{suffix or 'files without an extension'} not supported")

    docs = []
    for page, section, text in parts:
        text = text.strip()
        # Skip empty parts and headings that are directly followed by another heading.
        if not text or text.lstrip("#").strip() == section:
            continue
        doc = Document(text=text, metadata={"file_name": name, "page": page, "section": section})
        # Page numbers are noise for the embedding; file name and heading are useful signal.
        doc.excluded_embed_metadata_keys = ["page"]
        docs.append(doc)
    return docs


def _pdf_pages(path: Path) -> Iterator[tuple[int, None, str]]:
    reader = PdfReader(path)
    for number, page in enumerate(reader.pages, start=1):
        yield number, None, page.extract_text() or ""


def _markdown_sections(text: str) -> Iterator[tuple[None, str | None, str]]:
    section: str | None = None
    lines: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        heading = None if in_fence else _MD_HEADING.match(line)
        if heading:
            yield None, section, "\n".join(lines)
            section, lines = heading.group(2), []
        lines.append(line)
    yield None, section, "\n".join(lines)


def _docx_parts(path: Path) -> Iterator[tuple[int, str | None, str]]:
    """Walk the body in order, cutting a new part at every page break or heading.

    .docx files don't store page numbers. Word records where pages ended the last time
    it laid the file out (w:lastRenderedPageBreak); files written by other tools usually
    only have explicit page breaks. We use the rendered breaks when present, otherwise
    the explicit ones, so page numbers are approximate for long flowing documents.
    """
    doc = open_docx(str(path))
    body = doc.element.body
    if body.find(".//" + qn("w:lastRenderedPageBreak")) is not None:
        is_break = _is_rendered_break
    else:
        is_break = _is_explicit_break

    page, section = 1, None
    buffer: list[str] = []

    for block in doc.iter_inner_content():
        if isinstance(block, Table):
            buffer.append(_table_text(block))
            continue

        if _is_heading(block) and block.text.strip():
            yield page, section, "\n".join(buffer)
            section, buffer = block.text.strip(), []

        line: list[str] = []
        for el in block._p.iter(qn("w:t"), qn("w:tab"), qn("w:br"), qn("w:lastRenderedPageBreak")):
            if is_break(el):
                buffer.append("".join(line))
                yield page, section, "\n".join(buffer)
                page, buffer, line = page + 1, [], []
            elif el.tag == qn("w:t"):
                line.append(el.text or "")
            elif el.tag == qn("w:tab"):
                line.append("\t")
            elif el.tag == qn("w:br") and el.get(qn("w:type")) in (None, "textWrapping"):
                line.append("\n")
        buffer.append("".join(line))

    yield page, section, "\n".join(buffer)


def _is_rendered_break(el) -> bool:
    return el.tag == qn("w:lastRenderedPageBreak")


def _is_explicit_break(el) -> bool:
    return el.tag == qn("w:br") and el.get(qn("w:type")) == "page"


def _is_heading(paragraph: Paragraph) -> bool:
    style = paragraph.style.name if paragraph.style is not None else ""
    return style.startswith("Heading") or style == "Title"


def _table_text(table: Table) -> str:
    rows = []
    for row in table.rows:
        cells = [cell.text.strip() for cell in row.cells]
        rows.append(" | ".join(cells))
    return "\n".join(rows)
