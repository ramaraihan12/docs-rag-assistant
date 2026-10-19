"""Command line for indexing documents and asking questions about them."""

import argparse
import asyncio
import sys
from collections.abc import Iterator
from pathlib import Path

from sqlalchemy.exc import OperationalError

from docs_rag.config import get_settings
from docs_rag.errors import PROVIDER_ERRORS, provider_error_message
from docs_rag.factory import build_knowledge_base
from docs_rag.knowledge import KnowledgeBase
from docs_rag.loaders import SUPPORTED_SUFFIXES, DocumentError
from docs_rag.models import AnswerDelta, SourcesFound


def _expand(paths: list[Path]) -> Iterator[Path]:
    for path in paths:
        if path.is_dir():
            yield from sorted(p for p in path.rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES)
        else:
            yield path


def cmd_ingest(kb: KnowledgeBase, args: argparse.Namespace) -> int:
    failed = 0
    for path in _expand(args.paths):
        try:
            result = kb.ingest_file(path)
        except (DocumentError, OSError) as exc:
            print(f"skip   {path}: {exc}", file=sys.stderr)
            failed += 1
            continue
        if result.skipped:
            print(f"same   {result.file_name} (unchanged)")
        else:
            pages = f"{result.pages} pages, " if result.pages else ""
            print(f"added  {result.file_name} ({pages}{result.chunks} chunks)")
    return 1 if failed else 0


def cmd_ask(kb: KnowledgeBase, args: argparse.Namespace) -> int:
    async def run() -> None:
        sources = []
        async for event in kb.astream_answer(args.question):
            if isinstance(event, SourcesFound):
                sources = event.sources
            elif isinstance(event, AnswerDelta):
                print(event.text, end="", flush=True)
        print()
        if sources:
            print("\nSources")
            for s in sources:
                where = f", {s.location}" if s.location else ""
                print(f"  [{s.number}] {s.file_name}{where}  (score {s.score:.2f})")

    asyncio.run(run())
    return 0


def cmd_list(kb: KnowledgeBase, args: argparse.Namespace) -> int:
    docs = kb.documents()
    if not docs:
        print("No documents yet. Add some with: docs-rag ingest <path>")
    for doc in docs:
        pages = f"{doc.pages:>3} pages" if doc.pages else "         "
        print(f"{doc.file_name:<45} {pages}  {doc.chunks:>4} chunks")
    return 0


def cmd_remove(kb: KnowledgeBase, args: argparse.Namespace) -> int:
    if kb.remove(args.file_name):
        print(f"removed {args.file_name}")
        return 0
    print(f"{args.file_name} is not in the index", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="docs-rag", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="add files or folders to the index")
    ingest.add_argument("paths", nargs="+", type=Path)
    ingest.set_defaults(func=cmd_ingest)

    ask = sub.add_parser("ask", help="ask a question and stream the answer")
    ask.add_argument("question")
    ask.set_defaults(func=cmd_ask)

    sub.add_parser("list", help="show indexed documents").set_defaults(func=cmd_list)

    remove = sub.add_parser("remove", help="drop a document from the index")
    remove.add_argument("file_name")
    remove.set_defaults(func=cmd_remove)

    serve = sub.add_parser("serve", help="run the web app")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)

    if args.command == "serve":
        import uvicorn

        uvicorn.run("docs_rag.api:create_app", factory=True, host=args.host, port=args.port)
        return 0

    try:
        kb = build_knowledge_base(get_settings())
        return args.func(kb, args)
    except OperationalError as exc:
        print(f"Can't reach Postgres: {exc.orig}", file=sys.stderr)
    except PROVIDER_ERRORS as exc:
        print(provider_error_message(exc), file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
