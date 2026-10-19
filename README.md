# docs-rag-assistant

[![ci](https://github.com/ramaraihan12/docs-rag-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/ramaraihan12/docs-rag-assistant/actions/workflows/ci.yml)

Ask questions about your own PDFs, Word files and Markdown notes, and get short answers that point to the file and page they came from. It runs on a local Ollama model by default, so it costs nothing, and it works in English and Bahasa Indonesia.

![The web UI answering a question with two cited sources](docs/screenshot.png)

From the command line, asking in Indonesian about an English PDF:

```
$ uv run docs-rag ask "Berapa hari cuti tahunan yang saya dapat?"
Saya mendapatkan 14 hari kerja cuti tahunan per tahun [1].

Sources
  [1] kabut-labs-employee-handbook.pdf, page 3  (score 0.62)
  [2] kabut-labs-employee-handbook.pdf, page 4  (score 0.56)
  [3] kebijakan-perjalanan-dinas.md, Kebijakan Perjalanan Dinas dan Penggantian Biaya  (score 0.51)
  [4] kabut-labs-employee-handbook.pdf, page 7  (score 0.51)
  [5] kabut-labs-employee-handbook.pdf, page 2  (score 0.50)
```

## How it works

- Each PDF page, each DOCX page and each Markdown section becomes a document tagged with `file_name`, `page` and `section`. The splitter copies those tags onto every chunk, which is what makes "page 3" possible.
- Chunks are embedded with `qwen3-embedding:0.6b` and stored in Postgres with pgvector through LlamaIndex. A small `chunks_files` table keeps a hash per file, so re-ingesting an unchanged file is skipped and a changed one replaces its old chunks.
- A question pulls the top 5 chunks and drops anything under 0.4 cosine similarity. If nothing is left, it says the documents don't cover it and never calls the LLM, so there is nothing to make up.
- Otherwise the chunks go to the model numbered `[1]..[n]` and the answer streams back over SSE: a `sources` event, `token` events, then `done` with the numbers the answer actually cited.
- The page shows cited sources as cards. Clicking a `[n]` opens the quoted chunk.

The API behind the page: `GET /api/documents`, `POST /api/documents` (multipart upload), `DELETE /api/documents/{file_name}`, `POST /api/ask` (SSE) and `GET /api/health`.

## Stack

Python 3.12 with uv, FastAPI, LlamaIndex 0.14 (pgvector store, Ollama / OpenAI / Gemini / Anthropic integrations), Postgres 17 + pgvector, pypdf, python-docx, plain HTML/CSS/JS, pytest and ruff.

## Run it

You need [uv](https://docs.astral.sh/uv/), Docker and [Ollama](https://ollama.com).

```bash
ollama pull qwen3.5:9b
ollama pull qwen3-embedding:0.6b
cp .env.example .env
```

Everything in Docker (Ollama stays on the host):

```bash
docker compose up -d --build
docker compose exec app docs-rag ingest samples
```

Or only Postgres in Docker and the app on your machine:

```bash
docker compose up -d db
uv sync
uv run docs-rag ingest samples
uv run docs-rag serve
```

Either way, open http://127.0.0.1:8000. Compose publishes Postgres on 5432, so stop any local Postgres first or change the port in `docker-compose.yml` and `DATABASE_URL`.

The CLI also has `docs-rag list`, `docs-rag remove <file_name>` and `docs-rag ask "<question>"`. Settings live in `.env.example`. The ones you are most likely to touch:

| Variable | Default | |
| --- | --- | --- |
| `LLM_PROVIDER` / `LLM_MODEL` | `ollama` / `qwen3.5:9b` | also `openai`, `gemini`, `anthropic` with the matching API key |
| `EMBED_PROVIDER` / `EMBED_MODEL` | `ollama` / `qwen3-embedding:0.6b` | a new embedding model needs a new `TABLE_NAME` and a re-ingest |
| `SIMILARITY_CUTOFF` | `0.4` | below this a chunk is not used |

`qwen3.5:9b` is the Qwen3.5 9B model from the Ollama library. My laptop has a 6 GB RTX 3060, so I run a smaller IQ4_XS quant of the same model that I imported from a GGUF file (`LLM_MODEL=qwen3.5-9b:iq4xs`). The example output, the timings and the screenshot above all come from that build. An answer took 35 to 90 seconds in my runs, partly because only one model fits in VRAM, so Ollama swaps the embedding model out for the chat model on every question.

## Tests

```bash
uv run pytest
```

The unit and API tests use LlamaIndex's `MockLLM`, a tiny word-hashing embedding and an in-memory vector store, so they need no network. The Postgres tests run when `TEST_DATABASE_URL` points at a database with pgvector, for example `TEST_DATABASE_URL=postgresql://postgres:postgres@localhost:5432/rag_assistant uv run pytest`. CI runs both, plus `ruff check` and `ruff format --check`.

## Retrieval eval

`samples/` has three made-up documents for a company called Kabut Labs: an employee handbook (PDF, English), a travel policy (Markdown, Indonesian) and a product manual (DOCX, English). `evals/questions.jsonl` has 16 questions with the page or section that answers them, plus 3 that the documents can't answer.

```bash
uv run python evals/run_eval.py
```

It ingests the samples into its own `eval_chunks` table and only calls the embedding model. Result with Ollama `qwen3-embedding:0.6b`:

| | |
| --- | --- |
| hit@1 | 1.00 (16/16) |
| hit@5 | 1.00 |
| MRR | 1.00 |
| answerable questions still above the 0.4 cutoff | 16/16 |
| off-topic questions with nothing above the cutoff | 3/3 |

Lowest score for a right answer was 0.43 ("How much is the yearly learning budget?"), highest for an off-topic question 0.31 (a fried rice recipe in Indonesian). That gap is where the 0.4 cutoff comes from. The script prints the full per-question table.

## Notes

- The eval is small and I wrote the questions after writing the documents, so a perfect score mostly says nothing is broken. Real documents with overlapping topics will do worse, and the cutoff is tuned for this embedding model only.
- `.docx` files don't store page numbers. The loader uses the page breaks Word saved the last time it laid out the file, or explicit page breaks if there are none, so DOCX pages can be off for long flowing text.
- Scanned PDFs without a text layer are rejected with a clear error. OCR would be the next thing I'd add, then a reranker before the cutoff.
- There is no auth and one shared index. It's meant for a laptop or a small internal box, not a multi-tenant service.

## License

MIT
