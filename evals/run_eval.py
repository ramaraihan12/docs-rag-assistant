"""Measure retrieval quality on the sample documents.

Ingests samples/ into a separate table (so it never touches your real index), asks
every question in questions.jsonl, and checks where the expected chunk shows up.

    uv run python evals/run_eval.py
    uv run python evals/run_eval.py --top-k 3

Only embeddings are called, never the chat model, so a run takes seconds.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from docs_rag.models import Source

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Case:
    question: str
    file: str | None
    page: int | None = None
    section: str | None = None

    def matches(self, source: Source) -> bool:
        if source.file_name != self.file:
            return False
        if self.page is not None:
            return source.page == self.page
        return self.section is None or source.section == self.section


@dataclass(frozen=True)
class Outcome:
    case: Case
    rank: int | None  # 1-based position of the first matching chunk, None if missing
    top_score: float
    matched_score: float | None


def load_cases(path: Path) -> list[Case]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [Case(**json.loads(line)) for line in lines if line.strip()]


def evaluate(case: Case, sources: list[Source]) -> Outcome:
    top = sources[0].score if sources else 0.0
    for rank, source in enumerate(sources, start=1):
        if case.matches(source):
            return Outcome(case, rank, top, source.score)
    return Outcome(case, None, top, None)


def hit_rate(outcomes: list[Outcome], k: int | None = None) -> float:
    hits = [o for o in outcomes if o.rank is not None and (k is None or o.rank <= k)]
    return len(hits) / len(outcomes) if outcomes else 0.0


def mean_reciprocal_rank(outcomes: list[Outcome]) -> float:
    if not outcomes:
        return 0.0
    return sum(1 / o.rank for o in outcomes if o.rank) / len(outcomes)


def where(case: Case) -> str:
    if case.file is None:
        return "(not in the docs)"
    detail = f"p. {case.page}" if case.page is not None else case.section
    return f"{case.file}, {detail}"


def report(outcomes: list[Outcome], top_k: int, cutoff: float) -> str:
    answerable = [o for o in outcomes if o.case.file is not None]
    off_topic = [o for o in outcomes if o.case.file is None]
    kept = [o for o in answerable if o.matched_score is not None and o.matched_score >= cutoff]
    refused = [o for o in off_topic if o.top_score < cutoff]

    rows = ["| Question | Expected | Rank | Score |", "| --- | --- | --- | --- |"]
    for o in outcomes:
        rank = str(o.rank) if o.rank else "-"
        score = o.matched_score if o.matched_score is not None else o.top_score
        rows.append(f"| {o.case.question} | {where(o.case)} | {rank} | {score:.2f} |")

    summary = [
        "",
        f"Answerable questions: {len(answerable)}",
        f"  hit@1             {hit_rate(answerable, 1):.2f}",
        f"  hit@{top_k}             {hit_rate(answerable):.2f}",
        f"  MRR               {mean_reciprocal_rank(answerable):.2f}",
        f"  still found after the {cutoff} cutoff: {len(kept)}/{len(answerable)}",
        f"Off-topic questions with nothing above the cutoff: {len(refused)}/{len(off_topic)}",
    ]
    return "\n".join(rows + summary)


def main() -> None:
    from docs_rag.config import Settings
    from docs_rag.factory import build_knowledge_base

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--table", default="eval_chunks")
    args = parser.parse_args()

    settings = Settings().model_copy(update={"top_k": args.top_k})
    kb = build_knowledge_base(settings, table_name=args.table)
    for path in sorted((ROOT / "samples").iterdir()):
        kb.ingest_file(path)

    cases = load_cases(ROOT / "evals" / "questions.jsonl")
    outcomes = [evaluate(c, kb.retrieve(c.question, apply_cutoff=False)) for c in cases]

    print(f"embeddings: {settings.embed_provider}/{settings.embed_model}")
    print(f"query instruction: {'on' if settings.embed_query_instruction else 'off'}\n")
    print(report(outcomes, args.top_k, settings.similarity_cutoff))


if __name__ == "__main__":
    main()
