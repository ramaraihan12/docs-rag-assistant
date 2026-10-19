from pathlib import Path

from docs_rag.models import Source
from evals.run_eval import Case, evaluate, hit_rate, load_cases, mean_reciprocal_rank


def source(file_name: str, page: int | None = None, section: str | None = None, score=0.5):
    return Source(1, file_name, page, section, score, "text")


def test_case_matches_on_page_or_section():
    by_page = Case("q", "manual.docx", page=4)
    by_section = Case("q", "policy.md", section="Akomodasi")

    assert by_page.matches(source("manual.docx", page=4))
    assert not by_page.matches(source("manual.docx", page=5))
    assert by_section.matches(source("policy.md", section="Akomodasi"))
    assert not by_section.matches(source("other.md", section="Akomodasi"))


def test_rank_and_metrics():
    case = Case("q", "a.pdf", page=2)
    first = evaluate(case, [source("a.pdf", page=2, score=0.8)])
    third = evaluate(case, [source("b.pdf"), source("a.pdf", page=1), source("a.pdf", page=2)])
    missing = evaluate(case, [source("b.pdf")])
    outcomes = [first, third, missing]

    assert (first.rank, third.rank, missing.rank) == (1, 3, None)
    assert hit_rate(outcomes, k=1) == 1 / 3
    assert hit_rate(outcomes) == 2 / 3
    assert mean_reciprocal_rank(outcomes) == (1 + 1 / 3) / 3


def test_question_file_points_at_real_samples():
    root = Path(__file__).resolve().parent.parent
    samples = {p.name for p in (root / "samples").iterdir()}
    cases = load_cases(root / "evals" / "questions.jsonl")

    assert len(cases) >= 15
    assert {c.file for c in cases if c.file} <= samples
