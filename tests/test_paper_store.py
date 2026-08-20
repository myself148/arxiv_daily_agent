from pathlib import Path

from tools.paper_store import (
    PaperStore,
    canonical_arxiv_id,
    infer_tags,
    normalize_title,
)


def _paper(title: str, entry_id: str):
    return {
        "title": title,
        "entry_id": entry_id,
        "authors": ["Ada Lovelace"],
        "published_date": "2026-08-20",
        "summary": "A Transformer with frequency-domain attention for detection.",
        "pdf_url": entry_id.replace("/abs/", "/pdf/"),
    }


def test_normalizes_titles_and_arxiv_versions():
    assert normalize_title("  Vision   Transformer  ") == "vision transformer"
    assert canonical_arxiv_id("https://arxiv.org/abs/2608.12345v2") == "2608.12345"
    assert canonical_arxiv_id("https://arxiv.org/pdf/2608.12345v1.pdf") == "2608.12345"


def test_infers_multiple_technology_tags():
    tags = infer_tags(
        "Frequency Mamba Detector",
        "A state-space model with convolution and cross-attention.",
    )

    assert tags == ["CNN", "注意力机制", "Mamba", "频域"]
    assert infer_tags("Uncategorized method") == ["其他"]


def test_store_saves_summary_tags_and_filters_duplicates(tmp_path: Path):
    store = PaperStore(str(tmp_path / "papers.db"))
    original = _paper(
        "Frequency-Aware Transformer Detector",
        "https://arxiv.org/abs/2608.12345v1",
    )

    tags = store.save_paper(
        original,
        "The method combines spectral features with attention.",
        mode="graph",
        search_query="cat:cs.CV",
    )

    assert store.count() == 1
    assert tags == ["Transformer", "注意力机制", "频域"]
    saved = store.list_papers()[0]
    assert saved["title"] == original["title"]
    assert saved["generated_summary"].startswith("The method combines")
    assert saved["tags"] == tags

    store.save_paper(
        _paper("Renamed Paper", "https://arxiv.org/abs/2608.12345v2"),
        "This must not create a duplicate row.",
        mode="single",
        search_query="cat:cs.CV",
    )
    assert store.count() == 1

    same_id_new_version = _paper(
        "A Changed Title",
        "https://arxiv.org/abs/2608.12345v2",
    )
    same_title_new_id = _paper(
        "  FREQUENCY-AWARE   TRANSFORMER DETECTOR ",
        "https://arxiv.org/abs/2608.99999v1",
    )
    genuinely_new = _paper(
        "Mamba for Tiny Objects",
        "https://arxiv.org/abs/2608.54321v1",
    )

    new_papers, skipped_papers = store.filter_new_papers(
        [same_id_new_version, same_title_new_id, genuinely_new]
    )

    assert new_papers == [genuinely_new]
    assert skipped_papers == [same_id_new_version, same_title_new_id]


def test_filter_removes_duplicates_within_one_search(tmp_path: Path):
    store = PaperStore(str(tmp_path / "papers.db"))
    paper = _paper("Duplicate Candidate", "https://arxiv.org/abs/2608.11111v1")

    new_papers, skipped_papers = store.filter_new_papers([paper, dict(paper)])

    assert new_papers == [paper]
    assert len(skipped_papers) == 1
