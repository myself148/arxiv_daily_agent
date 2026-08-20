from pathlib import Path
from types import SimpleNamespace

import agent as single_agent_module
import graph_agent as graph_agent_module
from tools.paper_store import PaperStore


def _paper(title: str, arxiv_id: str, summary: str):
    entry_id = f"https://arxiv.org/abs/{arxiv_id}"
    return {
        "title": title,
        "entry_id": entry_id,
        "authors": ["Test Author"],
        "published_date": "2026-08-20",
        "summary": summary,
        "pdf_url": entry_id.replace("/abs/", "/pdf/"),
    }


def test_single_agent_skips_stored_paper_and_saves_new_summary(tmp_path: Path, monkeypatch):
    database_path = tmp_path / "papers.db"
    store = PaperStore(str(database_path))
    old_paper = _paper("Stored Transformer", "2608.10001v1", "Transformer detector.")
    new_paper = _paper("Frequency Attention", "2608.10002v1", "Fourier attention detector.")
    store.save_paper(
        old_paper,
        "Already summarized.",
        mode="single",
        search_query="demo",
    )

    config = SimpleNamespace(
        arxiv_query="demo",
        arxiv_max_results=1,
        arxiv_candidate_multiplier=3,
        paper_store_path=str(database_path),
        abstract_preview_chars=100,
        single_report_path=str(tmp_path / "daily.md"),
        single_archive_dir=str(tmp_path / "archives"),
    )
    monkeypatch.setattr(single_agent_module, "APP_CONFIG", config)
    monkeypatch.setattr(
        single_agent_module,
        "fetch_latest_cv_papers",
        lambda query, limit: [old_paper, new_paper],
    )

    summarized_titles = []
    agent = single_agent_module.ArxivAgent()

    def summarize(paper):
        summarized_titles.append(paper["title"])
        return "Uses frequency-domain attention."

    monkeypatch.setattr(agent, "_generate_summary", summarize)
    report = agent.generate_daily_report()

    assert summarized_titles == [new_paper["title"]]
    assert old_paper["title"] not in report
    assert "**标签**: 注意力机制, 频域" in report
    assert PaperStore(str(database_path)).count() == 2


def test_graph_researcher_downloads_only_new_papers(tmp_path: Path, monkeypatch):
    database_path = tmp_path / "papers.db"
    store = PaperStore(str(database_path))
    old_paper = _paper("Stored CNN", "2608.20001v1", "Convolutional detector.")
    new_paper = _paper("Mamba Detector", "2608.20002v1", "State-space detection model.")
    store.save_paper(old_paper, "Stored summary.", mode="graph", search_query="demo")

    config = SimpleNamespace(
        paper_store_path=str(database_path),
        arxiv_candidate_multiplier=3,
        pdf_timeout=10,
        pdf_max_retries=0,
        pdf_retry_base_delay=0,
    )
    monkeypatch.setattr(graph_agent_module, "APP_CONFIG", config)
    monkeypatch.setattr(
        graph_agent_module,
        "fetch_latest_cv_papers",
        lambda query, limit: [old_paper, new_paper],
    )
    downloaded_urls = []

    def download(pdf_url, **kwargs):
        downloaded_urls.append(pdf_url)
        return "parsed PDF text"

    monkeypatch.setattr(graph_agent_module, "download_and_parse_pdf", download)
    result = graph_agent_module.researcher_node({"query": "demo", "max_results": 1})

    assert result["papers"] == [new_paper]
    assert result["skipped_count"] == 1
    assert downloaded_urls == [new_paper["pdf_url"]]


def test_graph_reviewer_saves_generated_review_and_tags(tmp_path: Path, monkeypatch):
    database_path = tmp_path / "papers.db"
    paper = _paper("Mamba in Frequency Domain", "2608.30001v1", "State-space model.")
    config = SimpleNamespace(
        paper_store_path=str(database_path),
        inter_request_delay=0,
    )
    monkeypatch.setattr(graph_agent_module, "APP_CONFIG", config)
    monkeypatch.setattr(
        graph_agent_module,
        "_summarize_from_abstract",
        lambda paper, retrieved_context="": "A Mamba model using Fourier features.",
    )

    result = graph_agent_module.reviewer_node(
        {
            "query": "demo",
            "papers": [paper],
            "full_texts": [""],
            "rag_enabled": False,
        }
    )

    assert result["paper_tags"] == [["Mamba", "频域"]]
    saved = PaperStore(str(database_path)).list_papers()[0]
    assert saved["generated_summary"] == "A Mamba model using Fourier features."
