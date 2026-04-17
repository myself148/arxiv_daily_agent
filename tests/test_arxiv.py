import os

import pytest

from tools.arxiv_client import fetch_latest_cv_papers


RUN_NETWORK_TESTS = os.getenv("RUN_NETWORK_TESTS") == "1"


@pytest.mark.skipif(
    not RUN_NETWORK_TESTS,
    reason="Set RUN_NETWORK_TESTS=1 to run the live ArXiv integration test.",
)
def test_fetch_latest_cv_papers():
    papers = fetch_latest_cv_papers(max_results=3)

    assert len(papers) <= 3
    assert len(papers) > 0, "No papers were returned from ArXiv."

    first_paper = papers[0]
    expected_keys = ["title", "authors", "published_date", "summary", "entry_id", "pdf_url"]
    for key in expected_keys:
        assert key in first_paper
        assert first_paper[key] is not None
