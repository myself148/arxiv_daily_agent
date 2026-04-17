import logging
import time
from typing import Dict, List

import arxiv
import fitz
import requests

from tools.text_utils import normalize_text


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

REQUEST_HEADERS = {
    "User-Agent": "arxiv-daily-agent/2.0 (+https://github.com/myself148/arxiv_daily_agent)"
}


def fetch_latest_cv_papers(
    query: str = 'cat:cs.CV AND "object detection"',
    max_results: int = 5,
) -> List[Dict]:
    """Fetch the latest papers that match the given ArXiv query."""
    logging.info("Starting ArXiv search for query: %s", query)

    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    papers: List[Dict] = []
    try:
        for result in client.results(search):
            papers.append(
                {
                    "title": result.title.replace("\n", " "),
                    "authors": [author.name for author in result.authors],
                    "published_date": result.published.strftime("%Y-%m-%d"),
                    "summary": result.summary.replace("\n", " "),
                    "entry_id": result.entry_id,
                    "pdf_url": result.pdf_url,
                }
            )
        logging.info("Fetched %s papers from ArXiv.", len(papers))
    except Exception as exc:  # noqa: BLE001 - external API errors are varied
        logging.error("Failed to fetch papers from ArXiv: %s", exc)

    return papers


def ensure_pdf_url(pdf_url: str) -> str:
    return pdf_url if pdf_url.endswith(".pdf") else f"{pdf_url}.pdf"


def download_and_parse_pdf(
    pdf_url: str,
    *,
    timeout: int = 60,
    max_retries: int = 3,
    base_delay: float = 2.0,
) -> str:
    """Download a PDF from ArXiv and extract plain text from it."""
    normalized_pdf_url = ensure_pdf_url(pdf_url)
    logging.info("Downloading and parsing PDF: %s", normalized_pdf_url)

    for attempt in range(1, max_retries + 2):
        try:
            response = requests.get(
                normalized_pdf_url,
                headers=REQUEST_HEADERS,
                timeout=timeout,
            )
            response.raise_for_status()

            with fitz.open(stream=response.content, filetype="pdf") as document:
                extracted_pages = [page.get_text("text") for page in document]

            full_text = normalize_text("\n".join(extracted_pages))
            if not full_text:
                logging.warning("PDF %s was downloaded but no text could be extracted.", normalized_pdf_url)
                return ""

            logging.info("PDF parsed successfully with %s characters extracted.", len(full_text))
            return full_text
        except Exception as exc:  # noqa: BLE001 - network and PDF parsing errors vary
            if attempt > max_retries:
                logging.error("PDF download or parsing failed after %s attempts: %s", attempt, exc)
                return ""

            delay = base_delay * (2 ** (attempt - 1))
            logging.warning(
                "PDF attempt %s/%s failed for %s: %s. Retrying in %.1f seconds.",
                attempt,
                max_retries + 1,
                normalized_pdf_url,
                exc,
                delay,
            )
            time.sleep(delay)

    return ""


if __name__ == "__main__":
    results = fetch_latest_cv_papers(max_results=2)
    for paper in results:
        print(f"[{paper['published_date']}] {paper['title']}")
