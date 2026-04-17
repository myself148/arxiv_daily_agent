from pathlib import Path

from tools.arxiv_client import ensure_pdf_url
from tools.report_utils import save_report_with_archive
from tools.text_utils import normalize_text, split_text_into_chunks


def test_normalize_text_collapses_extra_blank_lines():
    raw_text = "Line 1\r\n\r\n\r\nLine 2\t\tLine 3"
    normalized = normalize_text(raw_text)

    assert normalized == "Line 1\n\nLine 2 Line 3"


def test_split_text_into_chunks_respects_limits():
    text = "\n\n".join(
        [
            "Paragraph A " * 40,
            "Paragraph B " * 40,
            "Paragraph C " * 40,
        ]
    )

    chunks = split_text_into_chunks(text, chunk_size=180, overlap=20, max_chunks=3)

    assert len(chunks) == 3
    assert all(len(chunk) <= 180 for chunk in chunks)


def test_ensure_pdf_url_appends_suffix_once():
    assert ensure_pdf_url("https://arxiv.org/pdf/1234.5678") == "https://arxiv.org/pdf/1234.5678.pdf"
    assert ensure_pdf_url("https://arxiv.org/pdf/1234.5678.pdf") == "https://arxiv.org/pdf/1234.5678.pdf"


def test_save_report_with_archive_writes_latest_and_history(tmp_path: Path):
    latest_path = tmp_path / "latest.md"
    archive_dir = tmp_path / "history"

    latest_file, archive_file = save_report_with_archive(
        "# demo",
        latest_path=str(latest_path),
        archive_dir=str(archive_dir),
        prefix="graph_report",
    )

    assert latest_file.exists()
    assert archive_file.exists()
    assert latest_file.read_text(encoding="utf-8") == "# demo"
    assert archive_file.read_text(encoding="utf-8") == "# demo"
    assert archive_file.parent == archive_dir
