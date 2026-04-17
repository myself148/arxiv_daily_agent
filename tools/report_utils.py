from datetime import datetime
from pathlib import Path
from typing import Tuple


def save_report_with_archive(
    content: str,
    *,
    latest_path: str,
    archive_dir: str,
    prefix: str,
) -> Tuple[Path, Path]:
    latest_file = Path(latest_path)
    latest_file.parent.mkdir(parents=True, exist_ok=True)
    latest_file.write_text(content, encoding="utf-8")

    archive_folder = Path(archive_dir)
    archive_folder.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_file = archive_folder / f"{prefix}_{timestamp}.md"
    archive_file.write_text(content, encoding="utf-8")

    return latest_file, archive_file
