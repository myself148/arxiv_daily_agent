from datetime import datetime
from pathlib import Path
import re
from typing import Optional, Tuple


def save_report_with_archive(
    content: str,
    *,
    latest_path: str,
    archive_dir: str,
    prefix: str,
    archive_key: Optional[str] = None,
) -> Tuple[Path, Path]:
    latest_file = Path(latest_path)
    latest_file.parent.mkdir(parents=True, exist_ok=True)
    latest_file.write_text(content, encoding="utf-8")

    archive_folder = Path(archive_dir)
    archive_folder.mkdir(parents=True, exist_ok=True)

    if archive_key:
        safe_key = re.sub(r"[^A-Za-z0-9_.-]+", "-", archive_key).strip("-.")
        if not safe_key:
            raise ValueError("archive_key must contain at least one safe character")
        archive_file = archive_folder / f"{prefix}_{safe_key}.md"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_file = archive_folder / f"{prefix}_{timestamp}.md"
    archive_file.write_text(content, encoding="utf-8")

    return latest_file, archive_file
