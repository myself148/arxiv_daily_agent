import json
import re
import sqlite3
import unicodedata
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple


ARXIV_VERSION_PATTERN = re.compile(r"v\d+$", re.IGNORECASE)
WHITESPACE_PATTERN = re.compile(r"\s+")


TAG_RULES = (
    ("CNN", (r"\bcnns?\b", r"\bconvnets?\b", r"convolution", r"卷积")),
    ("Transformer", (r"transformer", r"\bdetr\b", r"vision transformer", r"\bvit\b", r"变换器")),
    ("注意力机制", (r"attention", r"注意力")),
    ("Mamba", (r"\bmamba\b", r"state[ -]space model", r"\bssm\b", r"状态空间")),
    ("频域", (r"frequency", r"spectral", r"fourier", r"wavelet", r"\bfft\b", r"\bdct\b", r"频域", r"频谱", r"傅里叶", r"小波")),
    ("YOLO", (r"\byolo(?:v\d+)?\b",)),
    ("扩散模型", (r"diffusion", r"扩散模型")),
    ("多模态", (r"multimodal", r"multi-modal", r"vision-language", r"多模态", r"视觉语言")),
    ("三维检测", (r"\b3d\b", r"three-dimensional", r"point cloud", r"lidar", r"三维", r"点云")),
    ("小目标检测", (r"small object", r"tiny object", r"小目标")),
    ("知识蒸馏", (r"distillation", r"teacher-student", r"知识蒸馏")),
)


def normalize_title(title: str) -> str:
    """Normalize a paper title for stable duplicate detection."""
    normalized = unicodedata.normalize("NFKC", title or "")
    return WHITESPACE_PATTERN.sub(" ", normalized).strip().casefold()


def canonical_arxiv_id(entry_id: str) -> str:
    """Extract an ArXiv identifier and remove its version suffix."""
    value = (entry_id or "").strip().rstrip("/")
    lowered = value.lower()
    for marker in ("/abs/", "/pdf/"):
        position = lowered.find(marker)
        if position >= 0:
            value = value[position + len(marker):]
            break
    if value.lower().endswith(".pdf"):
        value = value[:-4]
    return ARXIV_VERSION_PATTERN.sub("", value).strip().casefold()


def infer_tags(*texts: str) -> List[str]:
    """Infer deterministic, human-readable technology tags from paper content."""
    content = "\n".join(text for text in texts if text)
    tags = [
        tag
        for tag, patterns in TAG_RULES
        if any(re.search(pattern, content, flags=re.IGNORECASE) for pattern in patterns)
    ]
    return tags or ["其他"]


class PaperStore:
    """SQLite-backed history of processed papers and their generated summaries."""

    def __init__(self, database_path: str):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.database_path), timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS papers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    arxiv_id TEXT NOT NULL DEFAULT '',
                    entry_id TEXT NOT NULL DEFAULT '',
                    title TEXT NOT NULL,
                    normalized_title TEXT NOT NULL UNIQUE,
                    authors_json TEXT NOT NULL DEFAULT '[]',
                    published_date TEXT NOT NULL DEFAULT '',
                    abstract TEXT NOT NULL DEFAULT '',
                    generated_summary TEXT NOT NULL,
                    tags_json TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    search_query TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_papers_arxiv_id
                ON papers(arxiv_id);

                CREATE UNIQUE INDEX IF NOT EXISTS uq_papers_arxiv_id
                ON papers(arxiv_id)
                WHERE arxiv_id <> '';
                """
            )

    def filter_new_papers(self, papers: Sequence[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Split fetched metadata into new and already-processed papers."""
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT arxiv_id, normalized_title FROM papers"
            ).fetchall()

        known_ids = {row["arxiv_id"] for row in rows if row["arxiv_id"]}
        known_titles = {row["normalized_title"] for row in rows}
        new_papers: List[Dict] = []
        skipped_papers: List[Dict] = []

        for paper in papers:
            title_key = normalize_title(str(paper.get("title", "")))
            arxiv_key = canonical_arxiv_id(str(paper.get("entry_id", "")))
            is_duplicate = (
                not title_key
                or title_key in known_titles
                or bool(arxiv_key and arxiv_key in known_ids)
            )
            if is_duplicate:
                skipped_papers.append(paper)
                continue

            new_papers.append(paper)
            known_titles.add(title_key)
            if arxiv_key:
                known_ids.add(arxiv_key)

        return new_papers, skipped_papers

    def save_paper(
        self,
        paper: Dict,
        generated_summary: str,
        *,
        mode: str,
        search_query: str,
        tags: Optional[Sequence[str]] = None,
    ) -> List[str]:
        """Persist one processed paper and return the tags stored with it."""
        title = str(paper.get("title", "")).strip()
        title_key = normalize_title(title)
        if not title_key:
            raise ValueError("A paper title is required before it can be stored.")
        if not generated_summary.strip():
            raise ValueError("A generated summary is required before a paper can be stored.")

        stored_tags = list(tags) if tags else infer_tags(
            title,
            str(paper.get("summary", "")),
            generated_summary,
        )
        created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

        with self._connection() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO papers (
                    arxiv_id,
                    entry_id,
                    title,
                    normalized_title,
                    authors_json,
                    published_date,
                    abstract,
                    generated_summary,
                    tags_json,
                    mode,
                    search_query,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    canonical_arxiv_id(str(paper.get("entry_id", ""))),
                    str(paper.get("entry_id", "")),
                    title,
                    title_key,
                    json.dumps(paper.get("authors", []), ensure_ascii=False),
                    str(paper.get("published_date", "")),
                    str(paper.get("summary", "")),
                    generated_summary.strip(),
                    json.dumps(stored_tags, ensure_ascii=False),
                    mode,
                    search_query,
                    created_at,
                ),
            )

        return stored_tags

    def count(self) -> int:
        with self._connection() as connection:
            row = connection.execute("SELECT COUNT(*) AS total FROM papers").fetchone()
        return int(row["total"])

    def list_papers(self) -> List[Dict]:
        """Return stored papers in newest-first order, primarily for inspection/tests."""
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT title, entry_id, generated_summary, tags_json, mode, created_at
                FROM papers
                ORDER BY id DESC
                """
            ).fetchall()

        return [
            {
                "title": row["title"],
                "entry_id": row["entry_id"],
                "generated_summary": row["generated_summary"],
                "tags": json.loads(row["tags_json"]),
                "mode": row["mode"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]
