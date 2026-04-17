import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _normalize_base_url(url: str) -> str:
    return url.rstrip("/")


@dataclass(frozen=True)
class AppConfig:
    api_key: str
    model_name: str
    base_url: str
    temperature: float
    llm_timeout: int
    llm_max_retries: int
    llm_retry_base_delay: float
    llm_retry_max_delay: float
    inter_request_delay: float
    arxiv_query: str
    arxiv_max_results: int
    abstract_preview_chars: int
    reviewer_chunk_chars: int
    reviewer_chunk_overlap: int
    reviewer_max_chunks: int
    pdf_timeout: int
    pdf_max_retries: int
    pdf_retry_base_delay: float
    single_report_path: str
    graph_report_path: str
    single_archive_dir: str
    graph_archive_dir: str

    @classmethod
    def from_env(cls) -> "AppConfig":
        return cls(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            model_name=os.getenv("OPENAI_MODEL", "glm-4-flash"),
            base_url=_normalize_base_url(
                os.getenv("OPENAI_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
            ),
            temperature=_get_float("MODEL_TEMPERATURE", 0.3),
            llm_timeout=_get_int("LLM_TIMEOUT_SECONDS", 90),
            llm_max_retries=_get_int("LLM_MAX_RETRIES", 4),
            llm_retry_base_delay=_get_float("LLM_RETRY_BASE_DELAY", 2.0),
            llm_retry_max_delay=_get_float("LLM_RETRY_MAX_DELAY", 20.0),
            inter_request_delay=_get_float("INTER_REQUEST_DELAY", 1.0),
            arxiv_query=os.getenv("ARXIV_QUERY", 'cat:cs.CV AND "object detection"'),
            arxiv_max_results=_get_int("ARXIV_MAX_RESULTS", 1),
            abstract_preview_chars=_get_int("ABSTRACT_PREVIEW_CHARS", 220),
            reviewer_chunk_chars=_get_int("REVIEWER_CHUNK_CHARS", 4000),
            reviewer_chunk_overlap=_get_int("REVIEWER_CHUNK_OVERLAP", 400),
            reviewer_max_chunks=_get_int("REVIEWER_MAX_CHUNKS", 3),
            pdf_timeout=_get_int("PDF_TIMEOUT_SECONDS", 60),
            pdf_max_retries=_get_int("PDF_MAX_RETRIES", 3),
            pdf_retry_base_delay=_get_float("PDF_RETRY_BASE_DELAY", 2.0),
            single_report_path=os.getenv("SINGLE_REPORT_PATH", "daily_report.md"),
            graph_report_path=os.getenv("GRAPH_REPORT_PATH", "multi_agent_report.md"),
            single_archive_dir=os.getenv("SINGLE_ARCHIVE_DIR", "archives/single"),
            graph_archive_dir=os.getenv("GRAPH_ARCHIVE_DIR", "archives/graph"),
        )


APP_CONFIG = AppConfig.from_env()
