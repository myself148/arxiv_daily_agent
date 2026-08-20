import os
from dataclasses import dataclass
from typing import Tuple

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


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_csv(name: str) -> Tuple[str, ...]:
    value = os.getenv(name, "")
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _normalize_base_url(url: str) -> str:
    return url.rstrip("/")


@dataclass(frozen=True)
class AppConfig:
    api_key: str
    model_name: str
    provider_name: str
    llm_fallback_models: Tuple[str, ...]
    base_url: str
    temperature: float
    llm_timeout: int
    llm_max_retries: int
    llm_retry_base_delay: float
    llm_retry_max_delay: float
    model_max_tool_iterations: int
    inter_request_delay: float
    arxiv_query: str
    arxiv_max_results: int
    arxiv_candidate_multiplier: int
    abstract_preview_chars: int
    reviewer_chunk_chars: int
    reviewer_chunk_overlap: int
    reviewer_max_chunks: int
    rag_enabled: bool
    rag_top_k: int
    rag_chunk_chars: int
    rag_chunk_overlap: int
    rag_max_chunks_per_paper: int
    rag_context_max_chars: int
    pdf_timeout: int
    pdf_max_retries: int
    pdf_retry_base_delay: float
    single_report_path: str
    graph_report_path: str
    single_archive_dir: str
    graph_archive_dir: str
    paper_store_path: str
    run_store_path: str
    scheduler_store_path: str
    scheduler_poll_seconds: float

    @classmethod
    def from_env(cls) -> "AppConfig":
        return cls(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            model_name=os.getenv("OPENAI_MODEL", "glm-4-flash"),
            provider_name=os.getenv("LLM_PROVIDER", "primary").strip() or "primary",
            llm_fallback_models=_get_csv("LLM_FALLBACK_MODELS"),
            base_url=_normalize_base_url(
                os.getenv("OPENAI_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
            ),
            temperature=_get_float("MODEL_TEMPERATURE", 0.3),
            llm_timeout=_get_int("LLM_TIMEOUT_SECONDS", 90),
            llm_max_retries=_get_int("LLM_MAX_RETRIES", 4),
            llm_retry_base_delay=_get_float("LLM_RETRY_BASE_DELAY", 2.0),
            llm_retry_max_delay=_get_float("LLM_RETRY_MAX_DELAY", 20.0),
            model_max_tool_iterations=max(_get_int("MODEL_MAX_TOOL_ITERATIONS", 8), 1),
            inter_request_delay=_get_float("INTER_REQUEST_DELAY", 1.0),
            arxiv_query=os.getenv("ARXIV_QUERY", 'cat:cs.CV AND "object detection"'),
            arxiv_max_results=_get_int("ARXIV_MAX_RESULTS", 1),
            arxiv_candidate_multiplier=max(_get_int("ARXIV_CANDIDATE_MULTIPLIER", 5), 1),
            abstract_preview_chars=_get_int("ABSTRACT_PREVIEW_CHARS", 220),
            reviewer_chunk_chars=_get_int("REVIEWER_CHUNK_CHARS", 4000),
            reviewer_chunk_overlap=_get_int("REVIEWER_CHUNK_OVERLAP", 400),
            reviewer_max_chunks=_get_int("REVIEWER_MAX_CHUNKS", 3),
            rag_enabled=_get_bool("RAG_ENABLED", True),
            rag_top_k=_get_int("RAG_TOP_K", 3),
            rag_chunk_chars=_get_int("RAG_CHUNK_CHARS", 1800),
            rag_chunk_overlap=_get_int("RAG_CHUNK_OVERLAP", 180),
            rag_max_chunks_per_paper=_get_int("RAG_MAX_CHUNKS_PER_PAPER", 24),
            rag_context_max_chars=_get_int("RAG_CONTEXT_MAX_CHARS", 4500),
            pdf_timeout=_get_int("PDF_TIMEOUT_SECONDS", 60),
            pdf_max_retries=_get_int("PDF_MAX_RETRIES", 3),
            pdf_retry_base_delay=_get_float("PDF_RETRY_BASE_DELAY", 2.0),
            single_report_path=os.getenv("SINGLE_REPORT_PATH", "daily_report.md"),
            graph_report_path=os.getenv("GRAPH_REPORT_PATH", "multi_agent_report.md"),
            single_archive_dir=os.getenv("SINGLE_ARCHIVE_DIR", "archives/single"),
            graph_archive_dir=os.getenv("GRAPH_ARCHIVE_DIR", "archives/graph"),
            paper_store_path=os.getenv("PAPER_STORE_PATH", "data/papers.db"),
            run_store_path=os.getenv("RUN_STORE_PATH", "data/runs.db"),
            scheduler_store_path=os.getenv("SCHEDULER_STORE_PATH", "data/scheduler.db"),
            scheduler_poll_seconds=max(_get_float("SCHEDULER_POLL_SECONDS", 30.0), 0.1),
        )


APP_CONFIG = AppConfig.from_env()
