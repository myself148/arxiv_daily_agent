from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

from core.events import EventBus
from core.tool_registry import ToolDefinition, ToolRegistry
from tools.arxiv_client import download_and_parse_pdf, fetch_latest_cv_papers
from tools.paper_store import PaperStore
from tools.report_utils import save_report_with_archive
from tools.retrieval import (
    build_retrieval_corpus,
    build_retrieval_query,
    format_retrieved_context,
    retrieve_relevant_chunks,
)


def build_service_registry(
    config: Any,
    *,
    event_bus: Optional[EventBus] = None,
    search_papers: Callable[..., List[Dict[str, Any]]] = fetch_latest_cv_papers,
    extract_pdf: Callable[..., str] = download_and_parse_pdf,
    report_writer: Callable[..., Any] = save_report_with_archive,
) -> ToolRegistry:
    """Register the project's external operations behind stable service names."""

    registry = ToolRegistry(event_bus)

    registry.register(
        ToolDefinition(
            name="arxiv.search",
            description="Search ArXiv for recent papers.",
            handler=lambda query, max_results: search_papers(query, max_results),
        )
    )
    registry.register(
        ToolDefinition(
            name="pdf.extract",
            description="Download a PDF and extract normalized plain text.",
            handler=lambda pdf_url: extract_pdf(
                pdf_url,
                timeout=config.pdf_timeout,
                max_retries=config.pdf_max_retries,
                base_delay=config.pdf_retry_base_delay,
            ),
        )
    )

    def filter_new(papers: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        store = PaperStore(config.paper_store_path)
        new_papers, skipped_papers = store.filter_new_papers(papers)
        return {
            "new_papers": new_papers,
            "skipped_papers": skipped_papers,
            "stored_count": store.count(),
        }

    registry.register(
        ToolDefinition(
            name="paper.filter_new",
            description="Split candidate papers into new and previously processed records.",
            handler=filter_new,
        )
    )
    registry.register(
        ToolDefinition(
            name="paper.save",
            description="Persist a processed paper and its generated summary.",
            handler=lambda paper, generated_summary, mode, search_query: PaperStore(
                config.paper_store_path
            ).save_paper(
                paper,
                generated_summary,
                mode=mode,
                search_query=search_query,
            ),
        )
    )

    def retrieve_contexts(
        papers: Sequence[Dict[str, Any]],
        full_texts: Sequence[str],
    ) -> List[str]:
        if not papers:
            return []
        chunk_size = max(config.rag_chunk_chars, 1)
        overlap = min(max(config.rag_chunk_overlap, 0), chunk_size - 1)
        max_chunks = config.rag_max_chunks_per_paper
        corpus = build_retrieval_corpus(
            papers,
            full_texts,
            chunk_size=chunk_size,
            overlap=overlap,
            max_chunks_per_paper=max_chunks if max_chunks > 0 else None,
        )
        contexts: List[str] = []
        for paper_index, paper in enumerate(papers):
            retrieved = retrieve_relevant_chunks(
                corpus,
                build_retrieval_query(paper),
                top_k=config.rag_top_k,
                paper_index=paper_index,
            )
            contexts.append(
                format_retrieved_context(
                    retrieved,
                    max_chars=config.rag_context_max_chars,
                )
            )
        return contexts

    registry.register(
        ToolDefinition(
            name="retrieval.contexts",
            description="Build local retrieval-augmented contexts for papers.",
            handler=retrieve_contexts,
        )
    )
    def save_report(
        content: str,
        latest_path: str,
        archive_dir: str,
        prefix: str,
        archive_key: Optional[str] = None,
    ) -> Dict[str, str]:
        latest_file, archive_file = report_writer(
            content,
            latest_path=latest_path,
            archive_dir=archive_dir,
            prefix=prefix,
            archive_key=archive_key,
        )
        return {
            "latest_path": str(latest_file),
            "archive_path": str(archive_file),
        }

    registry.register(
        ToolDefinition(
            name="report.save",
            description="Write the latest report and an idempotent archive copy.",
            handler=save_report,
        )
    )
    return registry
