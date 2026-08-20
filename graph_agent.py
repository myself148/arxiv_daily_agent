import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, TypedDict

from langgraph.graph import END, StateGraph

from config import APP_CONFIG
from core.context import get_execution_context
from core.contracts import RunRequest, RunSnapshot
from core.events import EventBus
from core.tool_registry import ToolRegistry
from prompts.summary_prompt import (
    abstract_review_prompt,
    final_review_prompt,
    section_summary_prompt,
)
from providers.router import build_provider_router
from runtime.model_loop import ModelToolLoop
from runtime.orchestrator import PipelineOrchestrator, StageDefinition
from runtime.run_store import RunStore
from runtime.services import build_service_registry
from tools.arxiv_client import download_and_parse_pdf, fetch_latest_cv_papers
from tools.llm_utils import build_chat_model
from tools.report_utils import save_report_with_archive
from tools.retrieval import (
    build_retrieval_corpus,
    build_retrieval_query,
    format_retrieved_context,
    retrieve_relevant_chunks,
)
from tools.text_utils import split_text_into_chunks


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class AgentState(TypedDict):
    run_id: str
    mode: str
    query: str
    max_results: int
    rag_enabled: bool
    provider_name: Optional[str]
    skipped_count: int
    papers: List[Dict]
    full_texts: List[str]
    chunk_summaries: List[List[str]]
    retrieved_contexts: List[str]
    review_sources: List[str]
    reviews: List[str]
    paper_tags: List[List[str]]
    final_report: str


_llm = None
_model_loops: Dict[str, ModelToolLoop] = {}


def _get_llm():
    global _llm
    if _llm is None:
        _llm = build_chat_model(APP_CONFIG)
    return _llm


def _get_model_loop() -> ModelToolLoop:
    context = get_execution_context()
    cache_key = context.run_id or "default"
    if cache_key not in _model_loops:
        router = build_provider_router(APP_CONFIG, event_bus=context.event_bus)
        _model_loops[cache_key] = ModelToolLoop(
            router,
            ToolRegistry(context.event_bus),
            max_iterations=getattr(APP_CONFIG, "model_max_tool_iterations", 8),
        )
    return _model_loops[cache_key]


def _pause_between_requests() -> None:
    if APP_CONFIG.inter_request_delay > 0:
        time.sleep(APP_CONFIG.inter_request_delay)


def _invoke_messages(messages, operation_name: str) -> str:
    return _get_model_loop().run(messages, operation_name=operation_name)


def _build_services(event_bus: Optional[EventBus] = None) -> ToolRegistry:
    return build_service_registry(
        APP_CONFIG,
        event_bus=event_bus,
        search_papers=fetch_latest_cv_papers,
        extract_pdf=download_and_parse_pdf,
        report_writer=save_report_with_archive,
    )


def _prompt_context(retrieved_context: str) -> str:
    return retrieved_context or "（未启用检索增强，或未检索到可用片段。）"


def _build_retrieved_contexts(papers: List[Dict], full_texts: List[str]) -> List[str]:
    if not papers:
        return []

    chunk_size = max(APP_CONFIG.rag_chunk_chars, 1)
    overlap = max(APP_CONFIG.rag_chunk_overlap, 0)
    overlap = min(overlap, chunk_size - 1)
    max_chunks = APP_CONFIG.rag_max_chunks_per_paper
    max_chunks_per_paper = max_chunks if max_chunks > 0 else None

    try:
        corpus = build_retrieval_corpus(
            papers,
            full_texts,
            chunk_size=chunk_size,
            overlap=overlap,
            max_chunks_per_paper=max_chunks_per_paper,
        )

        contexts: List[str] = []
        for paper_index, paper in enumerate(papers):
            retrieved_chunks = retrieve_relevant_chunks(
                corpus,
                build_retrieval_query(paper),
                top_k=APP_CONFIG.rag_top_k,
                paper_index=paper_index,
            )
            contexts.append(
                format_retrieved_context(
                    retrieved_chunks,
                    max_chars=APP_CONFIG.rag_context_max_chars,
                )
            )
        return contexts
    except Exception as exc:  # noqa: BLE001 - retrieval should not break report generation
        logging.warning("[Retriever] Failed to build retrieved contexts: %s", exc)
        return [""] * len(papers)


def _format_report_retrieval_context(retrieved_context: str) -> List[str]:
    if not retrieved_context:
        return []

    safe_context = retrieved_context.replace("```", "'''")
    return [
        "",
        "<details>",
        "<summary>检索增强片段</summary>",
        "",
        "```text",
        safe_context,
        "```",
        "",
        "</details>",
        "",
    ]


def _build_emergency_review(paper: Dict, reason: str) -> str:
    preview = paper["summary"][: APP_CONFIG.abstract_preview_chars]
    if len(paper["summary"]) > APP_CONFIG.abstract_preview_chars:
        preview += "..."

    return (
        "### 1. 论文的一句话核心\n"
        "本次运行未能完成稳定的模型解读，下面提供基于摘要的保底信息。\n\n"
        "### 2. 背景知识铺垫\n"
        f"该论文主题与《{paper['title']}》相关，建议结合原文继续确认技术细节。\n\n"
        "### 3. 当前可确认的信息\n"
        f"- 摘要速读：{preview}\n"
        f"- 论文链接：{paper['entry_id']}\n\n"
        "### 4. 本次运行限制\n"
        f"- 失败原因：{reason}\n"
    )


def _summarize_chunks(
    paper: Dict,
    full_text: str,
    *,
    retrieved_context: str = "",
) -> Tuple[List[str], str]:
    chunks = split_text_into_chunks(
        full_text,
        chunk_size=APP_CONFIG.reviewer_chunk_chars,
        overlap=APP_CONFIG.reviewer_chunk_overlap,
        max_chunks=APP_CONFIG.reviewer_max_chunks,
    )
    if not chunks:
        raise ValueError("No valid chunks were produced from the PDF text.")

    chunk_summaries: List[str] = []
    total_chunks = len(chunks)

    for index, chunk in enumerate(chunks, start=1):
        messages = section_summary_prompt.format_messages(
            title=paper["title"],
            summary=paper["summary"],
            chunk_index=index,
            total_chunks=total_chunks,
            chunk_text=chunk,
        )
        chunk_summary = _invoke_messages(
            messages,
            f"section summary for paper chunk {index}/{total_chunks}",
        )
        chunk_summaries.append(chunk_summary)
        _pause_between_requests()

    final_messages = final_review_prompt.format_messages(
        title=paper["title"],
        summary=paper["summary"],
        retrieved_context=_prompt_context(retrieved_context),
        chunk_summaries="\n\n".join(chunk_summaries),
    )
    final_review = _invoke_messages(final_messages, f"final review for {paper['title']}")
    return chunk_summaries, final_review


def _summarize_from_abstract(paper: Dict, *, retrieved_context: str = "") -> str:
    messages = abstract_review_prompt.format_messages(
        title=paper["title"],
        summary=paper["summary"],
        retrieved_context=_prompt_context(retrieved_context),
    )
    return _invoke_messages(messages, f"abstract fallback review for {paper['title']}")


def researcher_node(
    state: AgentState,
    services: Optional[ToolRegistry] = None,
):
    logging.info("[Researcher] Searching ArXiv and downloading paper PDFs...")
    services = services or _build_services()
    candidate_limit = state["max_results"] * APP_CONFIG.arxiv_candidate_multiplier
    fetched_papers = services.invoke(
        "arxiv.search",
        {"query": state["query"], "max_results": candidate_limit},
    )
    filtered = services.invoke("paper.filter_new", {"papers": fetched_papers})
    new_papers = filtered["new_papers"]
    skipped_papers = filtered["skipped_papers"]
    papers = new_papers[: state["max_results"]]

    logging.info(
        "[Researcher] Paper store contains %s papers; skipped %s processed candidates; "
        "%s new papers will be reviewed.",
        filtered["stored_count"],
        len(skipped_papers),
        len(papers),
    )

    full_texts: List[str] = []
    for paper in papers:
        text = services.invoke(
            "pdf.extract",
            {"pdf_url": paper["pdf_url"]},
        )
        full_texts.append(text)

        if text:
            logging.info("[Researcher] PDF text ready for paper: %s", paper["title"])
        else:
            logging.warning(
                "[Researcher] Falling back to abstract-only mode because PDF extraction failed: %s",
                paper["title"],
            )

    return {
        "papers": papers,
        "full_texts": full_texts,
        "skipped_count": len(skipped_papers),
    }


def reviewer_node(
    state: AgentState,
    services: Optional[ToolRegistry] = None,
):
    logging.info("[Reviewer] Generating chunked paper reviews...")
    services = services or _build_services()
    papers = state["papers"]
    full_texts = state["full_texts"]
    rag_enabled = state["rag_enabled"]
    if rag_enabled:
        try:
            retrieved_contexts = services.invoke(
                "retrieval.contexts",
                {"papers": papers, "full_texts": full_texts},
            )
        except Exception as exc:  # noqa: BLE001 - retrieval must not break the report
            logging.warning("[Retriever] Failed to build retrieved contexts: %s", exc)
            retrieved_contexts = [""] * len(papers)
    else:
        retrieved_contexts = [""] * len(papers)

    chunk_summaries: List[List[str]] = []
    review_sources: List[str] = []
    reviews: List[str] = []
    paper_tags: List[List[str]] = []
    for paper_index, paper in enumerate(papers):
        full_text = full_texts[paper_index] if paper_index < len(full_texts) else ""
        retrieved_context = retrieved_contexts[paper_index] if paper_index < len(retrieved_contexts) else ""
        try:
            if full_text:
                summaries, review = _summarize_chunks(
                    paper,
                    full_text,
                    retrieved_context=retrieved_context,
                )
                chunk_summaries.append(summaries)
                review_sources.append("全文分块解读 + 检索增强" if retrieved_context else "全文分块解读")
            else:
                review = _summarize_from_abstract(paper, retrieved_context=retrieved_context)
                chunk_summaries.append([])
                review_sources.append("摘要降级模式 + 检索增强" if retrieved_context else "摘要降级模式")

            reviews.append(review)
            _pause_between_requests()
        except Exception as exc:  # noqa: BLE001 - keep the workflow alive
            logging.error("[Reviewer] Failed to process paper '%s': %s", paper["title"], exc)
            try:
                review = _summarize_from_abstract(paper, retrieved_context=retrieved_context)
                reviews.append(review)
                chunk_summaries.append([])
                review_sources.append("摘要二级降级模式 + 检索增强" if retrieved_context else "摘要二级降级模式")
            except Exception as fallback_exc:  # noqa: BLE001 - final safety net
                logging.error(
                    "[Reviewer] Abstract fallback also failed for '%s': %s",
                    paper["title"],
                    fallback_exc,
                )
                review = _build_emergency_review(paper, str(exc))
                reviews.append(review)
                chunk_summaries.append([])
                review_sources.append("保底摘要模式")

        paper_tags.append(
            services.invoke(
                "paper.save",
                {
                    "paper": paper,
                    "generated_summary": review,
                    "mode": "graph",
                    "search_query": state["query"],
                },
            )
        )

    return {
        "chunk_summaries": chunk_summaries,
        "retrieved_contexts": retrieved_contexts,
        "review_sources": review_sources,
        "reviews": reviews,
        "paper_tags": paper_tags,
    }


def editor_node(
    state: AgentState,
    services: Optional[ToolRegistry] = None,
):
    logging.info("[Editor] Building the final Markdown report...")
    services = services or _build_services()
    papers = state["papers"]
    reviews = state["reviews"]
    review_sources = state["review_sources"]
    retrieved_contexts = state["retrieved_contexts"]
    rag_enabled = state["rag_enabled"]
    paper_tags = state["paper_tags"]
    skipped_count = state["skipped_count"]

    if not papers:
        status = (
            "本次检索结果均已存在于论文库中，没有重复生成总结。"
            if skipped_count
            else "当前没有获取到符合条件的新论文。"
        )
        report = (
            "# ArXiv Daily Agent Report\n\n"
            f"{status}\n\n"
            f"> 已跳过历史论文：{skipped_count}\n"
        )
        services.invoke(
            "report.save",
            {
                "content": report,
                "latest_path": APP_CONFIG.graph_report_path,
                "archive_dir": APP_CONFIG.graph_archive_dir,
                "prefix": "graph_report",
                "archive_key": state.get("run_id") or None,
            },
        )
        return {"final_report": report}

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# ArXiv Daily Agent 多智能体论文报告",
        "",
        f"> 生成时间：{generated_at}",
        f"> 检索条件：`{state['query']}`",
        f"> 处理论文数：{len(papers)}",
        f"> 已跳过历史论文：{skipped_count}",
        f"> 检索增强：{'开启' if rag_enabled else '关闭'}",
        "",
        "---",
        "",
    ]

    for index, paper in enumerate(papers, start=1):
        abstract_preview = paper["summary"][: APP_CONFIG.abstract_preview_chars]
        if len(paper["summary"]) > APP_CONFIG.abstract_preview_chars:
            abstract_preview += "..."
        retrieved_context = retrieved_contexts[index - 1] if index - 1 < len(retrieved_contexts) else ""
        rag_status = "已注入高相关片段" if retrieved_context else ("开启但未命中片段" if rag_enabled else "关闭")
        tags = paper_tags[index - 1] if index - 1 < len(paper_tags) else ["其他"]

        lines.extend(
            [
                f"## {index}. [{paper['title']}]({paper['entry_id']})",
                f"- **作者**: {', '.join(paper['authors'][:3])}",
                f"- **发布日期**: {paper['published_date']}",
                f"- **标签**: {', '.join(tags)}",
                f"- **解析模式**: {review_sources[index - 1]}",
                f"- **检索增强**: {rag_status}",
                f"- **摘要预览**: {abstract_preview}",
                f"- **PDF**: {paper['pdf_url']}",
                "",
                reviews[index - 1],
                *_format_report_retrieval_context(retrieved_context),
                "---",
                "",
            ]
        )

    report = "\n".join(lines)
    saved = services.invoke(
        "report.save",
        {
            "content": report,
            "latest_path": APP_CONFIG.graph_report_path,
            "archive_dir": APP_CONFIG.graph_archive_dir,
            "prefix": "graph_report",
            "archive_key": state.get("run_id") or None,
        },
    )
    logging.info("[Editor] Latest report saved to %s", saved["latest_path"])
    logging.info("[Editor] Archive report saved to %s", saved["archive_path"])
    return {"final_report": report}


def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("Researcher", researcher_node)
    workflow.add_node("Reviewer", reviewer_node)
    workflow.add_node("Editor", editor_node)

    workflow.set_entry_point("Researcher")
    workflow.add_edge("Researcher", "Reviewer")
    workflow.add_edge("Reviewer", "Editor")
    workflow.add_edge("Editor", END)
    return workflow.compile()


def build_graph_orchestrator() -> PipelineOrchestrator:
    event_bus = EventBus()
    services = _build_services(event_bus)
    run_store = RunStore(getattr(APP_CONFIG, "run_store_path", "data/runs.db"))
    return PipelineOrchestrator(
        run_store=run_store,
        event_bus=event_bus,
        stages=[
            StageDefinition(
                "research",
                lambda state: researcher_node(state, services=services),
            ),
            StageDefinition(
                "review",
                lambda state: reviewer_node(state, services=services),
            ),
            StageDefinition(
                "report",
                lambda state: editor_node(state, services=services),
            ),
        ],
    )


def run_graph_pipeline(
    query: Optional[str] = None,
    max_results: Optional[int] = None,
    rag_enabled: Optional[bool] = None,
    *,
    provider_name: Optional[str] = None,
    run_id: Optional[str] = None,
    resume_run_id: Optional[str] = None,
) -> RunSnapshot:
    orchestrator = build_graph_orchestrator()
    if resume_run_id:
        logging.info("Resuming graph run %s", resume_run_id)
        try:
            return orchestrator.execute(resume_run_id=resume_run_id)
        finally:
            _model_loops.pop(resume_run_id, None)

    requested_results = max_results if max_results is not None else APP_CONFIG.arxiv_max_results
    if requested_results <= 0:
        raise ValueError("max_results must be greater than 0")

    request_kwargs = {
        "mode": "graph",
        "query": query or APP_CONFIG.arxiv_query,
        "max_results": requested_results,
        "rag_enabled": APP_CONFIG.rag_enabled if rag_enabled is None else rag_enabled,
        "provider_name": provider_name,
    }
    if run_id:
        request_kwargs["run_id"] = run_id
    request = RunRequest(**request_kwargs)
    initial_state: AgentState = {
        "run_id": request.run_id,
        "mode": "graph",
        "query": request.query,
        "max_results": requested_results,
        "rag_enabled": request.rag_enabled,
        "provider_name": provider_name,
        "skipped_count": 0,
        "papers": [],
        "full_texts": [],
        "chunk_summaries": [],
        "retrieved_contexts": [],
        "review_sources": [],
        "reviews": [],
        "paper_tags": [],
        "final_report": "",
    }

    logging.info("Starting graph run %s...", request.run_id)
    try:
        snapshot = orchestrator.execute(request, initial_state=initial_state)
    finally:
        _model_loops.pop(request.run_id, None)
    logging.info("Graph run %s completed.", snapshot.run_id)
    return snapshot


def run_graph_agent(
    query: Optional[str] = None,
    max_results: Optional[int] = None,
    rag_enabled: Optional[bool] = None,
    *,
    provider_name: Optional[str] = None,
    run_id: Optional[str] = None,
    resume_run_id: Optional[str] = None,
) -> str:
    snapshot = run_graph_pipeline(
        query=query,
        max_results=max_results,
        rag_enabled=rag_enabled,
        provider_name=provider_name,
        run_id=run_id,
        resume_run_id=resume_run_id,
    )
    return snapshot.result or str(snapshot.state.get("final_report", ""))


if __name__ == "__main__":
    run_graph_agent()
