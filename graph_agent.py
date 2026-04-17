import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, TypedDict

from langgraph.graph import END, StateGraph

from config import APP_CONFIG
from prompts.summary_prompt import (
    abstract_review_prompt,
    final_review_prompt,
    section_summary_prompt,
)
from tools.arxiv_client import download_and_parse_pdf, fetch_latest_cv_papers
from tools.llm_utils import build_chat_model, run_with_retry
from tools.report_utils import save_report_with_archive
from tools.text_utils import split_text_into_chunks


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class AgentState(TypedDict):
    query: str
    max_results: int
    papers: List[Dict]
    full_texts: List[str]
    chunk_summaries: List[List[str]]
    review_sources: List[str]
    reviews: List[str]
    final_report: str


_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = build_chat_model(APP_CONFIG)
    return _llm


def _pause_between_requests() -> None:
    if APP_CONFIG.inter_request_delay > 0:
        time.sleep(APP_CONFIG.inter_request_delay)


def _invoke_messages(messages, operation_name: str) -> str:
    response = run_with_retry(
        lambda: _get_llm().invoke(messages),
        operation_name=operation_name,
        max_retries=APP_CONFIG.llm_max_retries,
        base_delay=APP_CONFIG.llm_retry_base_delay,
        max_delay=APP_CONFIG.llm_retry_max_delay,
    )
    return response.content.strip()


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


def _summarize_chunks(paper: Dict, full_text: str) -> Tuple[List[str], str]:
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
        chunk_summaries="\n\n".join(chunk_summaries),
    )
    final_review = _invoke_messages(final_messages, f"final review for {paper['title']}")
    return chunk_summaries, final_review


def _summarize_from_abstract(paper: Dict) -> str:
    messages = abstract_review_prompt.format_messages(
        title=paper["title"],
        summary=paper["summary"],
    )
    return _invoke_messages(messages, f"abstract fallback review for {paper['title']}")


def researcher_node(state: AgentState):
    logging.info("[Researcher] Searching ArXiv and downloading paper PDFs...")
    papers = fetch_latest_cv_papers(state["query"], state["max_results"])

    full_texts: List[str] = []
    for paper in papers:
        text = download_and_parse_pdf(
            paper["pdf_url"],
            timeout=APP_CONFIG.pdf_timeout,
            max_retries=APP_CONFIG.pdf_max_retries,
            base_delay=APP_CONFIG.pdf_retry_base_delay,
        )
        full_texts.append(text)

        if text:
            logging.info("[Researcher] PDF text ready for paper: %s", paper["title"])
        else:
            logging.warning(
                "[Researcher] Falling back to abstract-only mode because PDF extraction failed: %s",
                paper["title"],
            )

    return {"papers": papers, "full_texts": full_texts}


def reviewer_node(state: AgentState):
    logging.info("[Reviewer] Generating chunked paper reviews...")
    papers = state["papers"]
    full_texts = state["full_texts"]

    chunk_summaries: List[List[str]] = []
    review_sources: List[str] = []
    reviews: List[str] = []

    for paper, full_text in zip(papers, full_texts):
        try:
            if full_text:
                summaries, review = _summarize_chunks(paper, full_text)
                chunk_summaries.append(summaries)
                review_sources.append("全文分块解读")
            else:
                review = _summarize_from_abstract(paper)
                chunk_summaries.append([])
                review_sources.append("摘要降级模式")

            reviews.append(review)
            _pause_between_requests()
        except Exception as exc:  # noqa: BLE001 - keep the workflow alive
            logging.error("[Reviewer] Failed to process paper '%s': %s", paper["title"], exc)
            try:
                review = _summarize_from_abstract(paper)
                reviews.append(review)
                chunk_summaries.append([])
                review_sources.append("摘要二级降级模式")
            except Exception as fallback_exc:  # noqa: BLE001 - final safety net
                logging.error(
                    "[Reviewer] Abstract fallback also failed for '%s': %s",
                    paper["title"],
                    fallback_exc,
                )
                reviews.append(_build_emergency_review(paper, str(exc)))
                chunk_summaries.append([])
                review_sources.append("保底摘要模式")

    return {
        "chunk_summaries": chunk_summaries,
        "review_sources": review_sources,
        "reviews": reviews,
    }


def editor_node(state: AgentState):
    logging.info("[Editor] Building the final Markdown report...")
    papers = state["papers"]
    reviews = state["reviews"]
    review_sources = state["review_sources"]

    if not papers:
        report = "# ArXiv Daily Agent Report\n\n当前没有获取到符合条件的论文。\n"
        save_report_with_archive(
            report,
            latest_path=APP_CONFIG.graph_report_path,
            archive_dir=APP_CONFIG.graph_archive_dir,
            prefix="graph_report",
        )
        return {"final_report": report}

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# ArXiv Daily Agent 多智能体论文报告",
        "",
        f"> 生成时间：{generated_at}",
        f"> 检索条件：`{state['query']}`",
        f"> 处理论文数：{len(papers)}",
        "",
        "---",
        "",
    ]

    for index, paper in enumerate(papers, start=1):
        abstract_preview = paper["summary"][: APP_CONFIG.abstract_preview_chars]
        if len(paper["summary"]) > APP_CONFIG.abstract_preview_chars:
            abstract_preview += "..."

        lines.extend(
            [
                f"## {index}. [{paper['title']}]({paper['entry_id']})",
                f"- **作者**: {', '.join(paper['authors'][:3])}",
                f"- **发布日期**: {paper['published_date']}",
                f"- **解析模式**: {review_sources[index - 1]}",
                f"- **摘要预览**: {abstract_preview}",
                f"- **PDF**: {paper['pdf_url']}",
                "",
                reviews[index - 1],
                "",
                "---",
                "",
            ]
        )

    report = "\n".join(lines)
    latest_file, archive_file = save_report_with_archive(
        report,
        latest_path=APP_CONFIG.graph_report_path,
        archive_dir=APP_CONFIG.graph_archive_dir,
        prefix="graph_report",
    )
    logging.info("[Editor] Latest report saved to %s", latest_file)
    logging.info("[Editor] Archive report saved to %s", archive_file)
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


def run_graph_agent(
    query: Optional[str] = None,
    max_results: Optional[int] = None,
) -> str:
    app = build_graph()
    initial_state: AgentState = {
        "query": query or APP_CONFIG.arxiv_query,
        "max_results": max_results if max_results is not None else APP_CONFIG.arxiv_max_results,
        "papers": [],
        "full_texts": [],
        "chunk_summaries": [],
        "review_sources": [],
        "reviews": [],
        "final_report": "",
    }

    logging.info("Starting the multi-agent workflow...")
    result = app.invoke(initial_state)
    logging.info("Workflow completed. Latest report is %s", APP_CONFIG.graph_report_path)
    logging.info("Workflow archive directory is %s", APP_CONFIG.graph_archive_dir)
    return result["final_report"]


if __name__ == "__main__":
    run_graph_agent()
