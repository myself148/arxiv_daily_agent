import logging
from typing import Dict, List, Optional

from config import APP_CONFIG
from prompts.summary_prompt import summary_prompt
from tools.arxiv_client import fetch_latest_cv_papers
from tools.llm_utils import build_chat_model, run_with_retry
from tools.report_utils import save_report_with_archive


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class ArxivAgent:
    def __init__(self):
        self._llm = None

    @property
    def llm(self):
        if self._llm is None:
            self._llm = build_chat_model(APP_CONFIG)
        return self._llm

    def _generate_summary(self, paper: Dict) -> str:
        messages = summary_prompt.format_messages(
            title=paper["title"],
            date=paper["published_date"],
            summary=paper["summary"],
            url=paper["entry_id"],
        )
        response = run_with_retry(
            lambda: self.llm.invoke(messages),
            operation_name=f"single-agent summary for {paper['title']}",
            max_retries=APP_CONFIG.llm_max_retries,
            base_delay=APP_CONFIG.llm_retry_base_delay,
            max_delay=APP_CONFIG.llm_retry_max_delay,
        )
        return response.content.strip()

    def generate_daily_report(
        self,
        query: Optional[str] = None,
        max_results: Optional[int] = None,
    ) -> str:
        logging.info("Starting the single-agent daily paper task...")

        papers = fetch_latest_cv_papers(
            query or APP_CONFIG.arxiv_query,
            max_results if max_results is not None else APP_CONFIG.arxiv_max_results,
        )
        if not papers:
            logging.warning("No papers were found for the current query.")
            return ""

        lines: List[str] = [
            "# ArXiv Daily Agent 摘要日报",
            "",
            "> 自动抓取自 ArXiv，并由单 Agent 模式总结输出。",
            "",
            "---",
            "",
        ]

        for index, paper in enumerate(papers, start=1):
            logging.info("Processing paper %s/%s: %s", index, len(papers), paper["title"])
            try:
                summary = self._generate_summary(paper)
            except Exception as exc:  # noqa: BLE001 - keep report generation alive
                logging.error("Failed to summarize '%s': %s", paper["title"], exc)
                abstract_preview = paper["summary"][: APP_CONFIG.abstract_preview_chars]
                if len(paper["summary"]) > APP_CONFIG.abstract_preview_chars:
                    abstract_preview += "..."

                summary = (
                    "**核心痛点与动机**: 本次运行未能成功调用模型，建议直接阅读原始摘要。\n"
                    f"**创新方案与架构**: {abstract_preview}\n"
                    "**实验与效果**: 由于调用失败，本次未能稳定生成实验总结。"
                )

            lines.extend(
                [
                    f"## {index}. [{paper['title']}]({paper['entry_id']})",
                    f"- **作者**: {', '.join(paper['authors'][:3])}",
                    f"- **发布日期**: {paper['published_date']}",
                    f"- **PDF**: {paper['pdf_url']}",
                    "",
                    summary,
                    "",
                    "---",
                    "",
                ]
            )

        report = "\n".join(lines)
        latest_file, archive_file = save_report_with_archive(
            report,
            latest_path=APP_CONFIG.single_report_path,
            archive_dir=APP_CONFIG.single_archive_dir,
            prefix="single_report",
        )
        logging.info("Single-agent latest report saved to %s", latest_file)
        logging.info("Single-agent archive report saved to %s", archive_file)
        return report


def run_single_agent(
    query: Optional[str] = None,
    max_results: Optional[int] = None,
) -> str:
    agent = ArxivAgent()
    return agent.generate_daily_report(query=query, max_results=max_results)


if __name__ == "__main__":
    run_single_agent()
