import logging
from pathlib import Path
from typing import Dict, List, Optional

from config import APP_CONFIG
from core.context import get_execution_context
from core.contracts import PipelineState, RunRequest, RunSnapshot
from core.events import EventBus
from core.tool_registry import ToolRegistry
from prompts.summary_prompt import summary_prompt
from providers.router import build_provider_router
from runtime.model_loop import ModelToolLoop
from runtime.orchestrator import PipelineOrchestrator, StageDefinition
from runtime.run_store import RunStore
from runtime.services import build_service_registry
from tools.arxiv_client import fetch_latest_cv_papers
from tools.llm_utils import build_chat_model
from tools.report_utils import save_report_with_archive


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class ArxivAgent:
    def __init__(
        self,
        *,
        services: Optional[ToolRegistry] = None,
        model_loop: Optional[ModelToolLoop] = None,
    ):
        self._llm = None
        self._services = services
        self._model_loop = model_loop

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
        return self._get_model_loop().run(
            messages,
            operation_name=f"single-agent summary for {paper['title']}",
        )

    def _get_model_loop(self) -> ModelToolLoop:
        if self._model_loop is None:
            context = get_execution_context()
            services = self._services or ToolRegistry(context.event_bus)
            self._model_loop = ModelToolLoop(
                build_provider_router(APP_CONFIG, event_bus=context.event_bus),
                services,
                max_iterations=getattr(APP_CONFIG, "model_max_tool_iterations", 8),
            )
        return self._model_loop

    def _build_services(self, event_bus: Optional[EventBus] = None) -> ToolRegistry:
        if self._services is None:
            self._services = build_service_registry(
                APP_CONFIG,
                event_bus=event_bus,
                search_papers=fetch_latest_cv_papers,
                report_writer=save_report_with_archive,
            )
        return self._services

    def _research_stage(self, state: PipelineState) -> Dict:
        services = self._build_services()
        candidate_limit = state["max_results"] * APP_CONFIG.arxiv_candidate_multiplier
        fetched_papers = services.invoke(
            "arxiv.search",
            {"query": state["query"], "max_results": candidate_limit},
        )
        filtered = services.invoke("paper.filter_new", {"papers": fetched_papers})
        papers = filtered["new_papers"][: state["max_results"]]
        logging.info(
            "Paper store contains %s papers; skipped %s previously processed candidates.",
            filtered["stored_count"],
            len(filtered["skipped_papers"]),
        )
        return {
            "papers": papers,
            "skipped_count": len(filtered["skipped_papers"]),
        }

    def _summarize_stage(self, state: PipelineState) -> Dict:
        services = self._build_services()
        summaries: List[str] = []
        paper_tags: List[List[str]] = []

        for index, paper in enumerate(state.get("papers", []), start=1):
            logging.info(
                "Processing paper %s/%s: %s",
                index,
                len(state.get("papers", [])),
                paper["title"],
            )
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

            tags = services.invoke(
                "paper.save",
                {
                    "paper": paper,
                    "generated_summary": summary,
                    "mode": "single",
                    "search_query": state["query"],
                },
            )
            summaries.append(summary)
            paper_tags.append(tags)

        return {"summaries": summaries, "paper_tags": paper_tags}

    def _report_stage(self, state: PipelineState) -> Dict:
        services = self._build_services()
        papers = state.get("papers", [])
        skipped_count = state.get("skipped_count", 0)

        if not papers:
            status = (
                "本次检索结果均已存在于论文库中，没有重复生成总结。"
                if skipped_count
                else "当前没有获取到符合条件的新论文。"
            )
            report = (
                "# ArXiv Daily Agent 摘要日报\n\n"
                f"> {status}\n\n"
                f"> 已跳过历史论文：{skipped_count}\n"
            )
            logging.info(status)
        else:
            lines: List[str] = [
                "# ArXiv Daily Agent 摘要日报",
                "",
                "> 自动抓取自 ArXiv，并由单 Agent 模式总结输出。",
                "",
                "---",
                "",
            ]
            summaries = state.get("summaries", [])
            paper_tags = state.get("paper_tags", [])
            for index, paper in enumerate(papers, start=1):
                tags = paper_tags[index - 1] if index <= len(paper_tags) else ["其他"]
                summary = summaries[index - 1] if index <= len(summaries) else ""
                lines.extend(
                    [
                        f"## {index}. [{paper['title']}]({paper['entry_id']})",
                        f"- **作者**: {', '.join(paper['authors'][:3])}",
                        f"- **发布日期**: {paper['published_date']}",
                        f"- **标签**: {', '.join(tags)}",
                        f"- **PDF**: {paper['pdf_url']}",
                        "",
                        summary,
                        "",
                        "---",
                        "",
                    ]
                )
            report = "\n".join(lines)

        saved = services.invoke(
            "report.save",
            {
                "content": report,
                "latest_path": APP_CONFIG.single_report_path,
                "archive_dir": APP_CONFIG.single_archive_dir,
                "prefix": "single_report",
                "archive_key": state.get("run_id") or None,
            },
        )
        logging.info("Single-agent latest report saved to %s", saved["latest_path"])
        logging.info("Single-agent archive report saved to %s", saved["archive_path"])
        return {"final_report": report}

    def run_pipeline(
        self,
        query: Optional[str] = None,
        max_results: Optional[int] = None,
        *,
        provider_name: Optional[str] = None,
        run_id: Optional[str] = None,
        resume_run_id: Optional[str] = None,
    ) -> RunSnapshot:
        event_bus = EventBus()
        services = self._build_services(event_bus)
        paper_store_path = Path(APP_CONFIG.paper_store_path)
        run_store_path = getattr(
            APP_CONFIG,
            "run_store_path",
            str(paper_store_path.with_name("runs.db")),
        )
        orchestrator = PipelineOrchestrator(
            run_store=RunStore(run_store_path),
            event_bus=event_bus,
            stages=[
                StageDefinition("research", self._research_stage),
                StageDefinition("review", self._summarize_stage),
                StageDefinition("report", self._report_stage),
            ],
        )
        self._services = services

        if resume_run_id:
            logging.info("Resuming single-agent run %s", resume_run_id)
            return orchestrator.execute(resume_run_id=resume_run_id)

        requested_results = (
            max_results if max_results is not None else APP_CONFIG.arxiv_max_results
        )
        request_kwargs = {
            "mode": "single",
            "query": query or APP_CONFIG.arxiv_query,
            "max_results": requested_results,
            "rag_enabled": False,
            "provider_name": provider_name,
        }
        if run_id:
            request_kwargs["run_id"] = run_id
        request = RunRequest(**request_kwargs)
        initial_state: PipelineState = {
            "papers": [],
            "skipped_count": 0,
            "summaries": [],
            "paper_tags": [],
            "final_report": "",
        }
        logging.info("Starting single-agent run %s...", request.run_id)
        return orchestrator.execute(request, initial_state=initial_state)

    def generate_daily_report(
        self,
        query: Optional[str] = None,
        max_results: Optional[int] = None,
        *,
        provider_name: Optional[str] = None,
        run_id: Optional[str] = None,
        resume_run_id: Optional[str] = None,
    ) -> str:
        snapshot = self.run_pipeline(
            query=query,
            max_results=max_results,
            provider_name=provider_name,
            run_id=run_id,
            resume_run_id=resume_run_id,
        )
        return snapshot.result or str(snapshot.state.get("final_report", ""))


def run_single_agent(
    query: Optional[str] = None,
    max_results: Optional[int] = None,
    *,
    provider_name: Optional[str] = None,
    run_id: Optional[str] = None,
    resume_run_id: Optional[str] = None,
) -> str:
    agent = ArxivAgent()
    return agent.generate_daily_report(
        query=query,
        max_results=max_results,
        provider_name=provider_name,
        run_id=run_id,
        resume_run_id=resume_run_id,
    )


if __name__ == "__main__":
    run_single_agent()
