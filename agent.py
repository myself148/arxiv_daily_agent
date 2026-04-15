import os
import logging
from typing import List, Dict
from dotenv import load_dotenv

# 导入 LangChain 相关组件
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import tool

# 导入我们之前写的组件
from tools.arxiv_client import fetch_latest_cv_papers
from prompts.summary_prompt import summary_prompt

# 1. 初始化配置
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# 2. 定义 Agent 工具 (用于扩展性，如果以后想让 Agent 自己决定搜什么关键词)
@tool
def arxiv_search_tool(query: str, max_results: int = 5) -> List[Dict]:
    """
    Search ArXiv for the latest research papers.
    Query should be in ArXiv format, e.g., 'cat:cs.CV AND "object detection"'.
    """
    return fetch_latest_cv_papers(query, max_results)


class ArxivAgent:
    def __init__(self, model_name="gpt-3.5-turbo", temperature=0.2):
        # 初始化 LLM，可以根据需要切换到 deepseek-chat 等
        self.llm = ChatOpenAI(
            model="glm-4-flash",  # 使用免费版模型
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url="https://open.bigmodel.cn/api/paas/v4/"  # 智谱的 OpenAI 兼容接口
        )
        # 构建处理链：Prompt -> LLM -> 解析为字符串
        self.summary_chain = summary_prompt | self.llm | StrOutputParser()

    def generate_daily_report(self, query: str, max_results: int = 3):
        logging.info("🚀 启动每日论文分析任务...")

        # 步骤 A: 获取论文数据
        papers = fetch_latest_cv_papers(query, max_results)
        if not papers:
            logging.warning("未找到相关论文。")
            return

        report_content = f"# 🚀 目标检测领域每日学术简报\n\n> 自动抓取自 ArXiv，由 AI 智能体提炼总结。\n\n---\n\n"

        # 步骤 B: 循环处理每篇论文
        for i, paper in enumerate(papers):
            logging.info(f"正在处理第 {i + 1}/{len(papers)} 篇: {paper['title']}")

            try:
                # 调用总结链
                summary = self.summary_chain.invoke({
                    "title": paper["title"],
                    "date": paper["published_date"],
                    "summary": paper["summary"],
                    "url": paper["entry_id"]
                })

                # 拼接 Markdown 格式
                report_content += f"## {i + 1}. [{paper['title']}]({paper['entry_id']})\n"
                report_content += f"- **作者**: {', '.join(paper['authors'][:3])} 等\n"
                report_content += f"- **发布日期**: {paper['published_date']}\n\n"
                report_content += f"{summary}\n\n"
                report_content += f"[查看 PDF]({paper['pdf_url']})\n\n---\n\n"

            except Exception as e:
                logging.error(f"处理论文 '{paper['title']}' 时出错: {e}")

        # 步骤 C: 保存报告
        self._save_report(report_content)

    def _save_report(self, content: str):
        filename = "daily_report.md"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        logging.info(f"✅ 任务完成！报告已生成至: {filename}")


if __name__ == "__main__":
    # 实例化并运行
    agent = ArxivAgent()
    # 针对目标检测领域进行检索
    agent.generate_daily_report(query='cat:cs.CV AND "object detection"', max_results=3)