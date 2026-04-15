import os
import logging
from typing import TypedDict, List, Dict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

from tools.arxiv_client import fetch_latest_cv_papers, download_and_parse_pdf
from prompts.summary_prompt import detailed_tutor_prompt

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# 1. 定义整个团队共享的“状态笔记本” (State)
class AgentState(TypedDict):
    query: str
    max_results: int
    papers: List[Dict]  # 研究员抓取的论文
    full_texts: List[str] # 存储每一篇论文的完整文本
    reviews: List[str]  # 审稿人写的评价
    final_report: str  # 主编输出的最终报告


# 初始化语言模型
llm = ChatOpenAI(
    model="glm-4-flash",  # 目前使用了智谱免费版
    temperature=0.3,
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)


# 2. 定义节点 1：研究员 (Researcher)
def researcher_node(state: AgentState):
    logging.info("🧑‍🔬 [研究员] 正在 ArXiv 检索最新论文...")
    papers = fetch_latest_cv_papers(state["query"], state["max_results"])

    full_texts = []
    # 下载全文并让大模型处理极度耗时，建议初期 max_results 设为 1
    for paper in papers:
        text = download_and_parse_pdf(paper["pdf_url"])
        full_texts.append(text)

    return {"papers": papers, "full_texts": full_texts}

# 3. 定义节点 2：导师
def reviewer_node(state: AgentState):
    logging.info("👨‍🏫 [导] 正在通读全文，为您准备研一小白专属的深度讲解...")
    papers = state["papers"]
    full_texts = state["full_texts"]
    reviews = []

    # 注意：确保你的 LLM 支持长上下文（如 deepseek-chat 或 glm-4-flash）
    for i in range(len(papers)):
        msg = detailed_tutor_prompt.format_messages(
            title=papers[i]["title"],
            full_text=full_texts[i][:60000]  # 截断一下防止个别论文长得离谱，通常6万字够覆盖核心了
        )
        response = llm.invoke(msg)
        reviews.append(response.content)

    return {"reviews": reviews}

# 4. 定义节点 3：主编 (Editor)
def editor_node(state: AgentState):
    logging.info("✍️ [主编] 正在汇总整理最终的学术简报...")
    papers = state["papers"]
    reviews = state["reviews"]

    report = "# 🚀 目标检测每日多视角学术简报\n\n"

    for i in range(len(papers)):
        report += f"## {i + 1}. [{papers[i]['title']}]({papers[i]['entry_id']})\n"
        report += f"- **作者**: {', '.join(papers[i]['authors'][:3])}\n"
        report += f"- **摘要速读**: {papers[i]['summary'][:200]}...\n"
        report += f"- ** 审稿人锐评**: *{reviews[i]}*\n\n---\n"

    # 保存到本地
    with open("multi_agent_report.md", "w", encoding="utf-8") as f:
        f.write(report)

    return {"final_report": report}


# 5. 构建图 (Graph) 流水线
def build_graph():
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("Researcher", researcher_node)
    workflow.add_node("Reviewer", reviewer_node)
    workflow.add_node("Editor", editor_node)

    # 定义执行顺序：开始 -> 研究员 -> 审稿人 -> 主编 -> 结束
    workflow.set_entry_point("Researcher")
    workflow.add_edge("Researcher", "Reviewer")
    workflow.add_edge("Reviewer", "Editor")
    workflow.add_edge("Editor", END)

    # 编译成可执行程序
    return workflow.compile()


if __name__ == "__main__":
    app = build_graph()

    # 初始化输入状态
    initial_state = {
        "query": 'cat:cs.CV AND "object detection"',
        "max_results": 2,
        "papers": [],
        "reviews": [],
        "final_report": ""
    }

    # 运行图
    print("🎬 开始执行多智能体工作流...\n")
    app.invoke(initial_state)
    print("\n 多智能体协作完成！请查看 multi_agent_report.md")