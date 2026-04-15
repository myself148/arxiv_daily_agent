import arxiv
import logging
from typing import List, Dict
import requests
import fitz  # PyMuPDF 的导入名


# 配置简单的日志输出，方便调试
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def fetch_latest_cv_papers(
        query: str = 'cat:cs.CV AND "object detection"',
        max_results: int = 5
) -> List[Dict]:
    """
    根据给定的查询词从 ArXiv 获取最新提交的论文。
    默认查询：计算机视觉分类下，包含“目标检测”关键词的论文。
    """
    logging.info(f"开始在 ArXiv 检索关键词: {query}")

    # 构建检索客户端
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,  # 按提交时间排序，保证是"最新"
        sort_order=arxiv.SortOrder.Descending
    )

    papers = []
    try:
        for result in client.results(search):
            # 提取核心信息，清洗成 LLM 易于理解的结构化数据
            paper_info = {
                "title": result.title.replace('\n', ' '),
                "authors": [author.name for author in result.authors],
                "published_date": result.published.strftime("%Y-%m-%d"),
                "summary": result.summary.replace('\n', ' '),  # 去除换行符，防止破坏 Prompt 格式
                "entry_id": result.entry_id,  # 论文的主页链接
                "pdf_url": result.pdf_url  # PDF 下载链接，后续 RAG 会用到
            }
            papers.append(paper_info)
        logging.info(f"成功获取 {len(papers)} 篇论文。")

    except Exception as e:
        logging.error(f"从 ArXiv 获取数据失败: {e}")

    return papers

def download_and_parse_pdf(pdf_url: str) -> str:
    """
    根据给定的 URL 下载 PDF 并提取全部纯文本。
    """
    # ArXiv 的 pdf_url 有时末尾没有 .pdf，我们加上以防万一
    if not pdf_url.endswith('.pdf'):
        pdf_url += '.pdf'

    logging.info(f"📥 正在下载并解析 PDF全文: {pdf_url} (这可能需要几秒钟...)")
    try:
        # 发起请求下载 PDF 字节流
        response = requests.get(pdf_url, stream=True, timeout=60)
        response.raise_for_status()

        # 使用 PyMuPDF 直接从内存中读取字节流，不写死到本地硬盘
        doc = fitz.open(stream=response.content, filetype="pdf")

        full_text = ""
        for page in doc:
            full_text += page.get_text("text") + "\n"

        logging.info(f"✅ 解析完成！共提取 {len(full_text)} 个字符。")
        return full_text

    except Exception as e:
        logging.error(f"❌ PDF 下载或解析失败: {e}")
        return "未能提取到全文文本。"


if __name__ == "__main__":
    # 简单的本地测试，看看能不能抓到 YOLO 或 DETR 之类的新鲜文章
    res = fetch_latest_cv_papers(max_results=2)
    for p in res:
        print(f"[{p['published_date']}] {p['title']}")