# 🚀 ArXiv Daily Agent - 多智能体学术导师系统

基于 **LangGraph** 和 **大语言模型 (LLM)** 构建的自动化论文阅读与解析流水线。
本项目旨在每天自动追踪 ArXiv 上最新的计算机视觉（CV）目标检测论文，不仅抓取摘要，还能自动下载并通读 PDF 全文，最终以“资深 AI 博导”的口吻，为初学者输出通俗易懂的深度解析报告。

---

## ✨ 核心特性 (Features)

- 🤖 **多智能体协作架构**: 基于 LangGraph 状态机，将任务优雅地拆分为 `研究员 (Researcher)`、`审稿人/导师 (Reviewer)` 和 `主编 (Editor)` 三个独立节点。
- 📄 **PDF 全文深度解析**: 突破传统的“只看摘要”限制，集成 `PyMuPDF` 自动下载并读取论文全文数据（支持 100k+ 超长上下文模型）。
- 👨‍🏫 **“保姆级”学术教学**: 专门针对研一小白优化的 Prompt 机制，不仅提取创新点，更注重**原理解释**与**通俗化拆解**（费曼技巧）。
- ⚡ **多模型无缝切换**: 兼容 OpenAI 接口标准，可轻松配置并使用低成本/免费的国内顶级模型（如 DeepSeek、智谱 GLM-4、硅基流动等）。
- 📅 **全自动化输出**: 一键运行，自动生成排版精美的 Markdown 格式每日学术简报。

---

## 📂 项目结构 (Project Structure)

```text
arxiv_daily_agent/
├── tools/                  
│   ├── __init__.py
│   └── arxiv_client.py     # 核心工具：封装 ArXiv 检索与 PDF 下载/纯文本解析逻辑
├── prompts/                
│   └── summary_prompt.py   # 提示词工程：定义“资深 CV 博导”的设定与精读输出模板
├── tests/                  
│   └── test_arxiv.py       # 单元测试：确保数据抓取与 API 交互的稳定性
├── agent.py                # (v1.0) 单智能体基础基线版本
├── graph_agent.py          # (v2.0) 基于 LangGraph 的多智能体核心工作流
├── requirements.txt        # 项目依赖清单
└── .env                    # (需手动创建) 存放 API Keys 的环境变量文件
