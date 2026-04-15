#  ArXiv Daily Agent - 多智能体论文每日阅读系统

基于 **LangGraph** 和 **大语言模型 (LLM)** 构建的自动化论文阅读与解析流水线。
本项目旨在每天自动追踪 ArXiv 上最新的计算机视觉（CV）目标检测论文，不仅抓取摘要，还能自动下载并通读 PDF 全文，最终以“资深 AI 博导”的口吻，为初学者输出通俗易懂的深度解析报告。

---

##  核心特性 (Features)

-  **多智能体协作架构**: 基于 LangGraph 状态机，将任务优雅地拆分为 `研究员 (Researcher)`、`审稿人/导师 (Reviewer)` 和 `主编 (Editor)` 三个独立节点。
-  **PDF 全文深度解析**: 突破传统的“只看摘要”限制，集成 `PyMuPDF` 自动下载并读取论文全文数据（支持 100k+ 超长上下文模型）。
-  **“保姆级”学术教学**: 专门针对小白优化的 Prompt 机制，不仅提取创新点，更注重**原理解释**与**通俗化拆解**（费曼技巧）。
-  **多模型无缝切换**: 兼容 OpenAI 接口标准，可轻松配置并使用低成本/免费的国内顶级模型（如 DeepSeek、智谱 GLM-4、硅基流动等）。
-  **全自动化输出**: 一键运行，自动生成排版精美的 Markdown 格式每日学术简报。

---

##  项目结构 (Project Structure)

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
```

##  快速开始 (Quick Start)
### 1. 克隆项目与安装环境
推荐使用 Conda 或 venv 创建虚拟环境（Python >= 3.8）。

```
git clone [https://github.com/myself148/arxiv_daily_agent.git](https://github.com/myself148/arxiv_daily_agent.git)
cd arxiv_daily_agent
pip install -r requirements.txt
```

### 2. 配置环境变量
在项目根目录创建一个 .env 文件，并填入你的大模型 API Key（以使用 DeepSeek 或 智谱 为例）：

```
# 请将下方替换为你真实的 API Key
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxx
(注意：请在 graph_agent.py 中修改对应提供商的 base_url)
```

### 3. 运行多智能体工作流

```
python graph_agent.py
```

终端将清晰展示状态机流转日志：

[研究员] 检索并下载论文全文。

[博导] 通读全文并生成深度教学解析。

[主编] 汇总整理并排版。

运行结束后，根目录下会自动生成一份 multi_agent_report.md 文件。

## 报告输出样例 (Example Output)
### 1. 论文的一句话核心
这篇论文解决了传统 YOLO 模型在红外弱小目标检测中容易丢失特征的痛点，提出了一种全新的跨模态特征对齐模块。

### 2. 背景知识铺垫
在讲创新点之前，你需要知道什么是 OBB（定向边界框）损失...

### 3. 核心创新点深度拆解

它叫什么：GCDLoss (广义交叉分布损失)

它是怎么运作的：你可以把它想象成一个...

##  版本历史 (Changelog)
v2.0 (Current): 引入 LangGraph 升级为多智能体工作流，集成 PyMuPDF 实现 PDF 全文下载与深度解析。

v1.0: 构建单智能体基础检索流水线，实现 ArXiv 摘要提取与自动化 Markdown 报告生成。

