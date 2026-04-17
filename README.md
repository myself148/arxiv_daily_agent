# ArXiv Daily Agent

基于 `LangChain`、`LangGraph` 和 OpenAI-compatible 大模型接口构建的论文分析 Agent 项目。  
项目会抓取 ArXiv 上最新的计算机视觉目标检测论文，支持两种工作模式：

- `single`：基于摘要生成简洁日报，速度更快，成本更低
- `graph`：基于多智能体流程执行检索、全文解析、分块总结与最终报告生成

当前版本重点做了三件事：

- 把模型调用、PDF 下载、文本切块等能力做成了可复用模块
- 给网络波动、接口限流、全文解析失败增加了重试和降级逻辑
- 新增历史归档机制，每次运行都会保留带时间戳的历史报告

## 核心能力

- `ArXiv 检索`：自动抓取最新论文标题、作者、摘要、发布日期和 PDF 链接
- `PDF 全文解析`：自动下载论文 PDF 并提取纯文本
- `长文本分块总结`：将论文正文切成多个片段，先局部总结再生成最终讲解
- `多智能体流程`：基于 `Researcher -> Reviewer -> Editor` 的状态机组织工作流
- `容错与降级`：模型连接失败时自动重试，全文模式失败时自动降级到摘要模式
- `历史归档`：同时保留最新报告和按时间戳归档的历史报告

## 项目结构

```text
arxiv_daily_agent/
├── agent.py                  # 单 Agent 模式
├── config.py                 # 统一配置中心
├── graph_agent.py            # 多 Agent 工作流
├── main.py                   # 统一命令行入口
├── prompts/
│   └── summary_prompt.py     # 摘要、分块总结、最终讲解 Prompt
├── tests/
│   ├── conftest.py
│   ├── test_arxiv.py
│   └── test_text_utils.py
├── tools/
│   ├── arxiv_client.py       # ArXiv 检索与 PDF 下载
│   ├── llm_utils.py          # 大模型构造与重试逻辑
│   ├── report_utils.py       # 报告保存与历史归档
│   └── text_utils.py         # 文本清洗与分块
├── requirements.txt
└── .env
```

## 环境要求

- Python `3.8+`
- 可用的 OpenAI-compatible 大模型 Key
- 已安装 `requirements.txt` 中依赖

## 安装方式

```bash
git clone https://github.com/myself148/arxiv_daily_agent.git
cd arxiv_daily_agent
pip install -r requirements.txt
```

## 环境变量

在项目根目录创建 `.env` 文件，至少配置以下内容：

```env
OPENAI_API_KEY=your_api_key
OPENAI_MODEL=glm-4-flash
OPENAI_BASE_URL=https://open.bigmodel.cn/api/paas/v4
```

常用可选配置：

```env
ARXIV_QUERY=cat:cs.CV AND "object detection"
ARXIV_MAX_RESULTS=1
MODEL_TEMPERATURE=0.3
LLM_MAX_RETRIES=4
LLM_TIMEOUT_SECONDS=90
PDF_MAX_RETRIES=3
PDF_TIMEOUT_SECONDS=60
REVIEWER_CHUNK_CHARS=4000
REVIEWER_CHUNK_OVERLAP=400
REVIEWER_MAX_CHUNKS=3

SINGLE_REPORT_PATH=daily_report.md
GRAPH_REPORT_PATH=multi_agent_report.md
SINGLE_ARCHIVE_DIR=archives/single
GRAPH_ARCHIVE_DIR=archives/graph
```

说明：

- `OPENAI_BASE_URL` 默认已经指向智谱 OpenAI-compatible 接口
- `ARXIV_MAX_RESULTS` 默认是 `1`，这是为了减少长文本请求带来的失败概率
- `REVIEWER_*` 参数控制全文分块策略，适合根据模型上下文能力继续调优

## 运行方式

统一入口是 [main.py](/E:/code/arxiv_daily_agent/main.py:1)。

运行多智能体模式：

```bash
python main.py --mode graph
```

运行单智能体模式：

```bash
python main.py --mode single
```

临时覆盖查询词和处理数量：

```bash
python main.py --mode graph --query "cat:cs.CV AND DETR" --max-results 2
```

## 输出与历史归档

每次运行都会同时生成两份文件：

- `latest`：最新报告，默认写入项目根目录
  - `daily_report.md`
  - `multi_agent_report.md`
- `archive`：历史归档，按时间戳保存
  - `archives/single/single_report_YYYYMMDD_HHMMSS.md`
  - `archives/graph/graph_report_YYYYMMDD_HHMMSS.md`

这样做的好处是：

- 日常查看时仍然可以直接打开最新报告
- 历史结果不会被下一次运行覆盖
- 后续做日报回溯、对比模型效果或生成周报会更方便

## 工作流说明

### Single 模式

1. 从 ArXiv 抓取论文摘要
2. 调用模型生成结构化摘要
3. 保存最新日报并写入历史归档

### Graph 模式

1. `Researcher`：检索论文并下载 PDF
2. `Reviewer`：按片段总结正文，再汇总成完整讲解
3. `Editor`：组装 Markdown 报告并落盘归档

## 稳定性设计

当前代码已经加入以下保护：

- PDF 下载失败会自动重试
- 模型连接错误、超时、限流会自动退避重试
- 全文模式失败后会降级到摘要模式
- 摘要模式也失败时，会输出保底说明而不是让流程直接中断

这意味着即使外部接口不稳定，流程仍然尽量保证“有结果输出”。

## 测试

运行本地单元测试：

```bash
pytest -q
```

默认会跳过联网的 ArXiv 集成测试。  
如果你想显式运行联网测试：

```bash
set RUN_NETWORK_TESTS=1
pytest -q
```

## 常见问题

### 1. 为什么会出现 `429` 或 `Connection error`？

这通常不是 LangGraph 本身的问题，而是模型服务限流、网络波动或代理链路不稳定导致的。  
当前版本已经内置重试和降级逻辑，但如果频繁出现，建议进一步：

- 降低 `ARXIV_MAX_RESULTS`
- 缩小 `REVIEWER_CHUNK_CHARS`
- 增加模型配额或切换更稳定的接口

### 2. 为什么默认只处理 1 篇论文？

因为全文模式请求体较大，处理多篇论文时更容易触发连接错误、限流或超时。默认保守配置更适合作为稳定基线。

### 3. 为什么 README 里写的是 OpenAI-compatible？

因为项目使用的是兼容 OpenAI 协议的接口客户端，但实际接入的模型可以是智谱、DeepSeek 等国产大模型，不要求必须是 OpenAI 官方模型。

## 后续可继续扩展的方向

- 增加论文缓存，避免重复下载 PDF
- 加入论文筛选 Agent，优先处理更相关的论文
- 输出结构化 JSON，方便后续做知识库或前端展示
- 支持定时运行，自动生成每日归档
- 增加前端页面或检索界面
