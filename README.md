# ArXiv Daily Agent

当前版本：`v4.0`

基于 `LangChain`、`LangGraph` 和 OpenAI-compatible 大模型接口构建的论文分析 Agent 项目。  
项目会抓取 ArXiv 上最新的计算机视觉目标检测论文，支持两种工作模式：

- `single`：基于摘要生成简洁日报，速度更快，成本更低
- `graph`：基于多智能体流程执行检索、全文解析、分块总结与最终报告生成

`v4.0` 完成了运行时架构升级：

- 将任务 Pipeline 编排与模型—工具调用循环解耦
- 建立论文、运行状态、阶段结果和运行事件的稳定协议
- 引入可选择、可重试、可超时、可降级的 Provider 路由
- 将 ArXiv、PDF、检索、论文存储和报告存储注册为独立服务
- 新增阶段 checkpoint、失败恢复和独立持久化 Scheduler

## 核心能力

- `ArXiv 检索`：自动抓取最新论文标题、作者、摘要、发布日期和 PDF 链接
- `PDF 全文解析`：自动下载论文 PDF 并提取纯文本
- `长文本分块总结`：将论文正文切成多个片段，先局部总结再生成最终讲解
- `检索增强生成`：从论文全文或摘要中召回高相关片段，注入最终讲解并在报告中保留证据片段
- `论文库去重`：将标题、生成总结和技术标签写入 SQLite，后续运行自动跳过同一论文
- `多智能体流程`：基于 `Researcher -> Reviewer -> Editor` 的状态机组织工作流
- `容错与降级`：模型连接失败时自动重试，全文模式失败时自动降级到摘要模式
- `历史归档`：同时保留最新报告和按 `run_id` 幂等归档的历史报告

## 项目结构

```text
arxiv_daily_agent/
├── agent.py                  # 单 Agent 模式
├── config.py                 # 统一配置中心
├── core/                     # 稳定领域类型、运行事件、执行上下文、工具注册表
│   ├── contracts.py
│   ├── context.py
│   ├── events.py
│   └── tool_registry.py
├── graph_agent.py            # 多 Agent 工作流
├── main.py                   # 统一命令行入口
├── providers/                # 模型 Provider 接口、OpenAI-compatible 实现与降级路由
├── runtime/                  # Pipeline 编排、模型工具循环、Run 存储、服务与调度器
│   ├── model_loop.py
│   ├── orchestrator.py
│   ├── run_store.py
│   ├── scheduler.py
│   └── services.py
├── scheduler.py              # 独立调度服务命令行入口
├── .env.example              # 不含真实凭据的配置模板
├── prompts/
│   └── summary_prompt.py     # 摘要、分块总结、最终讲解 Prompt
├── tests/
│   ├── conftest.py
│   ├── test_arxiv.py
│   ├── test_paper_store.py
│   ├── test_agent_store_integration.py
│   ├── test_model_runtime.py
│   ├── test_retrieval.py
│   ├── test_run_orchestrator.py
│   ├── test_scheduler.py
│   └── test_text_utils.py
├── tools/
│   ├── arxiv_client.py       # ArXiv 检索与 PDF 下载
│   ├── llm_utils.py          # 大模型构造与重试逻辑
│   ├── paper_store.py        # SQLite 论文库、去重与自动标签
│   ├── report_utils.py       # 报告保存与历史归档
│   └── text_utils.py         # 文本清洗与分块
├── requirements.txt
└── .env
```

## 运行时架构

业务 Pipeline 与模型—工具调用循环现在是两条独立边界：

```text
CLI / Scheduler
      │
      ▼
PipelineOrchestrator ── checkpoint/event ──> RunStore (SQLite)
      │
      ├── arxiv.search / pdf.extract / retrieval.contexts
      ├── paper.filter_new / paper.save
      └── report.save

Review stage ──> ModelToolLoop ──> ProviderRouter
                                  ├── primary provider (retry/timeout)
                                  └── fallback providers
```

- `RunRequest`、`PipelineState`、`StageResult`、`RunEvent` 是跨模块的稳定协议。
- Pipeline 每完成一个阶段就写入 checkpoint；失败后使用原 `run_id` 从下一未完成阶段继续。
- Provider 路由负责模型选择和降级，具体 OpenAI-compatible 实现负责超时与退避重试。
- ArXiv、PDF、检索、论文存储和报告存储均通过命名服务注册，不再由编排器直接依赖实现细节。
- Scheduler 直接调用 Python Pipeline，并保存 cron、下次执行时间、对应 `run_id` 和错误信息，不启动不可追踪的 shell 脚本。

## 环境要求

- Python `3.8+`
- 可用的 OpenAI-compatible 大模型 Key
- 已安装 `requirements.txt` 中依赖

## 安装方式

```bash
git clone https://github.com/myself148/arxiv_daily_agent.git
cd arxiv_daily_agent
pip install -r requirements.txt
copy .env.example .env
```

Linux/macOS 使用 `cp .env.example .env`。随后编辑 `.env`，填入自己的模型服务凭据。

## 环境变量

在项目根目录创建 `.env` 文件，至少配置以下内容：

```env
OPENAI_API_KEY=your_api_key
OPENAI_MODEL=glm-4-flash
OPENAI_BASE_URL=https://open.bigmodel.cn/api/paas/v4
LLM_PROVIDER=primary
LLM_FALLBACK_MODELS=glm-4-flash-250414,glm-4-air
```

仓库中的 `.env.example` 只包含占位值，可以安全复制；真实 `.env` 已被 Git 忽略。

常用可选配置：

```env
ARXIV_QUERY=cat:cs.CV AND "object detection"
ARXIV_MAX_RESULTS=1
ARXIV_CANDIDATE_MULTIPLIER=5
MODEL_TEMPERATURE=0.3
LLM_MAX_RETRIES=4
LLM_TIMEOUT_SECONDS=90
MODEL_MAX_TOOL_ITERATIONS=8
PDF_MAX_RETRIES=3
PDF_TIMEOUT_SECONDS=60
REVIEWER_CHUNK_CHARS=4000
REVIEWER_CHUNK_OVERLAP=400
REVIEWER_MAX_CHUNKS=3
RAG_ENABLED=true
RAG_TOP_K=3
RAG_CHUNK_CHARS=1800
RAG_CHUNK_OVERLAP=180
RAG_MAX_CHUNKS_PER_PAPER=24
RAG_CONTEXT_MAX_CHARS=4500

SINGLE_REPORT_PATH=daily_report.md
GRAPH_REPORT_PATH=multi_agent_report.md
SINGLE_ARCHIVE_DIR=archives/single
GRAPH_ARCHIVE_DIR=archives/graph
PAPER_STORE_PATH=data/papers.db
RUN_STORE_PATH=data/runs.db
SCHEDULER_STORE_PATH=data/scheduler.db
SCHEDULER_POLL_SECONDS=30
```

说明：

- `OPENAI_BASE_URL` 默认指向智谱 OpenAI-compatible 接口
- `ARXIV_MAX_RESULTS` 默认是 `1`，是为了降低长文本请求失败概率
- `ARXIV_CANDIDATE_MULTIPLIER` 控制候选元数据的多取倍数，避免最新结果都已入库时找不到新论文
- `PAPER_STORE_PATH` 是本地 SQLite 论文库路径
- `RUN_STORE_PATH` 保存运行状态、阶段 checkpoint 和事件历史
- `LLM_FALLBACK_MODELS` 按顺序声明主模型失败后的备用模型；留空即不跨模型降级
- `REVIEWER_*` 参数控制全文分块策略，适合根据模型上下文能力继续调优

## 运行方式

统一入口是 `main.py`。

运行多智能体模式：

```bash
python main.py --mode graph
```

临时关闭检索增强：

```bash
python main.py --mode graph --no-rag
```

运行单智能体模式：

```bash
python main.py --mode single
```

临时覆盖查询词和处理数量：

```bash
python main.py --mode graph --query "cat:cs.CV AND DETR" --max-results 2
```

查看运行历史并恢复失败任务：

```bash
python main.py --list-runs
python main.py --resume-run RUN_ID
```

临时选择 Provider：

```bash
python main.py --mode graph --provider primary
```

## 独立调度器

调度器使用标准五段 cron（分、时、日、月、周），支持列表、范围和步长：

```bash
python scheduler.py add --name daily-report --cron "0 8 * * *" --mode graph
python scheduler.py list
python scheduler.py run-pending
python scheduler.py serve
```

可使用 `enable`、`disable`、`remove` 加调度 ID 管理任务。`serve` 是独立长驻进程；外部 Cron 或系统计划任务只需负责确保该服务启动，而不再直接触发业务脚本。

## 输出与历史归档

每次运行都会同时生成两份文件：

- `latest`：最新报告，默认写入项目根目录
  - `daily_report.md`
  - `multi_agent_report.md`
- `archive`：历史归档，按稳定 `run_id` 保存；恢复同一次运行时会更新同一份归档
  - `archives/single/single_report_RUN_ID.md`
  - `archives/graph/graph_report_RUN_ID.md`

这样做的好处是：

- 日常查看时仍然可以直接打开最新报告
- 历史结果不会被下一次运行覆盖
- 后续做日报回溯、效果对比或周报汇总会更方便

示例输出可参考 [单 Agent 日报](ex_daily_report.md) 和 [Graph 多智能体报告](ex_multi_agent_report.md)。示例仅包含公开论文信息，不包含本地凭据或运行数据库。

## 论文库存储与去重

默认论文库位于 `data/papers.db`。Single 和 Graph 模式共用同一个库，并执行以下流程：

1. 从 ArXiv 多取一批候选元数据
2. 按去掉版本号的 ArXiv ID 和规范化标题查询论文库
3. 跳过已有论文，只下载并总结指定数量的新论文
4. 将标题、作者、原摘要、生成总结、运行模式、查询条件、标签和入库时间写入论文库

标签由标题、原摘要和生成总结中的关键词自动推断。目前内置 `CNN`、`Transformer`、`注意力机制`、`Mamba`、`频域`、`YOLO`、`扩散模型`、`多模态`、`三维检测`、`小目标检测` 和 `知识蒸馏` 等类别；没有命中时使用 `其他`。一篇论文可以拥有多个标签，标签也会显示在日报中。

可以使用 SQLite 命令行查看历史记录：

```bash
sqlite3 data/papers.db "SELECT title, tags_json, created_at FROM papers ORDER BY id DESC;"
```

数据库文件默认不会提交到 Git。备份或删除 `data/papers.db` 会分别保留或重置本地去重历史。

## 工作流说明

### Single 模式

1. 从 ArXiv 抓取论文摘要，并通过论文库过滤历史结果
2. 调用模型生成结构化摘要
3. 自动打标签，将论文和总结写入论文库
4. 保存最新日报并写入历史归档

### Graph 模式

1. `Researcher`：检索论文、过滤论文库中的历史结果并下载新论文 PDF
2. `Reviewer`：按片段总结正文、检索高相关证据片段，并将总结和标签写入论文库
3. `Editor`：组装 Markdown 报告并落盘归档

检索增强默认只使用本地已抓取的论文内容，不需要额外的向量数据库或 embedding 服务。
如果 PDF 下载失败，检索模块会自动退回到摘要内容；如果检索失败，流程会继续生成普通报告。

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

## 版本说明

- `v1.0`：单 Agent MVP，完成 ArXiv 摘要抓取与日报生成
- `v2.0`：引入 LangGraph，多智能体工作流支持 PDF 全文解析
- `v3.0`：强化稳定性、长文本处理、统一入口、历史归档与测试体系
- `v4.0`：运行时解耦、Provider 路由、服务注册、可恢复 Run 与独立 Scheduler

## 隐私与本地数据

以下内容默认不会提交到 Git：

- `.env` 及其他本地环境变量文件
- `data/` 中的论文库、运行 checkpoint、事件历史和调度状态
- `daily_report.md`、`multi_agent_report.md` 与 `archives/` 中的本地生成报告
- IDE 工作区、缓存、日志和 Python 字节码

提交代码前仍建议执行一次密钥扫描，并确认示例报告中没有加入私有论文、内部链接或个人备注。

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
- 增加调度任务并发租约与分布式执行支持
- 增加前端页面或检索界面
