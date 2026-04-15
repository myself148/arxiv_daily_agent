from langchain_core.prompts import ChatPromptTemplate

summary_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一位资深的计算机视觉与深度学习研究员。请阅读以下目标检测领域的最新论文摘要，并用中文提炼出核心信息。
请重点关注并按以下结构输出：
**核心痛点与动机**: (简明扼要)
**创新方案与架构**: (如网络结构的改进、损失函数的创新等)
**实验与效果**: (使用的数据集及表现)"""),
    ("user", "论文标题: {title}\n发布时间: {date}\n论文摘要: {summary}\n论文链接: {url}")
])