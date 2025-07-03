#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/6/20 10:02
# @File  : tools.py.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  :
import os
from litellm import completion
from google.adk.tools import ToolContext
from google.adk.tools.agent_tool import AgentTool
from data import business_data
import time
from datetime import datetime
import random
from dotenv import load_dotenv
from DecisionAgent.embedding_utils import EmbeddingModel,ChromaDB
# import litellm
# litellm._turn_on_debug()
# 加载环境变量
load_dotenv()

TOOL_MODEL_API_BASE = os.environ["TOOL_MODEL_API_BASE"]
TOOL_MODEL_API_KEY = os.environ["TOOL_MODEL_API_KEY"]
TOOL_MODEL_NAME = os.environ["TOOL_MODEL_NAME"]
TOOL_MODEL_PROVIDER = os.environ["TOOL_MODEL_PROVIDER"]
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "business_data")
print(f"使用的向量表是: {COLLECTION_NAME}")
embedder = EmbeddingModel()
chromadb_instance = ChromaDB(embedder=embedder)

def query_deepseek(prompt):
    try:
        response = completion(provider=TOOL_MODEL_PROVIDER, model=TOOL_MODEL_NAME,
                                   messages=[{"content": prompt, "role": "user"}],
                                   api_base=TOOL_MODEL_API_BASE,
                                   api_key=TOOL_MODEL_API_KEY
                              )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

async def analyzeBusinessProblem(
    problems: list[str],
    tool_context: ToolContext,
) -> str:
    """
    根据用户的商业问题（如市场进入、竞争分析、营销策略等）提供初步的解决方案。
    params:
    problems：用户的商业问题描述列表
    :return: 返回所有可能的解决方案建议
    """
    agent_name = tool_context.agent_name
    history_problems = tool_context.state.get("problems", [])
    print(f"Agent {agent_name} 正在调用工具：analyzeBusinessProblem，传入的用户问题是：{problems}，历史问题有: {history_problems}")
    history_problems.extend(problems)
    history_problems = list(set(history_problems))
    tool_context.state["problems"] = history_problems #更新问题

    history_problems_text = ",".join(history_problems)
    print(f"\n查询文本: '{history_problems_text}'")
    query_results = chromadb_instance.query2collection(
        collection=COLLECTION_NAME,
        query_documents=[history_problems_text],
        topk=2
    )
    print("查询结果:")
    solution_infos = []
    for i in range(len(query_results["documents"][0])):
        doc = query_results["documents"][0][i]
        meta = query_results["metadatas"][0][i]
        distance = query_results["distances"][0][i]
        print(f"  文档: {doc[:50]}... (截断)")
        print(f"  名称: {meta.get('name')}")
        print(f"  距离: {distance:.4f}\n")
        for one_data in business_data:
            if one_data["solution_name"] == meta["name"]:
                solution_infos.append(f"{meta.get('name')}: {one_data['problem_description']}")
    solution_infos_text = "\n".join(solution_infos)
    prompt = f"""
你是一位顶级的商业分析师，擅长根据客户的商业问题，从解决方案数据库中匹配最合适的策略，并判断是否需要更多信息来精确推荐。

以下是已知的解决方案及其适用场景的描述：

```
{solution_infos_text}
```

请根据客户提供的商业问题，分析并返回可能适合的解决方案名称，并按照匹配程度排序。

**输出要求：**

* 返回**多个可能的解决方案**，并注明是否还需要其他信息以进一步确认。
* 如果客户提供的信息足以推荐某个解决方案，请注明“不再需要其他信息判断”。
* 如果还不能明确判断，请指出还需询问哪些关键信息来进一步确认。

**客户描述的商业问题如下：**

```
{history_problems}
```

**请输出如下格式：**

```
可能的解决方案：
1. 解决方案A：建议进一步了解您的【目标客户群体、预算规模】，以评估是否匹配
2. 解决方案B：根据您对市场扩张的需求，此方案匹配度很高，可以深入了解
3. 解决方案C：建议明确您的【品牌定位、竞争对手情况】，以提高推荐准确性
```
"""
    result = query_deepseek(prompt)
    return result

async def getIndustryReport(industry_name: str, tool_context: ToolContext) -> str:
    """
    获取某个行业的详细分析报告
    params:
    industry_name: 行业名称
    """
    print(f"Agent {tool_context.agent_name} 正在调用工具：getIndustryReport，传入的行业名称是：{industry_name}")
    for one_data in business_data:
        if one_data["solution_name"] == industry_name:
            return one_data["detailed_solution"]
    prompt = """
### 📈 Prompt：行业报告生成器

你是一位资深的行业分析师，能够提供**深入、全面、数据驱动**的行业分析报告。你精通市场研究、竞争格局分析和未来趋势预测。

请根据用户提供的行业名称，输出以下内容：

1.  ✅ **市场概览**（市场规模、增长率、主要驱动因素）
2.  📊 **竞争格局**（主要竞争对手、市场份额、竞争优势）
3.  💡 **机会与挑战**（新兴机会、潜在风险、成功关键因素）
4.  🚀 **未来趋势**（技术创新、消费者行为变化、法规政策影响）

请确保报告内容逻辑清晰、数据翔实、观点独到，能为商业决策提供有力支持。如果该行业有特定的进入壁垒（如技术、资金、牌照），请一并注明。

"""
    prompt += f"###  industry_name: {industry_name}"
    response = query_deepseek(prompt)
    return response


if __name__ == '__main__':
    result = analyzeBusinessProblem(problems=['我计划开一家咖啡馆，但市场竞争激烈，我该如何定位和制定营销策略？'])
    print(result)

    result = getIndustryReport(industry_name='市场定位与营销策略')
    print(result)