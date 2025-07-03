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
from data import customer_service_data
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
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "customer_service_data")
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

async def query_knowledge_base(
    query: str,
    tool_context: ToolContext,
) -> str:
    """
    根据用户的问题在知识库中查询相关的解决方案
    params:
    query：用户的问题描述
    :return: 返回所有可能的解决方案建议
    """
    agent_name = tool_context.agent_name
    history_queries = tool_context.state.get("queries", [])
    print(f"Agent {agent_name} 正在调用工具：query_knowledge_base，传入的用户问题是：{query}，历史问题有: {history_queries}")
    history_queries.append(query)
    history_queries = list(set(history_queries))
    tool_context.state["queries"] = history_queries #更新问题

    history_queries_text = ",".join(history_queries)
    print(f"\n查询文本: '{history_queries_text}'")
    query_results = chromadb_instance.query2collection(
        collection=COLLECTION_NAME,
        query_documents=[history_queries_text],
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
        for one_data in customer_service_data:
            if one_data["name"] == meta["name"]:
                solution_infos.append(f"{meta.get('name')}: {one_data['question']}")
    solution_infos_text = "\n".join(solution_infos)
    prompt = f"""
你是一位专业的客户服务专家，擅长根据用户的问题，从知识库中检索最相关的解决方案，并判断是否需要更多信息来精确解答。

以下是已知问题及其解决方案的描述：

```
{solution_infos_text}
```

请根据用户提出的问题，分析并返回可能适合的解决方案名称，并按照匹配程度排序。

**输出要求：**

* 返回**多个可能的解决方案名称**，并注明是否还需要其他信息以进一步确认。
* 如果用户提供的信息足以推荐某个解决方案，请注明“不再需要其他信息判断”。
* 如果还不能明确判断，请指出还需询问哪些关键信息来进一步确认。

**用户提出的问题如下：**

```
{history_queries}
```

**请输出如下格式：**

```
可能的解决方案：
1. 解决方案A：建议进一步了解您的【订单号、购买日期】，以评估是否匹配
2. 解决方案B：根据您对登录问题的描述，此方案匹配度很高，可以深入了解
3. 解决方案C：建议明确您的【设备型号、操作系统版本】，以提高推荐准确性
```
"""
    result = query_deepseek(prompt)
    return result

async def get_solution(solution_id: str, tool_context: ToolContext) -> str:
    """
    获取某个解决方案的详细介绍
    params:
    solution_id: 解决方案的ID
    """
    print(f"Agent {tool_context.agent_name} 正在调用工具：get_solution，传入的解决方案ID是：{solution_id}")
    for one_data in customer_service_data:
        if one_data["name"] == solution_id:
            return one_data["answer"]
    prompt = """
### 🔧 Prompt：解决方案助手

你是一位资深的客户服务专家，擅长用**清晰、简洁、友好**的语言为用户提供**详细的解决方案**。你能够将复杂的技术步骤转化为易于理解的指引。

请根据用户提供的解决方案ID，输出以下内容：

1. ✅ **问题描述**（简要说明该方案针对什么问题）
2. 📝 **操作步骤**（分步列出详细的操作流程，确保每一步都清晰易懂）
3. 💡 **温馨提示**（提供一些额外的建议或注意事项，例如“操作前请备份数据”）
4. ❓ **常见问题**（列出1-2个与该解决方案相关的常见问题及其解答）

请确保回答内容结构清晰、信息准确，能有效帮助用户解决问题。如果解决方案涉及特定软件版本或环境，请一并注明。

"""
    prompt += f"###  solution_id: {solution_id}"
    response = query_deepseek(prompt)
    return response


if __name__ == '__main__':
    result = query_knowledge_base(query='我无法登录我的账户')
    print(result)

    result = get_solution(solution_id='无法登录')
    print(result)