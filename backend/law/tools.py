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
from data import law_data
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
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "law_data")
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

async def matchLawByInfo(
    infos: list[str],
    tool_context: ToolContext,
) -> str:
    """
    根据用户的案情描述（如劳动纠纷、合同问题等）推荐相关的法律法规
    params:
    infos：用户的案情描述列表
    :return: 返回所有可能适用的法律建议
    """
    agent_name = tool_context.agent_name
    history_infos = tool_context.state.get("infos", [])
    print(f"Agent {agent_name} 正在调用工具：matchLawByInfo，传入的用户信息是：{infos}，历史信息有: {history_infos}")
    history_infos.extend(infos)
    history_infos = list(set(history_infos))
    tool_context.state["infos"] = history_infos #更新案情

    history_infos_text = ",".join(history_infos)
    print(f"\n查询文本: '{history_infos_text}'")
    query_results = chromadb_instance.query2collection(
        collection=COLLECTION_NAME,
        query_documents=[history_infos_text],
        topk=2
    )
    print("查询结果:")
    law_infos = []
    for i in range(len(query_results["documents"][0])):
        doc = query_results["documents"][0][i]
        meta = query_results["metadatas"][0][i]
        distance = query_results["distances"][0][i]
        print(f"  文档: {doc[:50]}... (截断)")
        print(f"  名称: {meta.get('name')}")
        print(f"  距离: {distance:.4f}\n")
        for one_data in law_data:
            if one_data["name"] == meta["name"]:
                law_infos.append(f"{meta.get('name')}: {one_data['matches']}")
    law_infos_text = "\n".join(law_infos)
    prompt = f"""
你是一位专业的法律顾问，擅长根据用户的案情描述，从法律数据库中推荐最相关的法律法规，并判断是否需要更多信息来精确推荐。

以下是已知法律及其适用范围的描述：

```
{law_infos_text}
```

请根据用户提供的案情描述，分析并返回可能适用的法律名称，并按照相关性排序。

**输出要求：**

* 返回**多个可能适用的法律名称**，并注明是否还需要其他信息以进一步确认。
* 如果用户提供的信息足以推荐某个法律，请注明“不再需要其他信息判断”。
* 如果还不能明确判断，请指出还需询问哪些关键信息来进一步确认。

**用户描述的案情如下：**

```
{history_infos}
```

**请输出如下格式：**

```
可能的法律：
1. 法律名称A：建议进一步了解您的【劳动合同细节、工资支付凭证】，以评估是否适用
2. 法律名称B：根据您的合同纠纷描述，此法律相关性很高，可以深入了解
3. 法律名称C：建议明确您的【婚姻状态、财产分割意愿】，以提高推荐准确性
```
"""
    result = query_deepseek(prompt)
    return result

async def getLawIntroduction(law_name: str, tool_context: ToolContext) -> str:
    """
    获取某个法律的详细介绍
    params:
    law_name: 法律名称
    """
    print(f"Agent {tool_context.agent_name} 正在调用工具：getLawIntroduction，传入的法律名称是：{law_name}")
    for one_data in law_data:
        if one_data["name"] == law_name:
            return one_data["treatment_plan"]
    prompt = """
### ⚖️ Prompt：法律介绍助手

你是一位资深的法律顾问，擅长用**通俗易懂、清晰明了**的语言为用户提供**详尽的法律介绍**。你能够整合最新的法律法规，提供全面的法律解读。

请根据用户提供的法律名称，输出以下内容：

1. ✅ **法律简介**（核心内容、适用范围等，简明扼要）
2. 📜 **主要条款**（列举几个核心条款并解释）
3. ⚖️ **适用场景**（举例说明该法在哪些情况下适用）
4. 🏛️ **相关机构**（说明与该法相关的管理或执行机构）

请确保回答内容结构清晰、信息准确，能帮助用户理解法律。

"""
    prompt += f"###  law_name: {law_name}"
    response = query_deepseek(prompt)
    return response


if __name__ == '__main__':
    result = matchLawByInfo(infos=['我遇到了劳动纠纷，公司拖欠我的工资'])
    print(result)

    result = getLawIntroduction(law_name='中华人民共和国劳动法')
    print(result)