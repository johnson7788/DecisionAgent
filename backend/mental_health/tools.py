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
from data import mental_health_data
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
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "mental_health_data")
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

async def diagnoseMentalHealth(
    symptoms: list[str],
    tool_context: ToolContext,
) -> str:
    """
    根据用户的心理困扰或情绪问题（如焦虑、失眠、情绪低落等）诊断可能的心理健康问题
    params:
    symptoms：用户的心理困扰或情绪问题描述列表
    :return: 返回所有可能的心理健康问题建议
    """
    agent_name = tool_context.agent_name
    history_symptoms = tool_context.state.get("symptoms", [])
    print(f"Agent {agent_name} 正在调用工具：diagnoseMentalHealth，传入的用户信息是：{symptoms}，历史信息有: {history_symptoms}")
    history_symptoms.extend(symptoms)
    history_symptoms = list(set(history_symptoms))
    tool_context.state["symptoms"] = history_symptoms #更新症状

    history_symptoms_text = ",".join(history_symptoms)
    print(f"\n查询文本: '{history_symptoms_text}'")
    query_results = chromadb_instance.query2collection(
        collection=COLLECTION_NAME,
        query_documents=[history_symptoms_text],
        topk=2
    )
    print("查询结果:")
    problem_infos = []
    for i in range(len(query_results["documents"][0])):
        doc = query_results["documents"][0][i]
        meta = query_results["metadatas"][0][i]
        distance = query_results["distances"][0][i]
        print(f"  文档: {doc[:50]}... (截断)")
        print(f"  名称: {meta.get('name')}")
        print(f"  距离: {distance:.4f}\n")
        for one_data in mental_health_data:
            if one_data["name"] == meta["name"]:
                problem_infos.append(f"{meta.get('name')}: {one_data['symptoms']}")
    problem_infos_text = "\n".join(problem_infos)
    prompt = f"""
你是一位专业的心理健康咨询师，擅长根据用户的心理困扰或情绪问题，从心理健康问题数据库中诊断最可能的问题，并判断是否需要更多信息来精确诊断。

以下是已知心理健康问题及其症状的描述：

```
{problem_infos_text}
```

请根据用户提供的心理困扰或情绪问题，分析并返回可能适合的心理健康问题名称，并按照匹配程度排序。

**输出要求：**

* 返回**多个可能的心理健康问题名称**，并注明是否还需要其他信息以进一步确认。
* 如果用户提供的信息足以诊断某个问题，请注明“不再需要其他信息判断”。
* 如果还不能明确判断，请指出还需询问哪些关键信息来进一步确认。

**用户描述的心理困扰或情绪问题如下：**

```
{history_symptoms}
```

**请输出如下格式：**

```
可能的问题：
1. 焦虑症：建议进一步了解你的【具体焦虑情境、持续时间】，以评估是否匹配
2. 抑郁症：根据你对情绪低落的描述，此问题匹配度很高，可以深入了解
3. 睡眠障碍：建议明确你对【入睡困难、睡眠质量】，以提高诊断准确性
```
"""
    result = query_deepseek(prompt)
    return result

async def provideCopingStrategies(problem_name: str, tool_context: ToolContext) -> str:
    """
    获取某个心理健康问题的详细应对策略
    params:
    problem_name: 心理健康问题名称
    """
    print(f"Agent {tool_context.agent_name} 正在调用工具：provideCopingStrategies，传入的心理健康问题名称是：{problem_name}")
    for one_data in mental_health_data:
        if one_data["name"] == problem_name:
            return one_data["treatment_plan"]
    prompt = """
### 🧠 Prompt：心理健康应对策略助手

你是一位专业的心理健康咨询师，擅长用**通俗易懂、富有同理心**的语言为用户提供**详尽的心理健康应对策略**。你能够整合最新的心理学研究和实践经验，提供全面的问题解读和实用建议。

请根据用户提供的心理健康问题名称，输出以下内容：

1. ✅ **问题简介**（核心症状、影响等，简明扼要）
2. 💡 **应对策略**（具体的、可操作的建议，如放松技巧、认知调整、寻求专业帮助等）
3. 🤝 **支持资源**（推荐相关的书籍、社区、线上平台或专业机构）
4. ⚠️ **注意事项**（何时需要寻求紧急帮助、避免的误区等）

请确保回答内容结构清晰、信息丰富，能帮助用户更好地理解和应对自己的心理问题。如果该问题有特殊的干预方法（如认知行为疗法、药物治疗），请一并注明。

"""
    prompt += f"###  problem_name: {problem_name}"
    response = query_deepseek(prompt)
    return response


if __name__ == '__main__':
    result = diagnoseMentalHealth(symptoms=['我最近感到很焦虑，晚上睡不着觉，怎么办？'])
    print(result)

    result = provideCopingStrategies(problem_name='焦虑症')
    print(result)
