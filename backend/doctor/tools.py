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
from data import doctor_data
import time
from datetime import datetime
import random
from DecisionAgent.embedding_utils import EmbeddingModel,ChromaDB
# import litellm
# litellm._turn_on_debug()

TOOL_MODEL_API_BASE = os.environ["TOOL_MODEL_API_BASE"]
TOOL_MODEL_API_KEY = os.environ["TOOL_MODEL_API_KEY"]
TOOL_MODEL_NAME = os.environ["TOOL_MODEL_NAME"]
TOOL_MODEL_PROVIDER = os.environ["TOOL_MODEL_PROVIDER"]
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "disease_data")

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

async def matchDiseaseBySymptoms(
    symptoms: list[str],
    tool_context: ToolContext,
) -> str:
    """
    根据疾病的症状搜索可能的疾病
    params:
    symptoms：症状的列表
    :return: 返回所有可能的疾病
    """
    # 需要使用疾病症状和数据库进行向量匹配，然后使用大模型进行判断，是否是一个或者多个疾病。返回给前端疾病名称和症状。
    agent_name = tool_context.agent_name
    history_symptoms = tool_context.state.get("symptoms", [])
    print(f"Agent {agent_name} 正在调用工具：matchDiseaseBySymptoms，传入的症状是：{symptoms}，历史症状有: {history_symptoms}")
    history_symptoms.extend(symptoms)
    history_symptoms = list(set(history_symptoms))
    tool_context.state["symptoms"] = history_symptoms #更新症状

    history_symptoms_text = ",".join(history_symptoms)
    print(f"\n查询文本: '{history_symptoms_text}'")
    query_results = chromadb_instance.query2collection(
        collection=collection_name,
        query_documents=[history_symptoms_text],
        topk=2
    )
    print("查询结果:")
    disease_symptoms = []
    for i in range(len(query_results["documents"][0])):
        doc = query_results["documents"][0][i]
        meta = query_results["metadatas"][0][i]
        distance = query_results["distances"][0][i]
        print(f"  文档: {doc[:50]}... (截断)")
        print(f"  名称: {meta.get('name')}")
        print(f"  距离: {distance:.4f}\n")
        for one_data in example_data:
            if one_data["name"] == meta["name"]:
                disease_symptoms.append(f"{meta.get('name')}: {one_data['matches']}")
    disease_symptoms_text = "\n".join(disease_symptoms)
    prompt = f"""
你是一名医学诊断助手，擅长根据患者描述的症状，从已有疾病症状数据库中识别出最可能的相关疾病，并判断是否需要补充更多症状来明确诊断。

以下是已知疾病及其典型症状的描述：

```
{disease_symptoms_text}
```

请根据患者提供的症状，分析并返回可能相关的疾病名称，按照与描述症状的匹配程度排序。

**输出要求：**

* 返回**多个可能的疾病名称**，并注明是否还需要其他症状以进一步确认。
* 如果某种疾病的现有症状足以支持判断，请注明“不再需要其他症状判断”。
* 如果还不能明确判断，请指出还需询问哪些关键症状来进一步确诊。

**用户描述的症状如下：**

```
{history_symptoms}
```

**请输出如下格式：**

```
可能的疾病：
1. 疾病名称A：建议进一步确认是否有【症状X、症状Y】，以排除其他疾病
2. 疾病名称B：当前症状已足以判断，基本可以确定为该疾病
3. 疾病名称C：建议确认是否伴随【症状Z】，以提高判断准确性
```
"""
    result = query_deepseek(prompt)
    return result

async def getTreatmentAdvice(disease_name: str, tool_context: ToolContext) -> str:
    """
    获取疾病的治疗建议
    params:
    disease_name: 疾病名称
    """
    #查询真实的某个疾病的治疗建议数据库，得到专业建议，这里使用大模型模拟
    print(f"Agent {tool_context.agent_name} 正在调用工具：getTreatmentAdvice，传入的疾病名称是：{disease_name}")
    for one_data in example_data:
        if one_data["name"] == disease_name:
            return one_data["treatment_plan"]
    prompt = """
### 🩺 Prompt：疾病治疗建议助手

你是一位具有丰富临床经验的医学专家，擅长用**通俗易懂的语言**为患者提供**可靠的治疗建议**。你具备对医学文献的检索和整合能力，能够基于最新的循证医学和权威指南，提供针对特定疾病的治疗方案。

请根据用户提供的疾病名称，输出以下内容：

1. ✅ **简要介绍该疾病**（病因、常见症状等，简明扼要）
2. 💊 **常用治疗方法**（包括药物、手术、生活方式干预等）
3. ⚠️ **治疗过程中的注意事项**（如药物副作用、禁忌等）
4. 📚 **参考的权威指南或共识**（如《中国X疾病诊疗指南2023》或WHO指南）

请确保回答内容简洁明了，适合患者理解。如果该疾病属于罕见病或特需专科干预，请注明需就诊相关专科医生。

"""
    prompt += f"###  disease_name: {disease_name}"
    response = query_deepseek(prompt)
    return response


if __name__ == '__main__':
    result = matchDiseaseBySymptoms(symptoms=['失眠'])
    print(result)