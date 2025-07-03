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
from data import education_data
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
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "education_data")
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

async def matchMajorByInfo(
    infos: list[str],
    tool_context: ToolContext,
) -> str:
    """
    根据用户的个人情况（如兴趣、特长、期望薪资等）推荐合适的专业
    params:
    infos：用户的个人情况描述列表
    :return: 返回所有可能的专业建议
    """
    agent_name = tool_context.agent_name
    history_infos = tool_context.state.get("infos", [])
    print(f"Agent {agent_name} 正在调用工具：matchMajorByInfo，传入的用户信息是：{infos}，历史信息有: {history_infos}")
    history_infos.extend(infos)
    history_infos = list(set(history_infos))
    tool_context.state["infos"] = history_infos #更新症状

    history_infos_text = ",".join(history_infos)
    print(f"\n查询文本: '{history_infos_text}'")
    query_results = chromadb_instance.query2collection(
        collection=COLLECTION_NAME,
        query_documents=[history_infos_text],
        topk=2
    )
    print("查询结果:")
    major_infos = []
    for i in range(len(query_results["documents"][0])):
        doc = query_results["documents"][0][i]
        meta = query_results["metadatas"][0][i]
        distance = query_results["distances"][0][i]
        print(f"  文档: {doc[:50]}... (截断)")
        print(f"  名称: {meta.get('name')}")
        print(f"  距离: {distance:.4f}\n")
        for one_data in education_data:
            if one_data["name"] == meta["name"]:
                major_infos.append(f"{meta.get('name')}: {one_data['matches']}")
    major_infos_text = "\n".join(major_infos)
    prompt = f"""
你是一位专业的教育规划师，擅长根据学生的个人情况（兴趣、特长、成绩、期望等），从专业数据库中推荐最适合的专业，并判断是否需要更多信息来精确推荐。

以下是已知专业及其特点的描述：

```
{major_infos_text}
```

请根据学生提供的个人情况，分析并返回可能适合的专业名称，并按照匹配程度排序。

**输出要求：**

* 返回**多个可能的专业名称**，并注明是否还需要其他信息以进一步确认。
* 如果学生提供的信息足以推荐某个专业，请注明“不再需要其他信息判断”。
* 如果还不能明确判断，请指出还需询问哪些关键信息来进一步确认。

**学生描述的个人情况如下：**

```
{history_infos}
```

**请输出如下格式：**

```
可能的专业：
1. 专业名称A：建议进一步了解你的【数学成绩、编程基础】，以评估是否匹配
2. 专业名称B：根据你对艺术的兴趣，此专业匹配度很高，可以深入了解
3. 专业名称C：建议明确你对【未来工作城市、薪资期望】，以提高推荐准确性
```
"""
    result = query_deepseek(prompt)
    return result

async def getMajorIntroduction(major_name: str, tool_context: ToolContext) -> str:
    """
    获取某个专业的详细介绍
    params:
    major_name: 专业名称
    """
    print(f"Agent {tool_context.agent_name} 正在调用工具：getMajorIntroduction，传入的专业名称是：{major_name}")
    for one_data in education_data:
        if one_data["name"] == major_name:
            return one_data["treatment_plan"]
    prompt = """
### 🎓 Prompt：专业介绍助手

你是一位资深的教育顾问，擅长用**通俗易懂、吸引人**的语言为学生提供**详尽的专业介绍**。你能够整合最新的教育信息和行业趋势，提供全面的专业解读。

请根据用户提供的专业名称，输出以下内容：

1. ✅ **专业简介**（核心课程、培养目标等，简明扼要）
2. 📚 **深造方向**（考研、出国留学的相关领域）
3. 💼 **就业前景**（主要就业行业、典型职位、薪资水平参考）
4. 🏫 **推荐院校**（列举几所在该专业领域有优势的院校）

请确保回答内容结构清晰、信息丰富，能激发学生的兴趣。如果该专业有特殊的报考要求（如美术加试、身体条件限制），请一并注明。

"""
    prompt += f"###  major_name: {major_name}"
    response = query_deepseek(prompt)
    return response


if __name__ == '__main__':
    result = matchMajorByInfo(infos=['我喜欢画画，以后想学艺术相关的专业'])
    print(result)

    result = getMajorIntroduction(major_name='绘画')
    print(result)
