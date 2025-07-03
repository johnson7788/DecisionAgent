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
from data import financial_data # 假设你有一个名为 financial_data 的新数据文件
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
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "financial_data")
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

async def matchFinancialProducts(
    infos: list[str],
    tool_context: ToolContext,
) -> str:
    """
    根据用户的个人情况（如风险偏好、投资期限、资金量、投资目标等）推荐合适的金融产品
    params:
    infos：用户的个人情况描述列表
    :return: 返回所有可能的金融产品建议
    """
    agent_name = tool_context.agent_name
    history_infos = tool_context.state.get("infos", [])
    print(f"Agent {agent_name} 正在调用工具：matchFinancialProducts，传入的用户信息是：{infos}，历史信息有: {history_infos}")
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
    product_infos = []
    for i in range(len(query_results["documents"][0])):
        doc = query_results["documents"][0][i]
        meta = query_results["metadatas"][0][i]
        distance = query_results["distances"][0][i]
        print(f"  文档: {doc[:50]}... (截断)")
        print(f"  名称: {meta.get('name')}")
        print(f"  距离: {distance:.4f}\n")
        for one_data in financial_data:
            if one_data["name"] == meta["name"]:
                product_infos.append(f"{meta.get('name')}: {one_data['matches']}")
    product_infos_text = "\n".join(product_infos)
    prompt = f"""
你是一位专业的金融投资顾问，擅长根据客户的个人情况（风险偏好、投资期限、资金量、投资目标等），从金融产品数据库中推荐最适合的金融产品，并判断是否需要更多信息来精确推荐。

以下是已知金融产品及其特点的描述：

```
{product_infos_text}
```

请根据客户提供的个人情况，分析并返回可能适合的金融产品名称，并按照匹配程度排序。

**输出要求：**

* 返回**多个可能的金融产品名称**，并注明是否还需要其他信息以进一步确认。
* 如果客户提供的信息足以推荐某个金融产品，请注明“不再需要其他信息判断”。
* 如果还不能明确判断，请指出还需询问哪些关键信息来进一步确认。

**客户描述的个人情况如下：**

```
{history_infos}
```

**请输出如下格式：**

```
可能的金融产品：
1. 金融产品A：建议进一步了解你的【资金量、投资期限】，以评估是否匹配
2. 金融产品B：根据你对稳健收益的偏好，此产品匹配度很高，可以深入了解
3. 金融产品C：建议明确你对【流动性、风险承受能力】，以提高推荐准确性
```
"""
    result = query_deepseek(prompt)
    return result

async def getFinancialProductIntroduction(product_name: str, tool_context: ToolContext) -> str:
    """
    获取某个金融产品的详细介绍
    params:
    product_name: 金融产品名称
    """
    print(f"Agent {tool_context.agent_name} 正在调用工具：getFinancialProductIntroduction，传入的金融产品名称是：{product_name}")
    for one_data in financial_data:
        if one_data["name"] == product_name:
            return one_data["treatment_plan"]
    prompt = """
### 💰 Prompt：金融产品介绍助手

你是一位资深的金融顾问，擅长用**通俗易懂、吸引人**的语言为客户提供**详尽的金融产品介绍**。你能够整合最新的金融信息和市场趋势，提供全面的产品解读。

请根据用户提供的金融产品名称，输出以下内容：

1. ✅ **产品简介**（产品类型、投资范围、风险等级等，简明扼要）
2. 📈 **收益特点**（预期收益、收益计算方式、历史表现等）
3. 🔒 **风险提示**（主要风险、风险控制措施等）
4. 📊 **适合人群**（适合的风险偏好、投资期限、资金量等）

请确保回答内容结构清晰、信息丰富，能激发客户的兴趣。如果该产品有特殊的购买要求（如起投金额、购买渠道），请一并注明。

"""
    prompt += f"###  product_name: {product_name}"
    response = query_deepseek(prompt)
    return response


if __name__ == '__main__':
    # 假设 financial_data 已经定义
    # from data import financial_data
    # result = matchFinancialProducts(infos=['我希望投资风险较低，收益稳健的产品'])
    # print(result)

    # result = getFinancialProductIntroduction(product_name='货币基金')
    # print(result)
    pass