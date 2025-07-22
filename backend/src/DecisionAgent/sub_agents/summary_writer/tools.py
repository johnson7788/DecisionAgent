#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/6/20 10:02
# @File  : tools.py.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  : 搜索文献内容

import re
import os
import time
from datetime import datetime
import random
import hashlib
from pathlib import Path
from fastmcp import FastMCP
from google.adk.tools import ToolContext
from google.adk.tools.agent_tool import AgentTool
import requests
from urllib.parse import quote


async def SearchDocuments(query: str, tool_context: ToolContext) -> list[dict]:
    """
    搜索一些文献内容
    :return:
    """
    agent_name = tool_context.agent_name
    print(f"Agent {agent_name}正在调用工具：SearchDocuments: " + query)
    metadata = tool_context.state.get("metadata", {})
    documents = "现在网络不可用，请暂时忽略该工具"
    print(f"研究员{agent_name}的metadata信息: {metadata}")
    # 一些参考信息
    tool_context.state["references"] = ["1. Alterations of lung microbiota in patients with non-small cell lung cancer. PMID: 35254206.",
"2. Recent progress in targeted therapy for non-small cell lung cancer. PMID: 36909198.",
"3. Ivonescimab: First Approval. PMID: 39073550.",
"4. Ivonescimab in non-small cell lung cancer: harmonizing immunotherapy and anti-angiogenesis. PMID: 40162997.",
"5. Ivonescimab Plus Chemotherapy in Non-Small Cell Lung Cancer With EGFR Variant: A Randomized Clinical Trial. PMID: 38820549.",
"6. Ivonescimab versus pembrolizumab for PD-L1-positive non-small cell lung cancer (HARMONi-2): a randomised, double-blind, phase 3 study in China. PMID: 40057343.",
"7. Safety, Pharmacokinetics, and Pharmacodynamics Evaluation of Ivonescimab, a Novel Bispecific Antibody Targeting PD-1 and VEGF, in Chinese Patients With Advanced Solid Tumors. PMID: 40114411."]
    tool_context.state["metadata"]["references"] = """1. Alterations of lung microbiota in patients with non-small cell lung cancer. PMID: 35254206.
2. Recent progress in targeted therapy for non-small cell lung cancer. PMID: 36909198.
3. Ivonescimab: First Approval. PMID: 39073550.
4. Ivonescimab in non-small cell lung cancer: harmonizing immunotherapy and anti-angiogenesis. PMID: 40162997.
5. Ivonescimab Plus Chemotherapy in Non-Small Cell Lung Cancer With EGFR Variant: A Randomized Clinical Trial. PMID: 38820549.
6. Ivonescimab versus pembrolizumab for PD-L1-positive non-small cell lung cancer (HARMONi-2): a randomised, double-blind, phase 3 study in China. PMID: 40057343.
7. Safety, Pharmacokinetics, and Pharmacodynamics Evaluation of Ivonescimab, a Novel Bispecific Antibody Targeting PD-1 and VEGF, in Chinese Patients With Advanced Solid Tumors. PMID: 40114411."""
    tool_context.state["news"] = {"references": ["hello","yes"]}
    return documents


if __name__ == '__main__':
    result = SearchDocuments("flowers")
    print(result)
