import os
from typing import Dict, Any, Optional
from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.adk.tools import BaseTool
from google.adk.tools.tool_context import ToolContext
from sub_agents.split_tender.agent import split_tendor_agent
from sub_agents.split_bid.agent import split_bid_agent
from sub_agents.audit_content.agent import audit_parallel_agent
from sub_agents.summary_writer.agent import summary_writer_agent
from dotenv import load_dotenv

load_dotenv()

def before_agent_callback(callback_context: CallbackContext):
    """
    在Agent调用之前，进行数据处理
    :param callback_context:
    :return:
    """
    # print(callback_context)
    medata = callback_context.state.get("metadata")
    tender_file = medata.get("tender_file")  # 招标文件路径
    bid_file = medata.get("bid_file")  #投标文件路径
    assert tender_file and bid_file, "请提供招标文件和投标文件路径"
    # 读取招标文件内容
    with open(tender_file, "r") as f:
        tender_content = f.read()
    with open(bid_file, "r") as f:
        bid_content = f.read()
    # 存储到state中
    callback_context.state["tender_content"] = tender_content
    callback_context.state["bid_content"] = bid_content
    return None

# 1. 切分招标书
# 2. 切分投标文件
# 3. 每个招标需求，对应投标文件进行检查
root_agent = SequentialAgent(
    name="coordinator_agent",
    description="协调者",
    sub_agents=[
        split_tendor_agent,
        split_bid_agent,
        audit_parallel_agent,
        summary_writer_agent,
    ],
    before_agent_callback=before_agent_callback,
)