import logging
import os
import time
from google.adk.agents.llm_agent import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.genai import types
from typing import Dict, List, Any, AsyncGenerator, Optional, Union
from ...config import SPLIT_TENDER_AGENT_CONFIG
from ...create_model import create_model
from . import prompt

logger = logging.getLogger(__name__)
module_path = os.path.dirname(os.path.abspath(__file__))
module_log_file = os.path.join(module_path, "split_topic.log")
file_handler = logging.FileHandler(module_log_file, encoding="utf-8")
file_handler.setLevel(logging.WARNING)
logger.addHandler(file_handler)

def my_before_model_callback(callback_context: CallbackContext, llm_request: LlmRequest) -> Optional[LlmResponse]:
    # 1. 检查用户输入
    start_time = time.time()
    callback_context.state["start_time"] = start_time
    # 读取招标文件内容
    tender_content = callback_context.state.get("tender_content")
    print(f"读取招标文件内容成功")
    # #放到对话框里
    # llm_request.contents.append(
    #     types.Content(role="user", parts=[types.Part(text=tender_content)])
    # )
    # 返回 None，继续调用 LLM
    return None

def my_after_model_callback(callback_context: CallbackContext, llm_response: LlmResponse):
    agent_name = callback_context.agent_name
    start_time = callback_context.state.get("start_time")
    cost_time = time.time() - start_time
    logger.warning(f"调用了{agent_name}模型后的callback, 耗时: {cost_time} 秒")
    return None

split_tender_agent = Agent(
    name="split_tender",
    model=create_model(model=SPLIT_TENDER_AGENT_CONFIG["model"], provider=SPLIT_TENDER_AGENT_CONFIG["provider"]),
    description="切分招标书,生成Json格式",
    instruction=prompt.SPLIT_TENDER_AGENT_PROMPT,
    output_key="split_tender",  #保存切分结果到split_tender里面，是state里面
    before_model_callback=my_before_model_callback,
    after_model_callback=my_after_model_callback
)