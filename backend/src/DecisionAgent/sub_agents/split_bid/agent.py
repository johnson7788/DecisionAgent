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

    user_input = callback_context.user_content.parts[0].text
    callback_context.state["outline"] = user_input
    print("调用了SplitTopicAgent的Outline的callback，存储outline信息")
    logger.info("调用了SplitTopicAgent的Outline的callback，存储outline信息")
    #为啥需要手动加入user_input?
    llm_request.contents.append(
        types.Content(role="user", parts=[types.Part(text=user_input)])
    )
    # 返回 None，继续调用 LLM
    return None

def my_after_model_callback(callback_context: CallbackContext, llm_response: LlmResponse):
    agent_name = callback_context.agent_name
    start_time = callback_context.state.get("start_time")
    cost_time = time.time() - start_time
    logger.warning(f"调用了{agent_name}模型后的callback, 耗时: {cost_time} 秒")
    return None

split_bid_agent = Agent(
    name="split_bid",
    model=create_model(model=SPLIT_TENDER_AGENT_CONFIG["model"], provider=SPLIT_TENDER_AGENT_CONFIG["provider"]),
    description="切分投标书",
    instruction=prompt.SPLIT_BID_AGENT_PROMPT,
    output_key="split_bid",
    before_model_callback=my_before_model_callback,
    after_model_callback=my_after_model_callback
)