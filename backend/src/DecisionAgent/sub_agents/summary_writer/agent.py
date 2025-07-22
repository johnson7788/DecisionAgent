
from typing import Dict, List, Any, AsyncGenerator, Optional, Union
from google.genai import types
from google.adk.agents.llm_agent import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from ...config import SUMMARY_AGENT_CONFIG
from ...create_model import create_model
from .tools import SearchDocuments
from . import prompt


def my_before_model_callback(callback_context: CallbackContext, llm_request: LlmRequest) -> Optional[LlmResponse]:
    # 1. 检查用户输入
    user_input = callback_context.user_content.parts[0].text
    callback_context.state["outline"] = user_input
    #为啥需要手动加入user_input?
    llm_request.contents.append(
        types.Content(role="user", parts=[types.Part(text=user_input)])
    )
    # 返回 None，继续调用 LLM
    return None

summary_writer_agent = Agent(
    model=create_model(model=SUMMARY_AGENT_CONFIG["model"], provider=SUMMARY_AGENT_CONFIG["provider"]),
    name="SummaryAgent",
    description="读取审计结果，并汇总",
    instruction=prompt.SUMMARY_AGENT_PROMPT,
    before_model_callback=my_before_model_callback,
    output_key="summary_document",
    tools=[SearchDocuments],
)
