# 文件名: slide_agent/sub_agents/research_topic/agent.py
import json
import logging
import os
import time
from typing import AsyncGenerator
from pydantic import PrivateAttr
from google.adk.agents.llm_agent import Agent
from google.adk.agents import ParallelAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.genai import types
from typing import Dict, List, Any, AsyncGenerator, Optional, Union
from google.adk.events.event import Event,EventActions
# 导入 ParallelAgent 源码中的内部辅助函数，这是实现并行的关键
from google.adk.agents.parallel_agent import (
    _create_branch_ctx_for_sub_agent,
    _merge_agent_run,
)

# from .load_mcp import load_mcp_tools
from ...config import AUDIT_AGENT_CONFIG
from ...create_model import create_model
from . import prompt

# 配置日志
# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
module_path = os.path.dirname(os.path.abspath(__file__))
module_log_file = os.path.join(module_path, "research_topic.log")
file_handler = logging.FileHandler(module_log_file, encoding="utf-8")
file_handler.setLevel(logging.WARNING)
logger.addHandler(file_handler)

# 审计Agent
audit_model = create_model(model=AUDIT_AGENT_CONFIG["model"],
                              provider=AUDIT_AGENT_CONFIG["provider"])

def audit_agent_before_model_callback(callback_context: CallbackContext, llm_request: LlmRequest) -> Optional[LlmResponse]:
    # 1. 检查用户输入
    agent_name = callback_context.agent_name
    history_length = len(llm_request.contents)
    print(f"调用了{agent_name} research Agent的callback, 现在Agent共有{history_length}条历史记录")
    #清空contents,不需要上一步的拆分topic的记录, 不能在这里清理，否则，每次调用工具都会清除记忆，白操作了
    # llm_request.contents.clear()
    # 返回 None，继续调用 LLM
    return None


# 自定义我们的动态并行 Agent
class DynamicParallelSearchAgent(ParallelAgent):
    """
    一个可以根据输入动态创建和并行执行子Agent的Agent。
    它期望从上一个Agent接收一个JSON字符串，其中包含一个'topics'列表。
    """
    _agent_template: Agent = PrivateAttr()

    def __init__(self, **kwargs):
        """
        初始化动态并行Agent。

        Args:
            agent_template: 一个Agent实例，用作创建动态子Agent的模板。
            **kwargs: 传递给父类ParallelAgent的参数。
        """
        # sub_agents 初始化为空，因为它们是动态生成的
        super().__init__(sub_agents=[], **kwargs)

    async def _run_async_impl(
            self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """
        重写核心运行逻辑，以实现动态并行化。
        """
        # 1. 从上下文中获取上一个Agent的输出
        # 读取招标书的切片
        split_tendor = ctx.session.state.get("split_tendor", {})
        logger.info(f"DynamicParallelSearchAgent 收到输入: {split_tendor}")
        split_tendor_list = []
        try:
            # 清理可能的Markdown代码块
            if isinstance(split_tendor, str):
                if split_tendor.strip().startswith("```json"):
                    split_tendor = split_tendor.strip()[7:-3]
                elif split_tendor.strip().startswith("```"):
                    split_tendor = split_tendor.strip()[3:-3]

            split_tendor_list = json.loads(split_tendor)
        except (json.JSONDecodeError, AttributeError) as e:
            yield Event(
                author=self.name,
                content=types.Content(parts=[types.Part(text=f"错误：解析主题JSON失败 - {e}")]),
                actions=EventActions(escalate=True)
            )
            return

        # 3. 为每个主题动态创建子Agent
        dynamic_sub_agents = []
        # 每个子Agent的输出key的集合，最终存储到state中
        research_output_keys = []
        for idx, topic in enumerate(split_tendor_list):
            topic_id = topic.get("requirements", "N/A")  # 一些要求

            # 创建一个定制化的指令，将主题信息注入到基础prompt中
            custom_instruction = (
                f"{prompt.RESEARCH_TOPIC_AGENT_PROMPT}\n\n"
                f"Your specific task is to research the following topic:\n"
                f"- **Topic Title**: {topic_title}\n"
                f"- **Description**: {topic_description}\n"
                f"- **Keywords**: {topic.get('keywords', [])}\n"
                f"- **Research Focus**: {topic.get('research_focus', '')}"
            )
            new_audit_agent = Agent(
                model=audit_model,
                name=f"audiot_agent_{topic_id}",  # 模板名称
                description="单独的1个审计Agent",
                instruction=custom_instruction,
                output_key=f"audiot_agent_{idx}",  #输出的内容的key
                before_model_callback=audit_agent_before_model_callback
            )
            audit_output_keys.append(f"audit_agent_{topic_id}")
            # 关键：设置父级Agent，ADK框架需要这个来构建Agent树
            new_audit_agent.parent_agent = self
            dynamic_sub_agents.append(new_audit_agent)
        ctx.session.state["audit_output_keys"] = audit_output_keys
        logger.info(f"成功创建了 {len(dynamic_sub_agents)} 个动态 audit agents.")

        # 4. 并行运行所有动态创建的Agent
        # 这部分逻辑直接借鉴自 ParallelAgent 的源码
        start_time = time.time()
        ctx.session.events = [] # 清空上个Agent的事件
        agent_runs = [
            sub_agent.run_async(
                _create_branch_ctx_for_sub_agent(self, sub_agent, ctx)
            )
            for sub_agent in dynamic_sub_agents
        ]

        # 5. 合并并产生事件流
        async for event in _merge_agent_run(agent_runs):
            yield event
        print(f"所有动态 Audit Agent 运行完毕")
        cost_time = time.time() - start_time
        logger.warning(f"所有 Audit Agent 的总耗时为: {cost_time} 秒")


# 实例化我们的新 Agent
audit_parallel_agent = DynamicParallelSearchAgent(
    name="audit_parallel_agent",
    description="根据拆分的审计要求，对投标书进行审计",
)