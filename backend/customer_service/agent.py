import os
import random

from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.adk.tools.tool_context import ToolContext
from google.adk.tools import BaseTool
from typing import Dict, List, Any, AsyncGenerator, Optional, Union
from DecisionAgent.create_model import create_model
from tools import query_knowledge_base, get_solution
from dotenv import load_dotenv
load_dotenv()

instruction = """
你是一位专业的**客户服务Agent**，具备以下职责：

---

### ✅ 任务流程：

1.  当用户提出问题或描述遇到的困难时，请立即执行以下操作：

2.  **提取关键信息**：从用户输入中识别核心问题，例如“无法登录”、“订单状态查询”、“产品功能咨询”等。
    -   将当前提取的信息与对话历史中的信息合并，形成完整的用户问题描述 `query`。

3.  **调用工具：查询知识库**
    -   使用 `query_knowledge_base(query)` 工具，根据用户问题在知识库中检索相关的解决方案或信息。

4.  **根据查询结果继续处理**：
    -   **如果找到解决方案**：
        -   调用 `get_solution(solution_id)` 工具，获取该解决方案的详细步骤，并以清晰、简洁的语言向用户解释。

    -   **如果未找到直接解决方案或信息不明确**：
        -   分析知识库返回的可能相关的信息，找出最具关联性的方面。
        -   基于这些信息，向用户提出**简洁、明确**的问题，以澄清用户的意图，例如：
            > “请问您具体是无法登录哪个系统？”
            > “您能否提供订单号以便我为您查询？”

        -   若用户提供了更多信息，请合并新信息后重新调用 `query_knowledge_base(query)` 工具进行查询。
        -   提问时，语言要简洁明了。

5.  如果用户提供的信息不足以在知识库中找到精确答案，也应根据现有信息给出最可能的建议，并引导用户提供更多细节。

---

### 🔧 工具说明：

-   `query_knowledge_base(query: str) -> list[str]`：根据用户问题在知识库中检索解决方案。
-   `get_solution(solution_id: str) -> str`：根据方案ID获取详细解决方案。

---

### 🎯 回答要求：

-   所有答复都应使用**通俗易懂的语言**。
-   保持友好、耐心、专业的客服语气。
-   提问要简洁明确，避免一次性提出过多问题。
-   如果无法立即解决问题，也要基于当前信息给出建议，并告知用户下一步操作。
"""

model = create_model(model=os.environ["LLM_MODEL"], provider=os.environ["MODEL_PROVIDER"])

def before_model_callback(callback_context: CallbackContext, llm_request: LlmRequest) -> Optional[LlmResponse]:
    # 1. 检查用户输入
    agent_name = callback_context.agent_name
    history_length = len(llm_request.contents)
    metadata = callback_context.state.get("metadata")
    print(f"调用了{agent_name}模型前的callback, 现在Agent共有{history_length}条历史记录,metadata数据为：{metadata}")
    #清空contents,不需要上一步的拆分topic的记录, 不能在这里清理，否则，每次调用工具都会清除记忆，白操作了
    # llm_request.contents.clear()
    # 返回 None，继续调用 LLM
    return None
def after_model_callback(callback_context: CallbackContext, llm_response: LlmResponse) -> Optional[LlmResponse]:
    # 1. 检查用户输入
    agent_name = callback_context.agent_name
    response_data = len(llm_response.content.parts)
    metadata = callback_context.state.get("metadata")
    print(f"调用了{agent_name}模型后的callback, 这次模型回复{response_data}条信息,metadata数据为：{metadata}")
    #清空contents,不需要上一步的拆分topic的记录, 不能在这里清理，否则，每次调用工具都会清除记忆，白操作了
    # llm_request.contents.clear()
    # 返回 None，继续调用 LLM
    return None

def after_tool_callback(
    tool: BaseTool, args: Dict[str, Any], tool_context: ToolContext, tool_response: Dict
) -> Optional[Dict]:

  tool_name = tool.name
  print(f"调用了{tool_name}工具后的callback, tool_response数据为：{tool_response}")
  return None

root_agent = Agent(
    name="customer_service",
    model=model,
    description=(
        "Customer Service"
    ),
    instruction=instruction,
    before_model_callback=before_model_callback,
    after_model_callback=after_model_callback,
    after_tool_callback=after_tool_callback,
    tools=[query_knowledge_base, get_solution],
)