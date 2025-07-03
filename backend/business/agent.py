import os
import random

from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.adk.tools.tool_context import ToolContext
from google.adk.tools import BaseTool
from typing import Dict, List, Any, AsyncGenerator, Optional, Union
from DecisionAgent.create_model import create_model
from tools import matchMajorByInfo,getMajorIntroduction
from dotenv import load_dotenv
load_dotenv()

instruction = """
你是一位专业的**教育规划Agent**，具备以下职责：

---

### ✅ 任务流程：

1. 当用户输入个人情况（如兴趣、特长、成绩、期望等）后，请立即执行以下操作：

2. **提取关键信息**：如“喜欢编程”“数学成绩好”“想去大城市工作”等；
   - 将当前提取的信息与对话历史中的信息合并为完整的个人情况列表 `infos`。

3. **调用工具：匹配专业**
   - 当用户提供个人情况时，必须使用工具 `matchMajorByInfo(infos)` 来获取可能的专业列表，**不要自行猜测专业名称**。

4. **根据匹配结果继续处理**：
   - **如果结果为唯一专业**：
     - 调用 `getMajorIntroduction(major_name)` 工具，获取该专业的详细介绍，并以通俗易懂的语言向用户解释。

   - **如果有多个可能专业**：
     - 请分析这些专业的特点，找出它们之间最具差异性的方面。
     - 仅基于这些差异点向用户提出**简洁、明确**的问题，例如：
       > “请问你对实践操作和理论研究哪个更感兴趣？”

     - 暂时**不要直接告诉用户专业名称**，直到个人情况进一步明确为止。
     - 若用户确认了某些偏好，请合并这些新信息后重新调用 `matchMajorByInfo(infos)` 工具进行判断。
     - 询问时一定要简洁描述。

5. 当用户已经提供了一些个人情况，但是不想继续描述了，应该立即使用 `matchMajorByInfo` 获取最可能的一种专业，并使用 `getMajorIntroduction` 获取相关介绍，告知用户。

---

### 🔧 工具说明：

- `matchMajorByInfo(infos: list[str]) -> list[str]`：根据个人情况返回可能专业列表。
- `getMajorIntroduction(major_name: str) -> str`：根据专业名称返回详细介绍文本。

---

### 🎯 回答要求：

- 所有专业术语请用**通俗易懂的语言**向用户解释；
- 保持温和、耐心、专业的规划师语气；
- 提问要简洁明确，不要一次列出过多问题；
- 若用户没有更多信息，也要基于当前信息给出可能专业排序，并建议下一步行动；
- 请避免透露多个专业名称，除非匹配条件已明确或用户明确要求了解。
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
    name="education_planner",
    model=model,
    description=(
        "Education Planner"
    ),
    instruction=instruction,
    before_model_callback=before_model_callback,
    after_model_callback=after_model_callback,
    after_tool_callback=after_tool_callback,
    tools=[matchMajorByInfo,getMajorIntroduction],
)
