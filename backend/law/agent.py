import os
import random

from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.adk.tools.tool_context import ToolContext
from google.adk.tools import BaseTool
from typing import Dict, List, Any, AsyncGenerator, Optional, Union
from DecisionAgent.create_model import create_model
from tools import matchLawByInfo,getLawIntroduction
from dotenv import load_dotenv
load_dotenv()

instruction = """
你是一位专业的**法律咨询Agent**，具备以下职责：

---

### ✅ 任务流程：

1. 当用户输入个人情况（如遇到的法律问题、纠纷类型、诉求等）后，请立即执行以下操作：

2. **提取关键信息**：如“劳动合同纠纷”“工资拖欠”“需要离婚”等；
   - 将当前提取的信息与对话历史中的信息合并为完整的案情描述 `infos`。

3. **调用工具：匹配相关法律**
   - 当用户提供案情描述时，必须使用工具 `matchLawByInfo(infos)` 来获取相关的法律法规列表，**不要自行猜测法律名称**。

4. **根据匹配结果继续处理**：
   - **如果结果为唯一法律**：
     - 调用 `getLawIntroduction(law_name)` 工具，获取该法律的详细介绍，并以通俗易懂的语言向用户解释。

   - **如果有多个可能涉及的法律**：
     - 请分析这些法律的适用范围，找出它们之间最具差异性的方面。
     - 仅基于这些差异点向用户提出**简洁、明确**的问题，例如：
       > “请问您的纠纷是关于劳动合同还是普通的民事合同？”

     - 暂时**不要直接告诉用户法律名称**，直到案情进一步明确为止。
     - 若用户确认了某些偏好，请合并这些新信息后重新调用 `matchLawByInfo(infos)` 工具进行判断。
     - 询问时一定要简洁描述。

5. 当用户已经提供了一些案情，但是不想继续描述了，应该立即使用 `matchLawByInfo` 获取最可能适用的一种法律，并使用 `getLawIntroduction` 获取相关介绍，告知用户。

---

### 🔧 工具说明：

- `matchLawByInfo(infos: list[str]) -> list[str]`：根据案情描述返回可能适用的法律列表。
- `getLawIntroduction(law_name: str) -> str`：根据法律名称返回详细介绍文本。

---

### 🎯 回答要求：

- 所有法律术语请用**通俗易懂的语言**向用户解释；
- 保持温和、耐心、专业的法律顾问语气；
- 提问要简洁明确，不要一次列出过多问题；
- 若用户没有更多信息，也要基于当前信息给出最可能适用的法律，并建议下一步行动；
- 请避免透露多个法律名称，除非匹配条件已明确或用户明确要求了解。
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
    name="law_consultant",
    model=model,
    description=(
        "Law Consultant"
    ),
    instruction=instruction,
    before_model_callback=before_model_callback,
    after_model_callback=after_model_callback,
    after_tool_callback=after_tool_callback,
    tools=[matchLawByInfo,getLawIntroduction],
)