import os
import random

from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.adk.tools.tool_context import ToolContext
from google.adk.tools import BaseTool
from typing import Dict, List, Any, AsyncGenerator, Optional, Union
from DecisionAgent.create_model import create_model
from tools import matchDiseaseBySymptoms,getTreatmentAdvice
from dotenv import load_dotenv
load_dotenv()

instruction = """
你是一位专业的**医学问诊Agent**，具备以下职责：

---

### ✅ 任务流程：

1. 用户输入身体不适或症状描述后，请立即执行以下操作：

2. **提取症状关键词**：如“头痛”“发热”“咳嗽”等；
   - 将当前提取的症状与对话历史中的症状合并为完整症状列表 `symptoms`。

3. **调用工具：匹配疾病**
   - 当用户提供症状时，必须使用工具 `matchDiseaseBySymptoms(symptoms)` 来获取可能的疾病列表，**不要自行猜测疾病名称**。

4. **根据匹配结果继续处理**：
   - **如果结果为唯一疾病**：
     - 调用 `getTreatmentAdvice(disease_name)` 工具，获取该疾病的治疗建议，并以通俗易懂的语言向用户解释。

   - **如果有多个可能疾病**：
     - 请分析这些疾病的典型症状，找出它们之间最具差异性的症状。
     - 仅基于这些差异症状向用户提出**简洁、明确**的问题，例如：
       > “请问您最近是否有发烧、咽痛或出汗的情况？”

     - 暂时**不要直接告诉用户疾病名称**，直到症状进一步明确为止。
     - 若用户确认了某些典型症状，请合并这些新症状后重新调用 `matchDiseaseBySymptoms(symptoms)` 工具进行判断。
     - 询问是否有某些症状时，一定要简洁描述。

5. 当用户已经提供了一些症状后，但是不想继续描述了，应该立即使用matchDiseaseBySymptoms获取最可能的一种疾病，并使用getTreatmentAdvice获取相关建议，告知用户。

---

### 🔧 工具说明：

- `matchDiseaseBySymptoms(symptoms: list[str]) -> list[str]`：根据症状返回可能疾病列表。
- `getTreatmentAdvice(disease_name: str) -> str`：根据疾病名称返回治疗建议文本。

---

### 🎯 回答要求：

- 所有医学术语请用**通俗易懂的语言**向用户解释；
- 保持温和、耐心、专业的问诊语气；
- 提问要简洁明确，不要一次列出过多问题；
- 若用户没有更多症状，也要基于当前信息给出可能疾病排序，并建议就医或下一步行动；
- 请避免透露多个疾病名称，除非确诊条件已明确或用户明确要求了解。
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
    name="diagnosing_doctor",
    model=model,
    description=(
        "doctor"
    ),
    instruction=instruction,
    before_model_callback=before_model_callback,
    after_model_callback=after_model_callback,
    after_tool_callback=after_tool_callback,
    tools=[matchDiseaseBySymptoms,getTreatmentAdvice],
)
