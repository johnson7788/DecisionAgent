import os
import random

from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.adk.tools.tool_context import ToolContext
from google.adk.tools import BaseTool
from typing import Dict, List, Any, AsyncGenerator, Optional, Union
from DecisionAgent.create_model import create_model
from tools import analyzeBusinessProblem, getIndustryReport
from dotenv import load_dotenv
load_dotenv()

instruction = """
你是一位顶级的**商业咨询Agent**，具备以下职责：

---

### ✅ 任务流程：

1.  当用户输入商业问题（如市场进入、竞争分析、运营效率等）后，请立即执行以下操作：

2.  **提取关键问题**：例如“咖啡馆市场定位”“新产品营销策略”等。
    -   将当前问题与对话历史中的信息整合，形成完整的商业问题描述 `problems`。

3.  **调用工具：分析商业问题**
    -   必须使用 `analyzeBusinessProblem(problems)` 工具来获取初步的解决方案和建议，**不要自行猜测解决方案**。

4.  **根据分析结果继续处理**：
    -   **如果分析结果提供了明确的解决方案**：
        -   以通俗易懂的语言向用户解释该方案，并阐述其优势和潜在风险。

    -   **如果分析结果表明需要更多行业信息**：
        -   调用 `getIndustryReport(industry_name)` 工具，获取相关行业的详细报告。
        -   整合行业报告中的信息，再次调用 `analyzeBusinessProblem(problems)`，以形成更完善的建议。
        -   在获取报告前，可以向用户确认需要深入研究的行业领域，例如：
            > “为了给您更精准的建议，我需要深入了解一下餐饮服务行业的市场趋势，您同意吗？”

5.  当用户的问题不够清晰时，应主动提出引导性问题，以帮助用户明确咨询需求。

---

### 🔧 工具说明：

-   `analyzeBusinessProblem(problems: list[str]) -> str`：根据商业问题描述返回初步的解决方案。
-   `getIndustryReport(industry_name: str) -> str`：根据行业名称返回详细的行业分析报告。

---

### 🎯 回答要求：

-   所有商业术语请用**通俗易懂的语言**向用户解释。
-   保持专业、客观、有洞察力的咨询顾问语气。
-   提问要精准，直击问题核心。
-   若信息不足，也要基于现有信息给出初步判断，并说明需要进一步分析的方向。
"""

model = create_model(model=os.environ["LLM_MODEL"], provider=os.environ["MODEL_PROVIDER"])

def before_model_callback(callback_context: CallbackContext, llm_request: LlmRequest) -> Optional[LlmResponse]:
    agent_name = callback_context.agent_name
    history_length = len(llm_request.contents)
    metadata = callback_context.state.get("metadata")
    print(f"调用了{agent_name}模型前的callback, 现在Agent共有{history_length}条历史记录,metadata数据为：{metadata}")
    return None

def after_model_callback(callback_context: CallbackContext, llm_response: LlmResponse) -> Optional[LlmResponse]:
    agent_name = callback_context.agent_name
    response_data = len(llm_response.content.parts)
    metadata = callback_context.state.get("metadata")
    print(f"调用了{agent_name}模型后的callback, 这次模型回复{response_data}条信息,metadata数据为：{metadata}")
    return None

def after_tool_callback(
    tool: BaseTool, args: Dict[str, Any], tool_context: ToolContext, tool_response: Dict
) -> Optional[Dict]:
    tool_name = tool.name
    print(f"调用了{tool_name}工具后的callback, tool_response数据为：{tool_response}")
    return None

root_agent = Agent(
    name="business_consultant",
    model=model,
    description="Business Consultant",
    instruction=instruction,
    before_model_callback=before_model_callback,
    after_model_callback=after_model_callback,
    after_tool_callback=after_tool_callback,
    tools=[analyzeBusinessProblem, getIndustryReport],
)