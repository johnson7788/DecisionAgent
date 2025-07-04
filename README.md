# 辅助决策型Agent

在企业管理、教育选择、法律咨询甚至心理健康等场景中，人们常常面临「信息复杂」「选项众多」「缺乏经验」的问题。传统的问答类AI无法提供真正“有逻辑”的建议，于是我构建了一个更实用的系统：辅助决策型 Agent（Decision Agent）。

它不是简单回答一个问题，而是：

分析用户的情况和目标；
累加上下文和历史；
调用工具或数据库查找方案；
主动追问缺失的信息；
最终给出结构化、有针对性的建议。

# 作者微信: johnsongzc

# 安装
前端
```
cd frontend
npm install
npm run dev
```

后端，任选1个Agent
```
cd backend
pip install -e .
cd doctor
cp env_template .env
python main_data_prepare.py
python main_api.py
```
## .env讲解
```
GOOGLE_API_KEY=xxx
DEEPSEEK_API_KEY=sk-xxx
#工具中使用的嵌入模型的key是ALI_API_KEY的，可以自行更改
ALI_API_KEY=sk-xxx
OPENAI_API_KEY=sk-xxx
CLAUDE_API_KEY=skxxx
# 流式的输出结果
STREAMING=true
# 默认使用的模型
MODEL_PROVIDER=openai
LLM_MODEL=gpt-4.1
# 工具中使用的模型，向量模型粗筛，LLM模型细筛
TOOL_MODEL_API_BASE=https://api.deepseek.com/v1
TOOL_MODEL_API_KEY=sk-xxx
TOOL_MODEL_NAME=deepseek-chat
TOOL_MODEL_PROVIDER=deepseek
```

## 数据

**决策型 Agent 场景的模拟数据（结构与疾病场景一致）**，每条数据包括 `name`（名称）、`matches`（匹配条件/特征描述）、`treatment_plan`（应对策略或处理方案）：


## ✅ 数据格式汇总

目前你已拥有以下结构化决策数据样例：

| 场景       | 数据变量名              |
| ---------- | ----------------------- |
| 教育路径   | `education_data`        |
| 法律咨询   | `law_data`              |
| 金融投资   | `finance_data`          |
| 心理健康   | `mental_health_data`    |
| 企业经营   | `business_data`         |
| 供应链优化 | `supply_chain_data`     |
| 危机应对   | `crisis_data`           |
| 智能客服   | `customer_service_data` |
| 疾病诊断   | `example_data`          |


下面是每个**辅助决策型 Agent 场景**的流程图。

---

### 🎓 教育路径（`education_data`）

```mermaid
flowchart TD
    A[用户描述背景与目标] --> B[提取当前教育需求]
    B --> C[累加历史学习经历]
    C --> D[匹配可能的教育路径]

    D --> E{是否唯一匹配}

    E -- 是唯一 --> F[查询推荐课程与进阶路径]

    E -- 多个可能 --> G[提取相关能力或目标]
    G --> H[询问用户是否具备相关条件]
    H --> B

    F --> I[输出教育路径建议并结束]
```

---

### ⚖️ 法律咨询（`law_data`）

```mermaid
flowchart TD
    A[用户描述法律问题] --> B[提取法律关键词]
    B --> C[累加案件背景信息]
    C --> D[匹配相关法律领域]

    D --> E{是否唯一匹配}

    E -- 是唯一 --> F[查询相关法律条款与建议]

    E -- 多个可能 --> G[提取更多法律细节]
    G --> H[追问相关背景细节]
    H --> B

    F --> I[输出法律建议并结束]
```

---

### 💰 金融投资（`finance_data`）

```mermaid
flowchart TD
    A[用户描述投资目标与风险偏好] --> B[提取金融偏好]
    B --> C[累加资产与市场信息]
    C --> D[匹配可选投资方案]

    D --> E{是否唯一匹配}

    E -- 是唯一 --> F[生成投资组合建议]

    E -- 多个可能 --> G[提取更多投资偏好]
    G --> H[追问投资期限或流动性要求]
    H --> B

    F --> I[输出理财建议并结束]
```

---

### 🧠 心理健康（`mental_health_data`）

```mermaid
flowchart TD
    A[用户描述心理困扰] --> B[提取情绪与行为特征]
    B --> C[结合历史心理状态]
    C --> D[匹配心理状态类型]

    D --> E{是否唯一匹配}

    E -- 是唯一 --> F[建议心理疏导方式或专业咨询]

    E -- 多个可能 --> G[提取其它心理特征]
    G --> H[询问更多情绪表现]
    H --> B

    F --> I[输出心理建议并结束]
```

---

### 🏢 企业经营（`business_data`）

```mermaid
flowchart TD
    A[用户描述经营问题] --> B[提取业务挑战特征]
    B --> C[结合企业历史数据]
    C --> D[匹配可能经营策略]

    D --> E{是否唯一匹配}

    E -- 是唯一 --> F[给出管理或战略调整建议]

    E -- 多个可能 --> G[获取更多企业背景信息]
    G --> H[追问市场、团队或财务数据]
    H --> B

    F --> I[输出经营建议并结束]
```

---

### 📦 供应链优化（`supply_chain_data`）

```mermaid
flowchart TD
    A[用户描述当前供应链问题] --> B[提取供应链环节]
    B --> C[分析历史供应数据]
    C --> D[匹配优化方案]

    D --> E{是否唯一匹配}

    E -- 是唯一 --> F[提供优化策略与工具推荐]

    E -- 多个可能 --> G[追问需求预测、库存或物流情况]
    G --> H[补充更多链路信息]
    H --> B

    F --> I[输出优化建议并结束]
```

---

### 🚨 危机应对（`crisis_data`）

```mermaid
flowchart TD
    A[用户描述危机事件] --> B[识别危机类型与影响范围]
    B --> C[参考过往应对策略]
    C --> D[匹配最佳应急响应措施]

    D --> E{是否唯一匹配}

    E -- 是唯一 --> F[提供响应流程与应急资源建议]

    E -- 多个可能 --> G[追问资源、时效或人员状态]
    G --> H[细化事件情况]
    H --> B

    F --> I[输出危机响应建议并结束]
```

---

### 🤖 智能客服（`customer_service_data`）

```mermaid
flowchart TD
    A[用户发起服务请求或投诉] --> B[识别请求类型]
    B --> C[分析历史服务记录]
    C --> D[匹配应答或解决方案]

    D --> E{是否唯一匹配}

    E -- 是唯一 --> F[生成应答内容并处理请求]

    E -- 多个可能 --> G[追问更多请求细节或上下文]
    G --> H[细化用户问题]
    H --> B

    F --> I[输出客服响应并结束]
```
