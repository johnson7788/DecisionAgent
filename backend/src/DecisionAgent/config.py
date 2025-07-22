#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/6/19 11:16
# @File  : config.py.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  :  GPU0服务器上部署时配置

##切分招标书的Agent
SPLIT_TENDER_AGENT_CONFIG = {
    # "provider": "deepseek",
    # "model": "deepseek-chat",
    # "provider": "ali",
    "provider": "local_ali",
    "model": "qwen-turbo-latest",
    # "provider": "local_doubao",
    # "model": "doubao-seed-1-6-flash-250615",
    # "model": "doubao-seed-1-6-250615",
}

TOPIC_RESEARCH_AGENT_CONFIG = {
    # "provider": "openai",
    # "provider": "ali",
    # "provider": "local_ali",
    # "model": "qwen-turbo-latest",
    # "provider": "doubao",
    # "model": "doubao-seed-1-6-flash-250615",
    # "model": "doubao-seed-1-6-250615",
    # "provider": "local_openai",
    "provider": "deepseek",
    "model": "deepseek-chat",
    # "provider": "google",
    # "provider": "local_ali",
    # "model": "qwen-turbo-latest",
    # "provider": "deepseek",
    # "model": "gpt-4.1",
    # "model": "gemini-2.0-flash",
    # "model": "gpt-4.1-nano-2025-04-14",
    # "model": "deepseek-chat",
}
# google gemini-2.0-flash summary效果不好
SUMMARY_AGENT_CONFIG = {
    # "provider": "local_ali",
    # "model": "qwen-turbo-latest",
    "provider": "deepseek",
    "model": "deepseek-chat",
    # "provider": "doubao",
    # "model": "doubao-seed-1-6-250615",
    # "model": "doubao-seed-1-6-flash-250615",
    # "provider": "openai",
    # "provider": "local_openai",
    # "provider": "local_google",
    # "model": "gpt-4.1",
    # "provider": "deepseek",
    # "model": "deepseek-chat",
    # "provider": "claude",
    # "model": "claude-sonnet-4-20250514",
    # "provider": "deepseek",
    # "model": "deepseek-chat",
    # "provider": "google",
    # "model": "gemini-2.5-pro",
    # "provider": "local_ali",
    # "model": "qwen-turbo-latest",
    # "model": "gemini-2.0-flash",
    # "provider": "local_ali",
    # "model": "qwen-turbo-latest",
}

# LLM模型，作为整合图片到文章
IMAGE_ANALYSIS_CONFIG = {
    "provider": "local_ali",
    "model": "qwen-turbo-latest",
    # "provider": "local_doubao",
    # "model": "doubao-seed-1-6-250615",
    # "model": "doubao-seed-1-6-flash-250615",
    # "provider": "openai",
    # "provider": "local_openai",
    # "model": "gpt-4.1",
    # "model": "gpt-4o-2024-08-06",
    # "provider": "local_ali",
    # "model": "qwen-turbo-latest",
}
# 多模态模型识别图片，作为工具
IMAGE_ANALYSIS_TOOL_CONFIG = {
    # "provider": "openai",
    "provider": "ali",
    "model": "qwen-vl-max-latest",
    # "model": "gpt-4o-2024-08-06",
    # "provider": "local_ali",
    # "model": "qwen-turbo-latest",
}

PPT_WRITER_AGENT_CONFIG = {
    # "provider": "openai",
    "provider": "deepseek",
    "model": "deepseek-chat",
    # "provider": "local_ali",
    # "model": "qwen-turbo-latest",
    # "provider": "doubao",
    # "model": "doubao-seed-1-6-250615",
    # "model": "doubao-seed-1-6-flash-250615",
    # "provider": "local_openai",
    # "model": "gpt-4.1",
    # "provider": "google",
    # "model": "gemini-2.0-flash",
    # "provider": "claude",
    # "model": "claude-sonnet-4-20250514",
    # "provider": "deepseek",
    # "model": "deepseek-chat",
    # "model": "gpt-4o-2024-08-06",
    # "provider": "local_ali",
    # "model": "qwen-turbo-latest",
}

# google gemini-2.0-flash refine效果不好
REFINE_AGENT_CONFIG = {
    # "provider": "openai",
    "provider": "deepseek",
    "model": "deepseek-chat",
    # "provider": "local_ali",
    # "model": "qwen-turbo-latest",
    # "provider": "doubao",
    # "model": "doubao-seed-1-6-250615",
    # "model": "doubao-seed-1-6-flash-250615",
    # "provider": "local_openai",
    # "model": "gpt-4.1",
    # "provider": "google",
    # "provider": "claude",
    # "model": "claude-sonnet-4-20250514",
    # "provider": "deepseek",
    # "model": "deepseek-chat",
    # "model": "gemini-2.5-pro",
    # "model": "gpt-4o-2024-08-06",
    # "model": "gpt-4.1-nano-2025-04-14",
    # "provider": "local_ali",
    # "model": "qwq-plus-latest",
    # "model": "qwen-turbo-latest",
}

PPT_CHECKER_AGENT_CONFIG = {
    # "provider": "openai",
    # "provider": "local_openai",
    # "model": "gpt-4.1",
    # "provider": "google",
    # "model": "gemini-2.0-flash",
    # "provider": "claude",
    # "model": "claude-sonnet-4-20250514",
    # "provider": "deepseek",
    # "model": "deepseek-chat",
    # "provider": "doubao",
    # "model": "doubao-seed-1-6-250615",
    # "model": "doubao-seed-1-6-flash-250615",
    # "model": "gpt-4o-2024-08-06",
    "provider": "local_ali",
    "model": "qwen-turbo-latest",
}

# PPT计划者
SLIDES_PLANNER_CONFIG = {
    # "provider": "openai",
    # "provider": "claude",
    # "provider": "local_ali",
    # "model": "qwen-turbo-latest",
    # "provider": "doubao",
    # "model": "doubao-seed-1-6-250615",
    # "model": "doubao-seed-1-6-flash-250615",
    # "provider": "local_openai",
    # "provider": "local_google",
    # "model": "gpt-4.1",
    "provider": "local_deepseek",
    "model": "deepseek-chat",
    # "provider": "google",
    # "model": "gemini-2.0-flash",
    # "provider": "local_ali",
    # "model": "qwen-turbo-latest",
    # "model": "gemini-2.0-flash",
    # "provider": "local_ali",
    # "model": "qwen-turbo-latest",
}
