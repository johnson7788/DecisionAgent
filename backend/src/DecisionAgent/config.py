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
# 切分投标书的Agent
SPLIT_BID_AGENT_CONFIG = {
    # "provider": "deepseek",
    # "model": "deepseek-chat",
    # "provider": "ali",
    "provider": "local_ali",
    "model": "qwen-turbo-latest",
    # "provider": "local_doubao",
    # "model": "doubao-seed-1-6-flash-250615",
    # "model": "doubao-seed-1-6-250615",
}
# 审计的Agent
AUDIT_AGENT_CONFIG = {
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
# 总结的Agent
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
