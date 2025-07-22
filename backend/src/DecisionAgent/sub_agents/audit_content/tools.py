#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/6/20 10:02
# @File  : tools.py.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  : 搜索图片，用于PPT的配图

import re
import os
import time
from datetime import datetime
import random
import hashlib
from pathlib import Path
from fastmcp import FastMCP
from google.adk.tools import ToolContext
from google.adk.tools.agent_tool import AgentTool
import requests
from urllib.parse import quote

def search_pixabay_images(query, image_type="all", per_page=5, safesearch=True):
    """

    参数:
    返回:
        List[dict]: 包含图片信息的字典列表，包括预览图、作者信息、原图等
    """
    pass

async def SearchImages(query: str, tool_context: ToolContext) -> list[dict]:
    """
    搜索非医学相关的图片，用于PPT的背景，等配图
    :return: 如果搜到了返回图片列表，否则返回空列表
    """
    images = [
        {
            "Description": "背景图1",
            "URL": "https://www.slideegg.com/image/catalog/700940-powerpoint-background-templates-medical.png",
        },
        {
            "Description": "背景图2",
            "URL": "https://img.freepik.com/free-vector/clean-medical-background_53876-97927.jpg",
        },
        {
            "Description": "背景图3",
            "URL": "https://slidescorner.com/wp-content/uploads/2022/11/01-Drake-Healthcare-and-Medical-Free-PPT-Background-by-SlidesCorner.com_.jpg",
        },
        {
            "Description": "背景图4",
            "URL": "https://slidescorner.com/wp-content/uploads/2023/03/02-6.jpg",
        },
        {
            "Description": "背景图5",
            "URL": "https://file.51pptmoban.com/d/file/2022/08/18/6fe3db16faacb7367706e2e9055819da.jpg",
        }
    ]
    return images


if __name__ == '__main__':
    result = search_pixabay_images("flowers", "all", 5, True)
    print(result)
