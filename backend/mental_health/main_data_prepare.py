#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/7/3 17:34
# @File  : main_data_prepare.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  : 准备数据，数据进行向量化
import os
from dotenv import load_dotenv
from DecisionAgent.embedding_utils import EmbeddingModel,ChromaDB
from data import education_data
load_dotenv()
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "education_data")
print(f"使用的向量表是: {COLLECTION_NAME}")
def get_matches_for_embedding():
    documents = [item["matches"] for item in education_data]
    metadatas = [{"name": item["name"]} for item in education_data]
    return documents, metadatas

if __name__ == '__main__':
    embedder = EmbeddingModel()
    chromadb_instance = ChromaDB(embedder=embedder)

    # 1. 获取要向量化的 documents 和对应的 metadata
    documents_to_embed, metadatas_for_documents = get_matches_for_embedding()

    # 2. 定义 ChromaDB 的 collection 名称
    # 3. 删除旧的 collection (可选，用于清空数据)
    print(f"尝试删除 collection: {COLLECTION_NAME}")
    chromadb_instance.delete_one_collection(COLLECTION_NAME)

    # 4. 将 matches 字段向量化并插入到 ChromaDB，name 字段作为 meta 信息
    print(f"开始插入数据到 collection: {COLLECTION_NAME}")
    insert_status = chromadb_instance.insert2collection(
        collection=COLLECTION_NAME,
        documents=documents_to_embed,
        meta=metadatas_for_documents
    )
    print(f"数据插入状态: {insert_status}")

    # 5. 列出所有已有的 collections，确认新的 collection 已创建
    print("\n现有 collections:")
    print(chromadb_instance.list_exist_collections())

    # 6. 列出新 collection 的内容，检查 documents 和 metadatas
    print(f"\nCollection '{COLLECTION_NAME}' 的内容:")
    print(chromadb_instance.list_collection(COLLECTION_NAME, number=len(documents_to_embed)))

    # 7. 进行查询测试
    query_text = "我喜欢画画，以后想学艺术相关的专业"
    print(f"\n查询文本: '{query_text}'")
    query_results = chromadb_instance.query2collection(
        collection=COLLECTION_NAME,
        query_documents=[query_text],
        topk=2
    )
    print("查询结果:")
    for i in range(len(query_results["documents"][0])):
        doc = query_results["documents"][0][i]
        meta = query_results["metadatas"][0][i]
        distance = query_results["distances"][0][i]
        print(f"  文档: {doc[:50]}... (截断)")
        print(f"  名称: {meta.get('name')}")
        print(f"  距离: {distance:.4f}\n")
