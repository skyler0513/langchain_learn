# -*- coding: utf-8 -*-
# @author  : su yang
# @date    : 2025/11/10 18:24
import os

from langchain_core.globals import set_llm_cache
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import BaseOutputParser
from langchain_community.cache import InMemoryCache
import time

class CommaSeparatedListOutputParser(BaseOutputParser):
    """Parse the output of an LLM call to a comma-separated list."""

    def parse(self, text: str):
        """Parse the output of an LLM call."""
        return text.strip().split(", ")


template = """You are a helpful assistant who generates comma separated lists.
A user will pass in a category, and you should generate 5 objects in that category in a comma separated list.
ONLY return a comma separated list, and nothing more."""

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    ("human", "{text}")
])

chatLLM = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("QWEN_PLUS_MODEL_NAME"),  # 此处以qwen-plus为例，您可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
)

set_llm_cache(InMemoryCache())
chain = chat_prompt | chatLLM | CommaSeparatedListOutputParser()

# 第一次调用（会发起 API 请求）
start = time.time()
result1 = chain.invoke({"text": "colors"})
print(f"第一次调用: {time.time() - start:.2f}秒")
print(result1)

# 第二次调用（从缓存返回，应该很快）
start = time.time()
result2 = chain.invoke({"text": "colors"})
print(f"第二次调用: {time.time() - start:.2f}秒")
print(result2)