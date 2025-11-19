# -*- coding: utf-8 -*-
# @author  : su yang
# @date    : 2025/11/7 10:52
import asyncio

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI


template = """Answer the question:

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chatLLM = ChatOpenAI(
    api_key="sk-cea4c9c48b00423196325f1f4f5ad2f0",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-plus",  # 此处以qwen-plus为例，您可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
)

chain = prompt | chatLLM

async def chain_astream_example():
    async for s in chain.astream({"question": "give me a poem about sunrise"}):
        print(s.content, end="", flush=True)

async def chain_ainvoke_example():
    s = await chain.ainvoke({"question": "give me a poem about sunrise"})
    print(s.content, end="", flush=True)

async def chain_batch_example():
    s = await chain.abatch([{"question": "give me a poem about sunrise"}, {"question": "give me a poem about moon"}])
    for r in s:
        print("-----")
        print(r.content)


asyncio.run(chain_batch_example())