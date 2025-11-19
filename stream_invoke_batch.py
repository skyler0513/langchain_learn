# -*- coding: utf-8 -*-
# @author  : su yang
# @date    : 2025/11/7 10:21

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

parts = []
for s in chain.stream({"question": "give me a poem about sunrise"}):
    # 这里end必须等于"", 因为s.content已经包含了换行符，每流式输出一次再多加一个换行符的话会打乱格式
    print(s.content,end="",flush=True)
    # collect each streamed chunk
    parts.append(s.content)

# # combine parts into a single string and print
# resultStream = "".join(map(str, parts)).rstrip("\n")
# print(resultStream)
#
# resultInvoke = chain.invoke({"question": "give me a poem about sunrise"})
# print("--------------------------------")
# print(resultInvoke.content)

# result = chain.batch([{"question": "give me a poem about sunrise"}, {"question": "give me a poem about moon"}])
# for r in result:
#     print("-----")
#     print(r.content)

