# -*- coding: utf-8 -*-
# @author  : su yang
# @date    : 2025/11/6 17:12

from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.embeddings import DashScopeEmbeddings

embeddings = DashScopeEmbeddings(
    dashscope_api_key="sk-cea4c9c48b00423196325f1f4f5ad2f0",
    model="text-embedding-v2"  # 使用 v2 版本更稳定
)
vectorstore = DocArrayInMemorySearch.from_texts(
    ["harrison worked at kensho", "bears like to eat honey"],
    embedding=embeddings,
)
retriever = vectorstore.as_retriever()

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chatLLM = ChatOpenAI(
    api_key="sk-cea4c9c48b00423196325f1f4f5ad2f0",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-plus",  # 此处以qwen-plus为例，您可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
)
output_parser = StrOutputParser()

setup_and_retrieval = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
)
chain = setup_and_retrieval | prompt | chatLLM | output_parser

# result = chain.invoke("where did harrison work?")
# print(result)
print(chain.input_schema.model_json_schema())
print(setup_and_retrieval.input_schema.model_json_schema())
print(prompt.input_schema.model_json_schema())
print(chain.output_schema.model_json_schema())
# print(chatLLM.input_schema.model_json_schema())
# print(output_parser.input_schema.model_json_schema())

