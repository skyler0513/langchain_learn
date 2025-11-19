from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import BaseOutputParser


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
    api_key="sk-cea4c9c48b00423196325f1f4f5ad2f0",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-plus",  # 此处以qwen-plus为例，您可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
)

chain = chat_prompt | chatLLM | CommaSeparatedListOutputParser()
result = chain.invoke({"text": "colors"})
print(result)
