# -*- coding: utf-8 -*-
# @author  : su yang
# @date    : 2025/11/13 14:03

import os
import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers.openai_tools import PydanticToolsParser

from pydantic import BaseModel, Field

# 注意，这里的文档字符串非常重要，因为它们将与类名一起传递给模型。
class Add(BaseModel):
    """将两个整数相加。"""

    a: int = Field(..., description="第一个整数")
    b: int = Field(..., description="第二个整数")


class Multiply(BaseModel):
    """将两个整数相乘。"""

    a: int = Field(..., description="第一个整数")
    b: int = Field(..., description="第二个整数")

def add(a: int, b: int) -> int:
    return a + b

def multiply(a: int, b: int) -> int:
    return a * b

tools = [Add, Multiply]
tool_map = {
    "Add": add,
    "Multiply": multiply,
}

chat_llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("QWEN_PLUS_MODEL_NAME"),
)



query = "3 * 12是多少？此外，11 + 49是多少？"

def invoke_chain():
    llm_with_tools = chat_llm.bind_tools(tools) | PydanticToolsParser(
        tools=tools)

    chain = llm_with_tools
    result = chain.invoke(query)
    return result

async def astream_chain():
    llm_with_tools = chat_llm.bind_tools(tools)

    first = True
    gathered = None
    async for chunk in llm_with_tools.astream(query):
        if first:
            gathered = chunk
            first = False
        else:
            gathered = gathered + chunk
    parser = PydanticToolsParser(tools=tools)
    result = parser.invoke(gathered)
    return result



result = asyncio.run(astream_chain())


# 3. 运行链并获取解析后的工具调用
parsed_tool_calls = asyncio.run(astream_chain())
print(f"模型解析出的工具调用: {parsed_tool_calls}")

# 4. 遍历并执行每个工具调用
for tool_call in parsed_tool_calls:
    # 获取工具名称，例如 "Add" 或 "Multiply"
    tool_name = tool_call.__class__.__name__
    # 从映射中查找对应的函数
    selected_tool = tool_map[tool_name]
    # 将 Pydantic 模型转换为字典作为函数参数
    tool_output = selected_tool(**tool_call.model_dump())
    print(f"执行工具 '{tool_name}'，参数为 {tool_call.model_dump()}，结果: {tool_output}")
