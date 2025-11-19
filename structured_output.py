# -*- coding: utf-8 -*-
# @author  : su yang
# @date    : 2025/11/13 15:19

import os
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

class Joke(BaseModel):
    setup: str = Field(description="笑话的设置部分")
    punchline: str = Field(description="笑话的结尾部分")

chat_llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("QWEN_PLUS_MODEL_NAME"),
).with_structured_output(Joke, method="json_mode")

prompt = "给我讲一个关于狗的笑话。请以 JSON 格式返回，并确保 JSON 对象包含 'setup' 和 'punchline' 两个键。"
# 这里必须使用model_dump_json函数，不然返回的是pydantic的对象
result = chat_llm.invoke(prompt).model_dump_json(indent=2, ensure_ascii=False)
print(result)