# -*- coding: utf-8 -*-
# @author  : su yang
# @date    : 2025/11/26
# Implementing Structured Output with LangChain Agent using Tool and Provider Strategies
#
# This file demonstrates:
# 1. How to use ToolStrategy and ProviderStrategy for structured outputs
# 2. How to implement CustomMiddleware with before_agent and before_model hooks
# 3. How to enable LangChain debugging to inspect raw model calls
# 4. How to configure prompt caching for Qwen models
# 5. How to convert LangChain messages to JSON with proper role mapping
#
# KEY CONCEPTS:
#
# ToolStrategy vs ProviderStrategy:
# - ToolStrategy(ContactInfo): Enforces structured output from tool calls
# - ProviderStrategy(ContactInfo): Enforces structured output from final agent response
# - Both can be used together for full-chain structured output guarantee
#
# Message Role Mapping:
# - LangChain internally uses: HumanMessage (type="human"), AIMessage (type="ai"), SystemMessage (type="system")
# - OpenAI API uses: "user", "assistant", "system"
# - The ROLE_MAPPING dict converts between these formats for JSON serialization
#
# OpenAI Protocol Compatibility:
# - Qwen, ChatGLM, Moonshot, and other Chinese LLM providers implement OpenAI-compatible APIs
# - This allows using ChatOpenAI class with different base_url values
# - OpenAI API format has become a de facto standard, similar to HTTP
#
# Prompt Caching:
# - Enabled via extra_body={"enable_cache": True} for Qwen models
# - System prompts should be extracted as constants for exact matching
# - Cache is valid for 5-10 minutes
# - Cached tokens are billed at ~1/10 the normal rate
# - Requires exact match including whitespace
#
# Middleware Execution Order:
# 1. before_agent: Executes FIRST - before agent processing starts
# 2. before_model: Executes SECOND - right before LLM call

import os
import json
import langchain
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents.structured_output import ToolStrategy, ProviderStrategy
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain.tools import tool
from pydantic import BaseModel, Field
from typing import Any

# Load environment variables
load_dotenv()

# Enable LangChain debugging to inspect raw messages sent to the model
langchain.debug = True

# Define structured output model
class ContactInfo(BaseModel):
    """Contact information including weather and recommendations."""
    city: str = Field(description="City name")
    weather: str = Field(description="Weather condition")
    recommends: str = Field(description="Recommendations for activities")

# Define tool that returns ContactInfo
@tool
def get_weather(city: str) -> ContactInfo:
    """Get weather information and recommendations for a city.
    
    Args:
        city: The name of the city to get weather for.
    
    Returns:
        ContactInfo with weather and recommendations.
    """
    return ContactInfo(
        city=city,
        weather="sunny",
        recommends="visit the park"
    )

# Extract system prompt as constant for prompt caching
SYSTEM_PROMPT = "You are a helpful assistant"

# Configure ChatOpenAI with Qwen model and prompt caching enabled
chat_llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),  # Qwen-compatible OpenAI endpoint
    model=os.getenv("QWEN_PLUS_MODEL_NAME"),
    # Enable prompt caching for Qwen using extra_body parameter
    extra_body={"enable_cache": True}
)

# Role mapping for converting LangChain message types to OpenAI format
# LangChain uses: human/ai/system
# OpenAI uses: user/assistant/system
ROLE_MAPPING = {
    "human": "user",
    "ai": "assistant",
    "system": "system"
}

# Custom Middleware Implementation
class CustomMiddleware(AgentMiddleware):
    """Custom middleware with before_agent and before_model hooks.
    
    Middleware execution order:
    1. before_agent: Executes FIRST - before agent processing starts
    2. before_model: Executes SECOND - right before LLM call
    """
    
    def before_agent(self, state, runtime):
        """Executes FIRST - before agent processing starts."""
        print("=== BEFORE_AGENT: Initialization ===")
        print(f"State keys: {list(state.keys())}")
        if 'messages' in state:
            print(f"Number of messages: {len(state['messages'])}")
        return None
    
    def before_model(self, state, runtime):
        """Executes SECOND - right before LLM call."""
        print("=== BEFORE_MODEL: Preparing LLM call ===")
        print(f"State keys: {list(state.keys())}")
        if 'messages' in state:
            print(f"Messages being sent to model: {len(state['messages'])}")
        return None

def convert_messages_to_json(messages):
    """Convert LangChain messages to JSON format with proper role mapping.
    
    Args:
        messages: List of LangChain message objects
        
    Returns:
        List of dictionaries with 'role' and 'content' keys
    """
    return [
        {
            "role": ROLE_MAPPING.get(msg.type, msg.type),
            "content": msg.content
        }
        for msg in messages
    ]

def create_agent_with_structured_output():
    """Create an agent with both ProviderStrategy and ToolStrategy for structured output.
    
    This demonstrates:
    1. ProviderStrategy: Enforces structured output from the final agent response
    2. ToolStrategy: Can be used at invocation time for tool-level structured output
    3. CustomMiddleware: Provides before_agent and before_model hooks
    
    Returns:
        Compiled agent graph
    """
    
    # Create agent with ProviderStrategy for agent-level structured output
    agent = create_agent(
        model=chat_llm,
        tools=[get_weather],
        system_prompt=SYSTEM_PROMPT,
        response_format=ProviderStrategy(ContactInfo),  # Agent-level structured output
        middleware=[CustomMiddleware()],  # Custom middleware with hooks
        debug=True  # Enable graph-level debugging
    )
    
    return agent

def invoke_agent_with_tool_strategy(agent):
    """Invoke agent with ToolStrategy for tool-level structured output.
    
    Args:
        agent: The compiled agent graph
        
    Returns:
        Response dictionary with structured_response and messages
    """
    # Invoke with both strategies:
    # - ProviderStrategy (set at agent creation): enforces agent response structure
    # - ToolStrategy (set at invocation): enforces tool call structure
    response = agent.invoke(
        {
            "messages": [{"role": "user", "content": "get me the weather in SF"}],
            "response_format": ToolStrategy(ContactInfo)  # Tool-level structured output
        }
    )
    
    return response

def format_response(response):
    """Format agent response to JSON with proper role mapping.
    
    Args:
        response: Agent response dictionary
        
    Returns:
        Formatted dictionary ready for JSON serialization
    """
    formatted = {}
    
    # Include structured response if available
    if "structured_response" in response:
        structured = response["structured_response"]
        if isinstance(structured, BaseModel):
            formatted["structured_response"] = structured.dict()
        else:
            formatted["structured_response"] = structured
    
    # Convert messages with role mapping
    if "messages" in response:
        formatted["messages"] = convert_messages_to_json(response["messages"])
    
    return formatted

def main():
    """Main execution function demonstrating the agent usage.
    
    This example shows:
    1. Creating an agent with ProviderStrategy (agent-level structured output)
    2. Invoking with ToolStrategy (tool-level structured output)
    3. Both strategies working together for end-to-end structured output
    4. Middleware hooks executing in proper order
    5. Converting response to JSON format
    """
    print("=" * 80)
    print("Creating Agent with Structured Output (ProviderStrategy + ToolStrategy)")
    print("=" * 80)
    print("\nNote: This example creates an agent but does NOT make actual API calls.")
    print("To run with real API calls, uncomment the invocation code below.")
    print("Middleware hooks and debugging are enabled to show execution flow.\n")
    
    # Create agent
    agent = create_agent_with_structured_output()
    print(f"\n✓ Agent created successfully: {agent.__class__.__name__}")
    
    print("\n" + "=" * 80)
    print("Example: How to Invoke Agent with Both Strategies")
    print("=" * 80)
    print("""
# EXAMPLE 1: Using ProviderStrategy (set at agent creation)
# This enforces structured output from the final agent response
response = agent.invoke({
    "messages": [{"role": "user", "content": "get me the weather in SF"}]
})

# EXAMPLE 2: Using ToolStrategy at invocation time
# This enforces structured output from tool calls
response = agent.invoke({
    "messages": [{"role": "user", "content": "get me the weather in SF"}],
    "response_format": ToolStrategy(ContactInfo)
})

# EXAMPLE 3: Using both strategies together
# - ProviderStrategy (from agent creation): enforces agent response structure
# - ToolStrategy (from invocation): enforces tool call structure
# This provides full-chain structured output guarantee
agent_with_provider = create_agent(
    model=chat_llm,
    tools=[get_weather],
    system_prompt=SYSTEM_PROMPT,
    response_format=ProviderStrategy(ContactInfo),  # Agent-level
    middleware=[CustomMiddleware()]
)

response = agent_with_provider.invoke({
    "messages": [{"role": "user", "content": "get me the weather in SF"}],
    "response_format": ToolStrategy(ContactInfo)  # Tool-level
})
    """)
    
    print("\n" + "=" * 80)
    print("Example: Converting Response to JSON")
    print("=" * 80)
    print("""
# Format the response with proper role mapping
formatted_response = format_response(response)

# Print as JSON
import json
print(json.dumps(formatted_response, ensure_ascii=False, indent=2))

# The output will have:
# {
#   "structured_response": {
#     "city": "SF",
#     "weather": "sunny",
#     "recommends": "visit the park"
#   },
#   "messages": [
#     {"role": "user", "content": "get me the weather in SF"},
#     {"role": "assistant", "content": "..."}
#   ]
# }
    """)
    
    # Uncomment below to make actual API call (requires valid API key)
    # print("\n" + "=" * 80)
    # print("Invoking Agent with User Query (Uncomment to run)")
    # print("=" * 80)
    # response = invoke_agent_with_tool_strategy(agent)
    # formatted_response = format_response(response)
    # print(json.dumps(formatted_response, ensure_ascii=False, indent=2))
    
    print("\n" + "=" * 80)
    print("Setup Complete - Ready to Use")
    print("=" * 80)
    print("\n✓ Agent is configured with:")
    print("  - ProviderStrategy for agent-level structured output")
    print("  - ToolStrategy available for tool-level structured output")
    print("  - CustomMiddleware with before_agent and before_model hooks")
    print("  - LangChain debugging enabled (langchain.debug = True)")
    print("  - Prompt caching enabled for Qwen model")
    print("  - JSON serialization with role mapping")
    print("\n✓ To make API calls, uncomment the invocation code in main()")

if __name__ == "__main__":
    main()
