# -*- coding: utf-8 -*-
# @author  : su yang
# @date    : 2025/11/26
# Implementing Structured Output with LangChain Agent using Tool and Provider Strategies

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
    """Main execution function demonstrating the agent usage."""
    print("=" * 80)
    print("Creating Agent with Structured Output (ProviderStrategy + ToolStrategy)")
    print("=" * 80)
    
    # Create agent
    agent = create_agent_with_structured_output()
    
    print("\n" + "=" * 80)
    print("Invoking Agent with User Query")
    print("=" * 80)
    
    # Invoke agent
    response = invoke_agent_with_tool_strategy(agent)
    
    print("\n" + "=" * 80)
    print("Formatting and Displaying Response")
    print("=" * 80)
    
    # Format response
    formatted_response = format_response(response)
    
    # Print as JSON
    print(json.dumps(formatted_response, ensure_ascii=False, indent=2))
    
    print("\n" + "=" * 80)
    print("Execution Complete")
    print("=" * 80)

if __name__ == "__main__":
    main()
