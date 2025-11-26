# Structured Output with LangChain Agent

This implementation demonstrates how to use structured output strategies with LangChain agents, specifically with the Qwen model via OpenAI-compatible API.

## Overview

The `agent.py` file showcases:

1. **ToolStrategy and ProviderStrategy**: Two approaches for enforcing structured outputs
2. **CustomMiddleware**: Implementation of before_agent and before_model hooks
3. **LangChain Debugging**: Enabling debug mode to inspect raw model calls
4. **Prompt Caching**: Configuration for Alibaba's Qwen model
5. **Message Conversion**: Proper role mapping from LangChain to OpenAI format

## Key Concepts

### ToolStrategy vs ProviderStrategy

**Note**: These are official LangChain classes from `langchain.agents.structured_output` module (available in langchain>=1.0.0).

- **ToolStrategy(ContactInfo)**: Enforces structured output from tool calls
  - Applied at invocation time via `response_format` parameter
  - Ensures tool returns match the specified schema
  
- **ProviderStrategy(ContactInfo)**: Enforces structured output from final agent response
  - Set during agent creation
  - Ensures the agent's final response matches the specified schema

- **Combined Usage**: Both strategies can be used together for full-chain structured output guarantee

### Message Role Mapping

LangChain internally uses different message types than OpenAI:

- LangChain: `HumanMessage` (type="human"), `AIMessage` (type="ai"), `SystemMessage` (type="system")
- OpenAI: "user", "assistant", "system"

The implementation provides `convert_messages_to_json()` function to handle this conversion.

### OpenAI Protocol Compatibility

Many Chinese LLM providers (Qwen, ChatGLM, Moonshot, etc.) implement OpenAI-compatible APIs. This allows:

- Using `ChatOpenAI` class with different `base_url` values
- Leveraging OpenAI's de facto standard API format
- Easy switching between different model providers

### Prompt Caching

For Qwen models, prompt caching is enabled via:

```python
ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("QWEN_PLUS_MODEL_NAME"),
    extra_body={"enable_cache": True}
)
```

Best practices:
- Extract system prompts as constants for exact matching
- Cache is valid for 5-10 minutes
- Cached tokens are billed at ~1/10 the normal rate
- Requires exact match (including whitespace)

### Middleware Execution Order

The `CustomMiddleware` class demonstrates the hook execution order:

1. **before_agent**: Executes FIRST - before agent processing starts
2. **before_model**: Executes SECOND - right before LLM call

## Usage Examples

### Example 1: Using ProviderStrategy Only

```python
from agent import create_agent_with_structured_output

agent = create_agent_with_structured_output()
response = agent.invoke({
    "messages": [{"role": "user", "content": "get me the weather in SF"}]
})
```

### Example 2: Using ToolStrategy at Invocation

```python
from agent import ContactInfo, ToolStrategy

response = agent.invoke({
    "messages": [{"role": "user", "content": "get me the weather in SF"}],
    "response_format": ToolStrategy(ContactInfo)
})
```

### Example 3: Using Both Strategies Together

```python
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy, ProviderStrategy
from agent import ContactInfo, chat_llm, get_weather, CustomMiddleware, SYSTEM_PROMPT

# Create agent with ProviderStrategy
agent = create_agent(
    model=chat_llm,
    tools=[get_weather],
    system_prompt=SYSTEM_PROMPT,
    response_format=ProviderStrategy(ContactInfo),  # Agent-level
    middleware=[CustomMiddleware()]
)

# Invoke with ToolStrategy
response = agent.invoke({
    "messages": [{"role": "user", "content": "get me the weather in SF"}],
    "response_format": ToolStrategy(ContactInfo)  # Tool-level
})
```

### Example 4: Converting Response to JSON

```python
from agent import format_response
import json

# Format the response with proper role mapping
formatted_response = format_response(response)

# Print as JSON
print(json.dumps(formatted_response, ensure_ascii=False, indent=2))
```

Expected output structure:
```json
{
  "structured_response": {
    "city": "SF",
    "weather": "sunny",
    "recommends": "visit the park"
  },
  "messages": [
    {"role": "user", "content": "get me the weather in SF"},
    {"role": "assistant", "content": "..."}
  ]
}
```

## Environment Setup

### Required Environment Variables

Create a `.env` file with:

```
API_KEY=your_api_key_here
BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_PLUS_MODEL_NAME=qwen-plus
```

### Dependencies

Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Minimum required packages:
- langchain>=1.0.0
- langchain-openai>=1.0.0
- pydantic>=2.0.0
- python-dotenv
- langgraph>=1.0.0

## Running the Demo

To run the demonstration script:

```bash
python agent.py
```

Note: The script creates and configures the agent but does not make actual API calls by default. To enable API calls, uncomment the invocation code in the `main()` function.

## Debugging

LangChain debugging is enabled in the script via:

```python
import langchain
langchain.debug = True
```

This will print detailed information about:
- Raw messages sent to the model
- Model responses
- Tool invocations
- State transitions

To disable debugging, set `langchain.debug = False` at the top of your script.

## Components

### ContactInfo Model

```python
class ContactInfo(BaseModel):
    city: str = Field(description="City name")
    weather: str = Field(description="Weather condition")
    recommends: str = Field(description="Recommendations for activities")
```

### get_weather Tool

```python
@tool
def get_weather(city: str) -> ContactInfo:
    """Get weather information and recommendations for a city."""
    return ContactInfo(
        city=city,
        weather="sunny",
        recommends="visit the park"
    )
```

### CustomMiddleware

```python
class CustomMiddleware(AgentMiddleware):
    def before_agent(self, state, runtime):
        """Executes FIRST - before agent processing starts."""
        print("=== BEFORE_AGENT: Initialization ===")
        return None
    
    def before_model(self, state, runtime):
        """Executes SECOND - right before LLM call."""
        print("=== BEFORE_MODEL: Preparing LLM call ===")
        return None
```

## Notes

1. The system prompt is extracted as a constant (`SYSTEM_PROMPT`) to enable proper prompt caching
2. Message type conversion is handled automatically by the utility functions
3. Both strategies can be used independently or together
4. The implementation is compatible with any OpenAI-compatible API endpoint
5. The agent uses `create_agent` from `langchain.agents` (not the deprecated `create_react_agent` from `langgraph.prebuilt`)

## Troubleshooting

### Issue: "ValidationError: model - Input should be a valid string"

**Solution**: Ensure environment variables are loaded correctly. Use `python-dotenv` to load from `.env` file:

```python
from dotenv import load_dotenv
load_dotenv()
```

### Issue: "ImportError: cannot import name 'create_agent'"

**Solution**: Ensure you have the latest version of langchain installed:

```bash
pip install --upgrade langchain langchain-openai
```

### Issue: TypeError when accessing message content

**Solution**: Use attribute access, not subscript notation:

```python
# Wrong
msg["content"]

# Correct
msg.content
msg.type
```
