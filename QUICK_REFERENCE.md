# Quick Reference Guide - Structured Output with LangChain Agent

## Quick Start

```python
from agent import create_agent_with_structured_output

# Create agent with structured output
agent = create_agent_with_structured_output()

# Invoke agent
response = agent.invoke({
    "messages": [{"role": "user", "content": "get me the weather in SF"}]
})
```

## Key Concepts

### Strategy Types

| Strategy | Level | When to Use |
|----------|-------|-------------|
| **ProviderStrategy** | Agent-level | Enforce structure on final agent response |
| **ToolStrategy** | Tool-level | Enforce structure on tool call outputs |

### Usage Patterns

**Pattern 1: Provider Only**
```python
agent = create_agent(
    model=chat_llm,
    tools=[get_weather],
    response_format=ProviderStrategy(ContactInfo)  # Agent enforces structure
)
```

**Pattern 2: Tool Only**
```python
response = agent.invoke({
    "messages": [...],
    "response_format": ToolStrategy(ContactInfo)  # Tool enforces structure
})
```

**Pattern 3: Both (Recommended)**
```python
# At creation
agent = create_agent(
    response_format=ProviderStrategy(ContactInfo)
)

# At invocation
response = agent.invoke({
    "messages": [...],
    "response_format": ToolStrategy(ContactInfo)
})
```

## Components

### ContactInfo Model
```python
class ContactInfo(BaseModel):
    city: str
    weather: str
    recommends: str
```

### get_weather Tool
```python
@tool
def get_weather(city: str) -> ContactInfo:
    """Get weather information for a city."""
    return ContactInfo(...)
```

### CustomMiddleware Hooks
```python
class CustomMiddleware(AgentMiddleware):
    def before_agent(self, state, runtime):
        # Runs FIRST - initialization
        return None
    
    def before_model(self, state, runtime):
        # Runs SECOND - pre-LLM call
        return None
```

## Message Role Mapping

| LangChain Type | OpenAI Format |
|----------------|---------------|
| `HumanMessage` (type="human") | "user" |
| `AIMessage` (type="ai") | "assistant" |
| `SystemMessage` (type="system") | "system" |

```python
# Convert messages
from agent import convert_messages_to_json

messages = [HumanMessage(content="Hello")]
json_messages = convert_messages_to_json(messages)
# [{"role": "user", "content": "Hello"}]
```

## Response Formatting

```python
from agent import format_response
import json

# Format response
formatted = format_response(response)

# Print as JSON
print(json.dumps(formatted, ensure_ascii=False, indent=2))
```

Output structure:
```json
{
  "structured_response": {
    "city": "SF",
    "weather": "sunny",
    "recommends": "visit the park"
  },
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

## Configuration

### Enable Debugging
```python
import langchain
langchain.debug = True  # Show raw model calls
```

### Prompt Caching (Qwen)
```python
ChatOpenAI(
    extra_body={"enable_cache": True}
)

# Extract system prompt as constant
SYSTEM_PROMPT = "You are a helpful assistant"
```

### Environment Variables
```bash
# .env file
API_KEY=your_api_key
BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_PLUS_MODEL_NAME=qwen-plus
```

## Common Tasks

### Create Agent with All Features
```python
from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy
from agent import CustomMiddleware, chat_llm, get_weather, ContactInfo, SYSTEM_PROMPT

agent = create_agent(
    model=chat_llm,
    tools=[get_weather],
    system_prompt=SYSTEM_PROMPT,
    response_format=ProviderStrategy(ContactInfo),
    middleware=[CustomMiddleware()],
    debug=True
)
```

### Invoke with Tool Strategy
```python
from langchain.agents.structured_output import ToolStrategy

response = agent.invoke({
    "messages": [{"role": "user", "content": "get weather in SF"}],
    "response_format": ToolStrategy(ContactInfo)
})
```

### Convert and Display Response
```python
from agent import format_response
import json

formatted = format_response(response)
print(json.dumps(formatted, ensure_ascii=False, indent=2))
```

## Testing

```bash
# Run all tests
python test_agent.py

# Run demo
python agent.py
```

## Troubleshooting

### Environment variables not loading
```python
from dotenv import load_dotenv
load_dotenv()
```

### Message access errors
```python
# Wrong
msg["content"]  # TypeError

# Correct
msg.content  # Use attribute access
msg.type
```

### Import errors
```bash
pip install --upgrade langchain langchain-openai
```

## File Organization

```
langchain_learn/
├── agent.py                   # Main implementation
├── AGENT_README.md           # Full documentation
├── test_agent.py             # Test suite
├── QUICK_REFERENCE.md        # This file
├── IMPLEMENTATION_SUMMARY.md # Project summary
├── requirements.txt          # Dependencies
└── .env                      # Environment variables
```

## Next Steps

1. Set up `.env` with your API credentials
2. Run `python test_agent.py` to verify setup
3. Run `python agent.py` to see demo output
4. Uncomment invocation code in `main()` to make real API calls
5. Refer to `AGENT_README.md` for detailed examples

## Reference

- **LangChain Docs**: https://python.langchain.com/docs/
- **Qwen API**: https://help.aliyun.com/document_detail/2712195.html
- **Pydantic**: https://docs.pydantic.dev/

---

For detailed explanations and troubleshooting, see `AGENT_README.md`
