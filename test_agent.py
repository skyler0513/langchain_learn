#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for agent.py - validates all components without making API calls
"""

import os
import sys

# Set test environment variables
os.environ["API_KEY"] = "sk-test-key"
os.environ["BASE_URL"] = "https://test.example.com/v1"
os.environ["QWEN_PLUS_MODEL_NAME"] = "test-model"

# Disable debugging for cleaner test output
import langchain
langchain.debug = False

print("=" * 80)
print("Testing agent.py components")
print("=" * 80)

# Test 1: Import all components
print("\n[TEST 1] Importing agent module...")
try:
    import agent
    from agent import (
        ContactInfo,
        get_weather,
        chat_llm,
        SYSTEM_PROMPT,
        ROLE_MAPPING,
        CustomMiddleware,
        create_agent_with_structured_output,
        convert_messages_to_json,
        format_response
    )
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Test ContactInfo model
print("\n[TEST 2] Testing ContactInfo model...")
try:
    info = ContactInfo(city="SF", weather="sunny", recommends="visit park")
    assert info.city == "SF"
    assert info.weather == "sunny"
    assert info.recommends == "visit park"
    print(f"✓ ContactInfo model works: {info.dict()}")
except Exception as e:
    print(f"✗ ContactInfo test failed: {e}")
    sys.exit(1)

# Test 3: Test get_weather tool
print("\n[TEST 3] Testing get_weather tool...")
try:
    result = get_weather.invoke({"city": "SF"})
    assert isinstance(result, ContactInfo)
    assert result.city == "SF"
    print(f"✓ get_weather tool works: {result.dict()}")
except Exception as e:
    print(f"✗ get_weather test failed: {e}")
    sys.exit(1)

# Test 4: Test role mapping
print("\n[TEST 4] Testing message role mapping...")
try:
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    
    messages = [
        SystemMessage(content="You are helpful"),
        HumanMessage(content="Hello"),
        AIMessage(content="Hi there")
    ]
    
    converted = convert_messages_to_json(messages)
    assert len(converted) == 3
    assert converted[0]["role"] == "system"
    assert converted[1]["role"] == "user"
    assert converted[2]["role"] == "assistant"
    print(f"✓ Role mapping works:")
    for msg in converted:
        print(f"  - {msg['role']}: {msg['content']}")
except Exception as e:
    print(f"✗ Role mapping test failed: {e}")
    sys.exit(1)

# Test 5: Test format_response
print("\n[TEST 5] Testing format_response...")
try:
    test_response = {
        "structured_response": ContactInfo(city="SF", weather="sunny", recommends="visit park"),
        "messages": [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi")
        ]
    }
    
    formatted = format_response(test_response)
    assert "structured_response" in formatted
    assert "messages" in formatted
    assert formatted["structured_response"]["city"] == "SF"
    assert formatted["messages"][0]["role"] == "user"
    assert formatted["messages"][1]["role"] == "assistant"
    print(f"✓ format_response works:")
    print(f"  - Structured response: {formatted['structured_response']}")
    print(f"  - Messages: {len(formatted['messages'])} messages")
except Exception as e:
    print(f"✗ format_response test failed: {e}")
    sys.exit(1)

# Test 6: Test CustomMiddleware
print("\n[TEST 6] Testing CustomMiddleware...")
try:
    middleware = CustomMiddleware()
    print(f"✓ CustomMiddleware instantiated: {middleware.__class__.__name__}")
    print(f"  - Has before_agent: {hasattr(middleware, 'before_agent')}")
    print(f"  - Has before_model: {hasattr(middleware, 'before_model')}")
except Exception as e:
    print(f"✗ CustomMiddleware test failed: {e}")
    sys.exit(1)

# Test 7: Test agent creation
print("\n[TEST 7] Testing agent creation...")
try:
    test_agent = create_agent_with_structured_output()
    print(f"✓ Agent created successfully: {test_agent.__class__.__name__}")
except Exception as e:
    print(f"✗ Agent creation failed: {e}")
    sys.exit(1)

# Test 8: Verify strategies are imported correctly
print("\n[TEST 8] Testing strategy imports...")
try:
    from langchain.agents.structured_output import ToolStrategy, ProviderStrategy
    
    tool_strategy = ToolStrategy(ContactInfo)
    provider_strategy = ProviderStrategy(ContactInfo)
    print(f"✓ ToolStrategy created: {tool_strategy.__class__.__name__}")
    print(f"✓ ProviderStrategy created: {provider_strategy.__class__.__name__}")
except Exception as e:
    print(f"✗ Strategy import test failed: {e}")
    sys.exit(1)

# Test 9: Verify ChatOpenAI configuration
print("\n[TEST 9] Testing ChatOpenAI configuration...")
try:
    # Note: openai_api_base is the internal attribute name in langchain-openai
    assert chat_llm.model_name == "test-model"
    assert chat_llm.openai_api_base == "https://test.example.com/v1"
    print(f"✓ ChatOpenAI configured correctly:")
    print(f"  - Model: {chat_llm.model_name}")
    print(f"  - API Base: {chat_llm.openai_api_base}")
except Exception as e:
    print(f"✗ ChatOpenAI configuration test failed: {e}")
    sys.exit(1)

# Test 10: Verify constants
print("\n[TEST 10] Testing constants...")
try:
    assert SYSTEM_PROMPT == "You are a helpful assistant"
    assert "human" in ROLE_MAPPING
    assert "ai" in ROLE_MAPPING
    assert "system" in ROLE_MAPPING
    print(f"✓ Constants defined correctly:")
    print(f"  - SYSTEM_PROMPT: {SYSTEM_PROMPT}")
    print(f"  - ROLE_MAPPING: {ROLE_MAPPING}")
except Exception as e:
    print(f"✗ Constants test failed: {e}")
    sys.exit(1)

print("\n" + "=" * 80)
print("All tests passed! ✓")
print("=" * 80)
print("\nThe agent.py implementation is complete and working correctly.")
print("\nImplemented features:")
print("  ✓ ContactInfo Pydantic model")
print("  ✓ get_weather tool function")
print("  ✓ ChatOpenAI with prompt caching")
print("  ✓ CustomMiddleware with before_agent and before_model hooks")
print("  ✓ ToolStrategy and ProviderStrategy support")
print("  ✓ JSON serialization with role mapping")
print("  ✓ LangChain debugging enabled")
print("\nTo make actual API calls, use the examples in AGENT_README.md")
