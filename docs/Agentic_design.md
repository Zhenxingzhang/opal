# Agentic Design - llm_agents

## Architecture Overview

The llm_agents codebase implements a **modular agentic framework** with three main layers:

1. **Agentic Layer** - LLM models and agent logic
2. **Environment Layer** - Session/step management and trajectory tracking
3. **Orchestration Layer** - Runner that coordinates everything

```
┌─────────────────────────────────────────────────────────────┐
│                        Runner                                │
│                   (Orchestration Layer)                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────────┐    ┌─────────────────────────────┐ │
│  │    Agentic Layer    │    │    Environment Layer        │ │
│  │                     │    │                             │ │
│  │  ┌───────────────┐  │    │  ┌─────────┐  ┌─────────┐  │ │
│  │  │    Agent      │  │◄───┤  │ Session │  │  Step   │  │ │
│  │  │ (ReAct/Default│  │    │  └─────────┘  └─────────┘  │ │
│  │  └───────────────┘  │    │                             │ │
│  │         │           │    └─────────────────────────────┘ │
│  │         ▼           │                                    │
│  │  ┌───────────────┐  │                                    │
│  │  │   LLMModel    │  │                                    │
│  │  │(OpenAI/Claude)│  │                                    │
│  │  └───────────────┘  │                                    │
│  │         │           │                                    │
│  │         ▼           │                                    │
│  │  ┌───────────────┐  │                                    │
│  │  │    Tools      │  │                                    │
│  │  └───────────────┘  │                                    │
│  └─────────────────────┘                                    │
└─────────────────────────────────────────────────────────────┘
```

## Key Abstractions

### 1. Agent (`llm_agents/agentic/agent.py`)

**Base Class: `Agent` (Abstract)**
- Defines the interface for all agents
- Key methods:
  - `run(user_query, session)` - Main execution loop (abstract)
  - `act(messages, session)` - Call the LLM with message history
  - `execute_tool(name, arguments_json)` - Execute tools by name

**Implementations:**

| Agent | Description | Use Case |
|-------|-------------|----------|
| `DefaultAgent` | Single LLM call, no tools | Simple completions, summarization |
| `ReActAgent` | Reasoning + Acting loop with tools | Complex tasks requiring multiple steps |

### 2. LLM Model (`llm_agents/agentic/llm_model.py`)

**Base Class: `LLMModel` (Abstract)**
- Standardizes interface across providers
- Core method: `call(messages, tools) -> ModelResponse`

**Implementations:**
- `OpenAIModel` - GPT-4, GPT-4o, etc.
- `AnthropicModel` - Claude models

**Data Classes:**
```python
@dataclass
class ToolCallInfo:
    id: str
    name: str
    arguments: str  # JSON string

@dataclass
class ModelResponse:
    content: str | None
    tool_calls: list[ToolCallInfo] | None
    raw_response: Any
```

### 3. Tool (`llm_agents/agentic/tool.py`)

```python
@dataclass
class Tool:
    name: str           # Tool identifier
    description: str    # Human-readable description for LLM
    parameters: dict    # JSON-schema style parameter spec
    function: Callable  # Actual Python function to call
```

**Built-in Tools:**
- `CALCULATOR_TOOL` - Math expression evaluation
- `SEARCH_PDF_TOOL` - Semantic search over PDF content
- `READ_PDF_TOOL` - PDF text extraction (PyMuPDF)
- `SEARCH_WEB_TOOL` - Web search (DuckDuckGo)

### 4. Session & Step (`llm_agents/environment/`)

**Session** - Manages conversation state:
- `id` - Unique session identifier (UUID)
- `trajectory` - List of Steps
- `metadata` - Execution context (timestamp, status, etc.)

**Step** - Represents one conversational turn:
- `role` - "user", "assistant", or "tool"
- `content` - Text message content
- `tool_call` - Dict with {id, name, arguments}
- `tool_result` - Tool execution output

### 5. Runner (`llm_agents/runner.py`)

Orchestrates agent execution:
- Creates and manages Session
- Builds appropriate Agent based on config
- Provides access to results and trajectory

**AgentConfig:**
```python
@dataclass
class AgentConfig:
    model_name: str = "gpt-4o-2024-11-20"
    agent_name: str = "react"  # "react" or "default"
    max_steps: int = 10
    log_llm_calls: bool = False
    tools: list[Tool] = field(default_factory=list)
```

## Data Flow

### ReAct Loop Flow

```
User Query
    │
    ▼
┌─────────────────────────────────────┐
│ Session.reset() → new UUID          │
│ Add Step(role="user", content=query)│
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ For each step (0 to max_steps):     │
│                                     │
│   1. Build messages from trajectory │
│   2. Call LLM via act()             │
│   3. Check response type:           │
│                                     │
│   ┌─────────────────────────────┐   │
│   │ Tool call present?          │   │
│   │   YES → Execute tool        │   │
│   │         Add observation     │   │
│   │         Continue loop       │   │
│   │   NO  → Return final answer │   │
│   └─────────────────────────────┘   │
└─────────────────────────────────────┘
    │
    ▼
Final Answer + Trajectory
```

### Message Building

```
Trajectory Steps              LLM Message Format
────────────────              ──────────────────
Step(role="user")      →      {"role": "user", "content": "..."}

Step(role="assistant", →      {"role": "assistant",
     tool_call={...})          "content": "...",
                               "tool_calls": [...]}

Step(role="tool",      →      {"role": "tool",
     tool_result="...")        "tool_call_id": "...",
                               "content": "..."}
```

## Tool Execution

### Definition Pattern

```python
# 1. Define implementation function
def _calculator(expression: str) -> str:
    result = eval(expression, {"__builtins__": {}})
    return str(result)

# 2. Create Tool with schema
CALCULATOR_TOOL = Tool(
    name="calculator",
    description="Evaluate a mathematical expression",
    parameters={
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "..."}
        },
        "required": ["expression"],
    },
    function=_calculator,
)
```

### Execution Flow

```
LLM Response with tool_call
    │
    ▼
ToolCallInfo: {id, name, arguments_json}
    │
    ▼
Agent.execute_tool(name, arguments_json)
    │
    ├─► Look up tool by name in _tool_map
    ├─► Parse arguments_json to dict
    ├─► Call tool.function(**args)
    ├─► Convert result to string
    │
    ▼
Return result as observation
```

## Usage Example

```python
from llm_agents import Runner, AgentConfig
from llm_agents.environment.builtin_tools import CALCULATOR_TOOL, SEARCH_WEB_TOOL

# Configure agent
config = AgentConfig(
    model_name="gpt-4o",
    agent_name="react",
    max_steps=10,
    tools=[CALCULATOR_TOOL, SEARCH_WEB_TOOL],
    log_llm_calls=True,
)

# Run
runner = Runner(config=config, verbose=True)
answer = runner.run("What is the square root of 144?")

# Inspect results
print(runner.trajectory)  # All steps
print(runner.metadata)  # Execution metadata
runner.print_trajectory()  # Pretty print
```

## File Structure

```
llm_agents/
├── __init__.py              # Public API exports
├── runner.py                # Orchestration (AgentConfig, Runner)
├── agentic/
│   ├── __init__.py          # Agentic module exports
│   ├── agent.py             # Agent base class and implementations
│   ├── llm_model.py         # LLM abstraction (OpenAI, Anthropic)
│   ├── tool.py              # Tool definition
│   └── builtin_tools.py     # Example tools
├── environment/
│   ├── __init__.py          # Environment module exports
│   ├── session.py           # Conversation state management
│   └── step.py              # Individual turn representation
└── prompt/
    ├── default_prompt.txt   # Simple agent prompt
    └── react_prompt.txt     # ReAct agent prompt
```

## Design Patterns

1. **Abstract Base Classes** - Clear interfaces (`Agent`, `LLMModel`)
2. **Factory Functions** - `build_model()`, `load_prompt()` for decoupling
3. **Dataclasses** - `Tool`, `ToolCallInfo`, `ModelResponse` for clarity
4. **Registry Pattern** - `AGENT_REGISTRY` maps config to agent classes
5. **Strategy Pattern** - Different agent implementations
6. **Provider Abstraction** - `LLMModel` abstracts OpenAI/Anthropic differences

## Model Selection

```python
def build_model(model_name: str) -> LLMModel:
    if model_name.startswith("claude"):
        return AnthropicModel(model_name=model_name)
    else:
        return OpenAIModel(model_name=model_name)
```

**Environment Variables:**
- OpenAI: `CHATGPT_API_KEY` or `OPENAI_API_KEY`
- Anthropic: `ANTHROPIC_API_KEY`

## LLM Call Logging

When `log_llm_calls=True`, all LLM calls are logged to:
```
results/llm_calls/{timestamp}_{session_id}/llm_call_{n}.json
```

Each log entry contains:
- Timestamp and call number
- Session ID and model name
- Raw input (messages, tools, parameters)
- Raw output (content, tool_calls)

Useful for debugging, analysis, and RL training data collection.
