# Agentic Design - Opal

## Architecture Overview

The Opal codebase implements a **modular agentic framework** with three main layers:

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

### 1. Agent (`opal/agentic/agent.py`)

**Base Class: `Agent` (Abstract)**
- Defines the interface for all agents
- Key methods:
  - `act(messages, call_number)` - Async: call the LLM with message history, returns `(ModelResponse, LLMCallMetrics)`
  - `build_messages(session)` - Build LLM message list from session trajectory
  - `record_tool_cycle(response, observation, session, step_idx)` - Record tool call + observation in trajectory

**Implementations:**

| Agent | Description | Use Case |
|-------|-------------|----------|
| `DefaultAgent` | Single LLM call, no tools | Simple completions, summarization |
| `ReActAgent` | Reasoning + Acting loop with tools | Complex tasks requiring multiple steps |

### 2. LLM Model (`opal/agentic/llm_model.py`)

**Base Class: `LLMModel` (Abstract)**
- Standardizes interface across providers
- Core method: `async call(messages, tools, call_number) -> (ModelResponse, LLMCallMetrics)`
- Uses async-only clients (no sync duplicates)

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

@dataclass(frozen=True)
class LLMCallMetrics:
    call_number: int
    timestamp: str
    model_name: str
    ai_cache_hit: bool
    raw_request: dict
    raw_response: dict
```

### 3. Tool (`opal/agentic/tool.py`)

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

### 4. Session & Step (`opal/environment/`)

**Session** - Manages conversation state:
- `id` - Unique session identifier (UUID)
- `trajectory` - List of Steps
- `metadata` - Execution context (timestamp, status, etc.)
- `llm_calls` - List of `LLMCallMetrics` from each LLM call

**Step** - Represents one conversational turn:
- `role` - "user", "assistant", or "tool"
- `content` - Text message content
- `tool_call` - Dict with {id, name, arguments}
- `tool_result` - Tool execution output

### 5. SessionRunner (`opal/session/session_runner.py`)

Orchestrates agent execution:
- Creates and manages SessionState
- Builds appropriate Agent based on config
- Owns the canonical async execution loop
- Provides access to results and trajectory

**SessionConfig:**
```python
@dataclass
class SessionConfig:
    max_steps: int = 10
    logging_dir_root: Path = ...
    enable_search: bool = False
    enable_logging: bool = False
```

**AgentConfig:**
```python
@dataclass
class AgentConfig:
    model_name: str = "gpt-4o-2024-11-20"
    agent_name: str = "react"  # "react" or "default"
    temperature: float = 0.0
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
│   2. Call LLM via await act()       │
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
ToolEnvironment.execute_tool(name, arguments_json)
    │
    ├─► Look up tool by name in tool_map
    ├─► Parse arguments_json to dict
    ├─► Call tool.function(**args)
    ├─► Convert result to string
    │
    ▼
Return result as observation
```

## Usage Example

```python
import asyncio
from opal.session.session_runner import SessionRunner
from opal.config import AgentConfig, SessionConfig
from opal.environment.tool_environment import CALCULATOR_TOOL, SEARCH_WEB_TOOL

# Configure
session_config = SessionConfig(
    max_steps=10,
    enable_logging=True,
)
agent_config = AgentConfig(
    model_name="gpt-4o",
    agent_name="react",
    tools=[CALCULATOR_TOOL, SEARCH_WEB_TOOL],
)

# Run (async)
async def main():
    runner = SessionRunner(session_config=session_config, agent_config=agent_config)
    answer = await runner.run("What is the square root of 144?")

    # Inspect results
    print(runner.trajectory)  # All steps
    print(runner.metadata)  # Execution metadata
    runner.print_trajectory()  # Pretty print

asyncio.run(main())
```

## File Structure

```
opal/
├── __init__.py              # Public API exports
├── config.py                # AgentConfig, SessionConfig, ExperimentConfig, load_config
├── agentic/
│   ├── __init__.py          # Agentic module exports
│   ├── agent.py             # Agent base class and implementations
│   ├── llm_model.py         # LLM abstraction (OpenAI, Anthropic), LLMCallMetrics
│   └── tool.py              # Tool definition
├── environment/
│   ├── __init__.py          # Environment module exports
│   ├── session.py           # Conversation state management (SessionState)
│   ├── step.py              # Individual turn representation
│   └── tool_environment.py  # Tool execution and built-in tools
├── session/
│   ├── __init__.py          # Session module exports
│   ├── session.py           # SessionState dataclass
│   ├── session_runner.py    # SessionRunner (orchestration)
│   └── logger.py            # SessionLogger (file I/O for logs)
└── prompt/
    ├── default_prompt.txt   # Simple agent prompt
    └── react_prompt.txt     # ReAct agent prompt
```

## Design Patterns

1. **Abstract Base Classes** - Clear interfaces (`Agent`, `LLMModel`)
2. **Factory Functions** - `build_model()`, `load_prompt()` for decoupling
3. **Dataclasses** - `Tool`, `ToolCallInfo`, `ModelResponse`, `LLMCallMetrics` (frozen) for clarity
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

When `enable_logging=True` in `SessionConfig`, `SessionLogger` writes all data to disk after the run completes:
```
results/<run_name>/<session_id[:8]>/
├── llm_calls/
│   ├── llm_call_1.json
│   ├── llm_call_2.json
│   └── ...
└── <session_id[:8]>_trajectory.json
```

Each LLM call log (`LLMCallMetrics`) contains:
- Call number and timestamp
- Model name
- Raw request (messages, tools, parameters)
- Raw response (content, tool_calls)

Metrics are captured as frozen dataclasses during execution and flushed to disk at the end of the run. Useful for debugging, analysis, and RL training data collection.
