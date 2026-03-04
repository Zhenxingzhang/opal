# opal

Open Platform for Agentic Learning — experimental code for LLM-based agentic design and reinforcement learning research.

## Architecture

Three-layer design:

- **Agentic** (`opal/agentic/`) — LLM model abstractions (OpenAI, Anthropic), agent implementations (DefaultAgent, ReActAgent), and tool definitions
- **Environment** (`opal/environment/`) — Session state, step/trajectory tracking, tool environments
- **Orchestration** (`opal/runner.py`) — Runner coordinates agent execution via config-driven `AgentConfig`

Supporting modules:

- **Embedding** (`opal/embedding/`) — Semantic retrieval with sentence-transformers and caching
- **Config** (`opal/config.py`) — YAML-based experiment configuration

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for Python package management.

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Verify (requires v0.7+)
uv --version

# Install dependencies
uv sync
```

## Environment Variables

Set in a `.env` file at the project root:

- `CHATGPT_API_KEY` or `OPENAI_API_KEY` — OpenAI access
- `ANTHROPIC_API_KEY` — Anthropic access

## Usage

```bash
# Run demo agent
uv run python run_agent.py

# Run async demo
uv run python run_agent_async.py

# Run FinanceBench benchmark (see finance_bench/README.md)
uv run python finance_bench/run.py --config finance_bench/configs/default.yaml -c 10 -n 5

# Evaluate benchmark results
uv run python finance_bench/eval.py results/<file>.jsonl --model gpt-4o-2024-11-20
```

## Development

```bash
# Lint and format
uv run ruff check .
uv run ruff format .
```
