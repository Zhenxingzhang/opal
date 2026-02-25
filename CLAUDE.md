# CLAUDE.md

## Project Overview

LLM agents framework for agent reinforcement learning research. Provides a modular system for building LLM-based agents with tool use and multi-step reasoning, with an initial focus on financial domain benchmarking (FinanceBench).

### Architecture

Three-layer design:
- **Agentic** (`llm_agents/agentic/`): LLM model abstractions (OpenAI, Anthropic), agent implementations (DefaultAgent, ReActAgent), tool definitions
- **Environment** (`llm_agents/environment/`): Session state, step/trajectory tracking
- **Orchestration** (`llm_agents/runner.py`): Runner coordinates agent execution via AgentConfig

Key entry points:
- `run_agent.py` / `run_agent_async.py` — demo scripts
- `finance_bench/run.py` — async benchmark runner
- `finance_bench/eval.py` — evaluation with response caching

## Commands

Package manager: `uv` (v0.7+). Always use `uv run` to execute scripts.

```bash
# Install/sync dependencies
uv sync

# Run demo agent
uv run python run_agent.py

# Run finance benchmark (concurrent)
uv run python finance_bench/run.py --concurrency 10 --max-tasks 1

# Run evaluation
uv run python finance_bench/eval.py

# Lint and format
uv run ruff check .
uv run ruff format .
```

## Code Style

- Python 3.12+ — use modern type hints (`str | None`, `list[Tool]`)
- Ruff for linting and formatting
- Dataclasses for data containers
- Both sync and async methods where applicable (`run()` / `run_async()`)
- No test suite yet — do not assume tests exist

## Environment Variables

- `CHATGPT_API_KEY` or `OPENAI_API_KEY` — OpenAI access
- `ANTHROPIC_API_KEY` — Anthropic access
- Loaded via `python-dotenv` (`.env` file)

## Key Patterns

- `AGENT_REGISTRY` maps agent names (`"default"`, `"react"`) to implementations and prompts
- Prompts live in `llm_agents/prompt/` as `.txt` files
- LLM call logs go to `results/llm_calls/` when `log_llm_calls=True`
- Evaluation cache in `.eval_cache/`
