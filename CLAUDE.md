# CLAUDE.md

## Project Overview

LLM agents framework (Opal) for agent reinforcement learning research. Provides a modular system for building LLM-based agents with tool use and multi-step reasoning, with an initial focus on financial domain benchmarking (FinanceBench).

### Architecture

Three-layer design:
- **Agentic** (`opal/agentic/`): LLM model abstractions (OpenAI, Anthropic), agent implementations (DefaultAgent, ReActAgent), tool definitions
- **Environment** (`opal/environment/`): Session state, step/trajectory tracking
- **Orchestration** (`opal/session/`): SessionRunner coordinates agent execution; SessionLogger handles file I/O

Key entry points:
- `run_agent.py` / `run_agent_async.py` — async demo scripts
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
- Async-only API — all LLM calls, agent actions, and runner methods are async
- No test suite yet — do not assume tests exist

## Environment Variables

- `CHATGPT_API_KEY` or `OPENAI_API_KEY` — OpenAI access
- `ANTHROPIC_API_KEY` — Anthropic access
- Loaded via `python-dotenv` (`.env` file)

## Key Patterns

- `AGENT_REGISTRY` maps agent names (`"default"`, `"react"`) to implementations and prompts
- Prompts live in `opal/prompt/` as `.txt` files
- LLM call logs go to `results/<run>/llm_calls/` when `enable_logging=True` in `SessionConfig`
- Evaluation cache in `.eval_cache/`


For FinanceBench analysis tasks, use the skill scripts in .claude/skills/ first. Results may be in JSONL format with varying schemas — always inspect the file structure before running analysis.
