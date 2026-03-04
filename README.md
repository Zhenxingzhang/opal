# opal

Open Platform for Agentic Learning — experimental code for LLM-based agentic design and reinforcement learning research.

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for Python package management.

### Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Verify installation (requires version 0.7+):

```bash
uv --version
```

### Install dependencies

```bash
uv sync
```

## Usage

Run the main script:

```bash
uv run python main.py
```

## Development

Install the package in editable mode:

```bash
uv pip install -e .
```
