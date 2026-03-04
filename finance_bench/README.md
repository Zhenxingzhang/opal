# FinanceBench

Benchmark suite for evaluating LLM agents on financial question answering using the [FinanceBench](https://huggingface.co/datasets/PatronusAI/financebench) dataset (150 open-source questions over 368 SEC filings).

## Quick Start

```bash
# Run benchmark (processes questions concurrently)
uv run python finance_bench/run.py --config finance_bench/configs/default.yaml -c 10 -n 5

# Evaluate results with a judge model
uv run python finance_bench/eval.py results/<file>.jsonl --model gpt-4o-2024-11-20

# Hybrid evaluation (correct if ANY judge agrees)
uv run python finance_bench/eval.py results/<file>.jsonl --hybrid gpt-4o-2024-11-20 o1-mini o3-mini
```

## Structure

```
finance_bench/
├── run.py          # Async benchmark runner
├── eval.py         # LLM-as-judge evaluator (3-way: correct/incorrect/no_answer)
├── utils.py        # PDF parsing and semantic retrieval helpers
├── configs/        # Experiment configs (agent, model, runner settings)
├── data/           # Dataset JSONL files (questions + document metadata)
└── pdfs/           # SEC filing PDFs (10-K, 10-Q, 8-K, earnings)
```

## How It Works

**Running:** For each question, the runner builds a semantic retriever over the relevant PDF, binds it as a `search_pdf` tool, and runs the configured agent (default/react) to produce an answer. Results stream to a timestamped JSONL file.

**Evaluation:** A judge LLM compares model answers against gold answers with strict numerical matching (exact values, no rounding tolerance). Verdicts are cached in `.eval_cache/` to avoid redundant API calls.

## Configuration

Experiments are configured via YAML (see `configs/default.yaml`):

```yaml
agent:
  model_name: gpt-4o-2024-11-20
  agent_name: default        # "default" or "react"
  log_llm_calls: true

runner:
  max_steps: 10
```
