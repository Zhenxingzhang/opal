import asyncio
from datetime import datetime
import json
import logging
import os
from pathlib import Path

import pandas as pd

from opal import AgentConfig, Runner
from opal.config import load_config
from opal.environment.tool_environment import SEARCH_PDF_TOOL
from opal.environment.tool_environment import ToolEnvironment
from utils import build_retriever, PATH_ROOT, PATH_FINANCE_BENCH

logging.basicConfig(level=logging.INFO)

##############################################################################
# DATASET CONFIG
##############################################################################
PATH_RESULTS = PATH_ROOT + "/results/"

PATH_DATASET_JSONL = PATH_FINANCE_BENCH + "/data/financebench_open_source.jsonl"
PATH_DOCUMENT_INFO_JSONL = (
    PATH_FINANCE_BENCH + "/data/financebench_document_information.jsonl"
)

# Choose DATASET PORTION:
# - ALL: Full Dataset
# - OPEN_SOURCE: Open Source Part (n=150)
# - CLOSED_SOURCE: Closed Source Part --> Request access at contact@patronus.ai
DATASET_PORTION = "OPEN_SOURCE"

# Experiment metadata
EVAL_MODE = "closedBook"

# Concurrency config
MAX_CONCURRENT_TASKS = 10


async def process_question(
    semaphore: asyncio.Semaphore,
    agent_config: AgentConfig,
    question_row: pd.Series,
    results_queue: asyncio.Queue,
    run_timestamp: str,
    retrieval_model_name: str = "all-MiniLM-L6-v2",
):
    """Process a single question with concurrency control."""
    async with semaphore:
        doc_name = question_row["doc_name"]
        # doc_name = "all"
        print(f"doc_name: {doc_name}")
        retriever = build_retriever(doc_name, model_name=retrieval_model_name)
        print(f"retriever: {retriever.summary()}")
        tool_env = ToolEnvironment(retriever=retriever)
        runner = Runner(config=agent_config, env=tool_env, run_timestamp=run_timestamp)
        question = question_row["question"]
        print(f"Processing: {question_row['financebench_id']} - {question[:50]}...")

        try:
            model_answer = await runner.run_async(question)
        except Exception as e:
            print(f"Error processing {question_row['financebench_id']}: {e}")
            model_answer = f"Error: {e}"

        question_result = {
            "financebench_id": question_row["financebench_id"],
            "model_name": agent_config.model_name,
            "system_prompt": runner.agent.system_prompt,
            "tools": [tool.name for tool in runner.agent.tools],
            "eval_mode": EVAL_MODE,
            "temp": agent_config.temperature,
            "question": question,
            "gold_answer": question_row["answer"],
            "model_answer": model_answer,
            "label": "",  # To be filled by evaluation
        }
        await results_queue.put(question_result)
        print(f"Completed: {question_row['financebench_id']}")


async def write_results(output_file: Path, results_queue: asyncio.Queue, total: int):
    """Write results to file as they complete."""
    written = 0
    with open(output_file, "w") as f:
        while written < total:
            result = await results_queue.get()
            f.write(json.dumps(result) + "\n")
            f.flush()
            written += 1
            print(f"Progress: {written}/{total}")


async def main(
    config_path: str,
    max_concurrent: int = MAX_CONCURRENT_TASKS,
    max_tasks: int | None = None,
):
    ##############################################################################
    # LOAD DATASET
    ##############################################################################

    # Load Full Dataset
    df_questions = pd.read_json(PATH_DATASET_JSONL, lines=True)
    # df_meta = pd.read_json(PATH_DOCUMENT_INFO_JSONL, lines=True)
    # df_full = pd.merge(df_questions, df_meta, on="doc_name")

    # Get all docs
    df_questions = df_questions.sort_values("doc_name")
    all_doc_names = df_questions["doc_name"].unique().tolist()
    print(f"Total number of distinct PDF: {len(all_doc_names)}")

    # Select relevant dataset portion
    if DATASET_PORTION != "ALL":
        df_questions = df_questions.loc[
            df_questions["dataset_subset_label"] == DATASET_PORTION
        ]
    print(f"Number of questions: {len(df_questions)}")

    # Limit number of tasks if specified
    if max_tasks is not None and max_tasks > 0:
        df_questions = df_questions.head(max_tasks)
        print(f"Limited to {max_tasks} tasks")

    # Check relevant documents
    df_questions = df_questions.sort_values("doc_name")
    selected_doc_names = df_questions["doc_name"].unique().tolist()
    print(f"Number of distinct PDF: {len(selected_doc_names)}")

    experiment = load_config(config_path)
    agent_config = experiment.agent
    max_concurrent = experiment.runner.parallelism
    retrieval_model_name = experiment.semantic_retrieval.model_name
    print(f"Experiment: {experiment.name}")

    # Shared timestamp for this run
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Output file path
    output_file = Path(
        PATH_RESULTS,
        f"{agent_config.agent_name}_{agent_config.get_system_prompt_name()}_{agent_config.model_name}_{EVAL_MODE}_{run_timestamp}.jsonl",
    )

    # Ensure results directory exists
    os.makedirs(output_file.parent, exist_ok=True)

    print(f"Running with max {max_concurrent} concurrent tasks")

    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(max_concurrent)
    results_queue = asyncio.Queue()

    # Create tasks for all questions
    question_tasks = [
        process_question(semaphore, agent_config, row, results_queue, run_timestamp, retrieval_model_name)
        for _, row in df_questions.iterrows()
    ]

    # Start writer task
    writer_task = asyncio.create_task(
        write_results(output_file, results_queue, len(question_tasks))
    )

    # Run all question tasks
    await asyncio.gather(*question_tasks)

    # Wait for writer to finish
    await writer_task

    print(f"\n\nCompleted. Results written to: {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run finance benchmark evaluation")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a YAML experiment config file",
    )
    parser.add_argument(
        "--concurrency",
        "-c",
        type=int,
        default=MAX_CONCURRENT_TASKS,
        help=f"Max concurrent tasks (default: {MAX_CONCURRENT_TASKS})",
    )
    parser.add_argument(
        "--max-tasks",
        "-n",
        type=int,
        default=1,
        help="Max total tasks to run (default: 1)",
    )
    args = parser.parse_args()

    asyncio.run(main(max_concurrent=args.concurrency, max_tasks=args.max_tasks, config_path=args.config))
