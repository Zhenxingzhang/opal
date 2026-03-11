import asyncio
from datetime import datetime
import json
import logging
import os
import shutil
from pathlib import Path

import pandas as pd

from opal import AgentConfig, SessionRunner, SessionConfig
from opal.config import load_config
from opal.environment.tool_environment import ToolEnvironment
from utils import build_retriever, PATH_ROOT, PATH_FINANCE_BENCH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    concurrency_limiter: asyncio.Semaphore,
    agent_config: AgentConfig,
    session_config: SessionConfig,
    question_row: pd.Series,
    results_queue: asyncio.Queue,
    retrieval_model_name: str = "all-MiniLM-L6-v2",
):
    """Process a single question with concurrency control."""
    async with concurrency_limiter:
        doc_name = question_row["doc_name"]
        logger.debug("doc_name: %s", doc_name)
        retriever = build_retriever(doc_name, model_name=retrieval_model_name)
        logger.debug("retriever: %s", retriever.summary())
        tool_env = ToolEnvironment(retriever=retriever)
        session_runner = SessionRunner(
            session_config=session_config,
            agent_config=agent_config,
            env=tool_env,
        )
        question = question_row["question"]
        logger.info(
            "Processing: %s - %s...", question_row["financebench_id"], question[:50]
        )

        try:
            model_answer = await session_runner.run(question)
        except Exception as e:
            logger.error("Error processing %s: %s", question_row["financebench_id"], e)
            model_answer = f"Error: {e}"

        question_result = {
            "financebench_id": question_row["financebench_id"],
            "model_name": agent_config.model_name,
            "system_prompt": session_runner.agent.system_prompt,
            "tools": [tool.name for tool in session_runner.agent.tools],
            "eval_mode": EVAL_MODE,
            "temp": agent_config.temperature,
            "question": question,
            "gold_answer": question_row["answer"],
            "model_answer": model_answer,
            "label": "",  # To be filled by evaluation
            "steps": session_runner.metadata.get("steps", 0),
            "total_tool_calls": session_runner.metadata.get("total_tool_calls", 0),
        }
        await results_queue.put(question_result)
        logger.info("Completed: %s", question_row["financebench_id"])


async def write_results(
    output_path: Path, results_queue: asyncio.Queue, total_questions: int
):
    """Write results to file as they complete."""
    results_written = 0
    with open(output_path, "w") as results_file:
        while results_written < total_questions:
            result = await results_queue.get()
            results_file.write(json.dumps(result) + "\n")
            results_file.flush()
            results_written += 1
            logger.info("Progress: %d/%d", results_written, total_questions)


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
    logger.info("Total number of distinct PDF: %d", len(all_doc_names))

    # Select relevant dataset portion
    if DATASET_PORTION != "ALL":
        df_questions = df_questions.loc[
            df_questions["dataset_subset_label"] == DATASET_PORTION
        ]
    logger.info("Number of questions: %d", len(df_questions))

    # Limit number of tasks if specified
    if max_tasks is not None and max_tasks > 0:
        df_questions = df_questions.head(max_tasks)
        logger.info("Limited to %d tasks", max_tasks)

    # Check relevant documents
    df_questions = df_questions.sort_values("doc_name")
    selected_doc_names = df_questions["doc_name"].unique().tolist()
    logger.info("Number of distinct PDF: %d", len(selected_doc_names))

    experiment_config = load_config(config_path)
    agent_config = experiment_config.agent_config
    session_config = experiment_config.session_config
    max_concurrent = experiment_config.parallelism
    retrieval_model_name = experiment_config.sem_retrieval_config.model_name
    logger.info("Experiment: %s", experiment_config.name)

    # Shared timestamp for this run
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Output file path
    output_folder = Path(PATH_RESULTS, f"{experiment_config.name}_{run_timestamp}")
    output_path = Path(output_folder, "outputs.jsonl")
    session_config.logging_dir_root = output_folder

    # Ensure results directory exists
    os.makedirs(output_path.parent, exist_ok=True)

    # Copy config file into the output folder for reproducibility
    shutil.copy2(config_path, output_folder / "config.yaml")

    logger.info("Running with max %d concurrent tasks", max_concurrent)

    # Create semaphore for concurrency control
    concurrency_limiter = asyncio.Semaphore(max_concurrent)
    results_queue = asyncio.Queue()

    # Create tasks for all questions
    question_coroutines = [
        process_question(
            concurrency_limiter,
            agent_config,
            session_config,
            question_row,
            results_queue,
            retrieval_model_name,
        )
        for _, question_row in df_questions.iterrows()
    ]

    # Start writer task
    writer_task = asyncio.create_task(
        write_results(output_path, results_queue, len(question_coroutines))
    )

    # Run all question tasks
    await asyncio.gather(*question_coroutines)

    # Wait for writer to finish
    await writer_task

    logger.info("Completed. Results written to: %s", output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run finance benchmark evaluation")
    parser.add_argument(
        "config",
        type=str,
        nargs="?",
        default=None,
        help="Path to a YAML experiment config file",
    )
    parser.add_argument(
        "--config",
        type=str,
        dest="config_flag",
        default=None,
        help="Path to a YAML experiment config file (alternative to positional)",
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
        default=150,
        help="Max total tasks to run (default: 1)",
    )
    args = parser.parse_args()

    config_path = args.config or args.config_flag
    if not config_path:
        parser.error("config is required (positional or via --config)")

    asyncio.run(
        main(
            max_concurrent=args.concurrency,
            max_tasks=args.max_tasks,
            config_path=config_path,
        )
    )
