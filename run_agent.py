"""Quick demo script for the agent environment."""

import argparse
import asyncio
from datetime import datetime
from pathlib import Path

from opal.session.session_runner import SessionRunner
from opal.config import AgentConfig, SessionConfig
from opal.environment.tool_environment import (
    CALCULATOR_TOOL,
    SEARCH_WEB_TOOL,
    READ_PDF_TOOL,
)


async def main():
    parser = argparse.ArgumentParser(description="Run an LLM agent.")
    parser.add_argument("--config", type=str, help="Path to a YAML config file.")
    parser.add_argument(
        "--query",
        type=str,
        help="User query to send to the agent.",
        default="what is 2+2?",
    )
    args = parser.parse_args()

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.config:
        from opal.config import load_config

        experiment = load_config(args.config)
        agent_config = experiment.agent_config
        session_config = experiment.session_config
        print(f"Experiment: {experiment.name}")
        print(
            f"Runner: max_steps={session_config.max_steps}, parallelism={experiment.parallelism}"
        )
        print(
            f"Semantic retrieval: top_k={experiment.sem_retrieval_config.top_k}, model={experiment.sem_retrieval_config.model_name}"
        )
    else:
        session_config = SessionConfig(
            max_steps=10,
            logging_dir_root=Path(f"results/default_{run_timestamp}"),
            enable_logging=True,
        )

        agent_config = AgentConfig(
            model_name="gpt-4o-2024-11-20",
            agent_name="default",
            tools=[CALCULATOR_TOOL, SEARCH_WEB_TOOL, READ_PDF_TOOL],
        )

    runner = SessionRunner(session_config=session_config, agent_config=agent_config)
    answer = await runner.run(args.query)

    print("\n\nFinal answer:", answer)
    runner.print_trajectory()


if __name__ == "__main__":
    asyncio.run(main())
