"""Async demo script for the agent environment."""

import argparse
import asyncio

from opal.runner import Runner, AgentConfig, RunnerConfig
from opal.environment.tool_environment import (
    CALCULATOR_TOOL,
    SEARCH_WEB_TOOL,
    READ_PDF_TOOL,
)


async def main():
    parser = argparse.ArgumentParser(description="Run an LLM agent (async).")
    parser.add_argument("query", type=str, help="User query to send to the agent.")
    args = parser.parse_args()

    config = AgentConfig(
        model_name="gpt-4o-2024-11-20",
        agent_name="react",
        log_llm_calls=True,
        tools=[CALCULATOR_TOOL, SEARCH_WEB_TOOL, READ_PDF_TOOL],
    )
    runner = Runner(config=config, runner_config=RunnerConfig(max_steps=10))
    answer = await runner.run_async(args.query)

    print("\n\nFinal answer:", answer)
    runner.print_trajectory()


if __name__ == "__main__":
    asyncio.run(main())
