"""Async demo script for the agent environment."""

import asyncio

from opal.runner import Runner, AgentConfig
from opal.environment.tool_environment import (
    CALCULATOR_TOOL,
    SEARCH_WEB_TOOL,
    READ_PDF_TOOL,
)


async def main():
    config = AgentConfig(
        model_name="gpt-4o-2024-11-20",
        agent_name="react",
        max_steps=10,
        log_llm_calls=True,
        tools=[CALCULATOR_TOOL, SEARCH_WEB_TOOL, READ_PDF_TOOL],
    )
    runner = Runner(config=config)
    answer = await runner.run_async(
        "read the /Users/zzhang/Downloads/2601.18226.pdf file and tell me in one sentence what its about"
    )

    print("\n\nFinal answer:", answer)
    runner.print_trajectory()


if __name__ == "__main__":
    asyncio.run(main())
