"""Quick demo script for the agent environment."""

import argparse

from opal.runner import Runner, AgentConfig
from opal.environment.tool_environment import (
    CALCULATOR_TOOL,
    SEARCH_WEB_TOOL,
    READ_PDF_TOOL,
)


def main():
    parser = argparse.ArgumentParser(description="Run an LLM agent.")
    parser.add_argument("--config", type=str, help="Path to a YAML config file.")
    parser.add_argument("--query", type=str, help="User query to send to the agent.")
    args = parser.parse_args()

    if args.config:
        from opal.config import load_config

        experiment = load_config(args.config)
        config = experiment.agent
        print(f"Experiment: {experiment.name}")
        print(f"Runner: max_steps={experiment.runner.max_steps}, parallelism={experiment.runner.parallelism}")
        print(f"Semantic retrieval: top_k={experiment.semantic_retrieval.top_k}, model={experiment.semantic_retrieval.model_name}")
    else:
        config = AgentConfig(
            model_name="gpt-4o-2024-11-20",
            agent_name="react",
            max_steps=10,
            log_llm_calls=True,
            tools=[CALCULATOR_TOOL, SEARCH_WEB_TOOL, READ_PDF_TOOL],
        )

    query = args.query or "read the /Users/zzhang/Downloads/2601.18226.pdf file and tell me in one sentence what its about"

    runner = Runner(config=config)
    answer = runner.run(query)

    print("\n\nFinal answer:", answer)
    runner.print_trajectory()


if __name__ == "__main__":
    main()
