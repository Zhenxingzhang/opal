"""Quick demo script for the agent environment."""

import argparse

from opal.runner import Runner, AgentConfig, RunnerConfig
from opal.environment.tool_environment import (
    CALCULATOR_TOOL,
    SEARCH_WEB_TOOL,
    READ_PDF_TOOL,
)


def main():
    parser = argparse.ArgumentParser(description="Run an LLM agent.")
    parser.add_argument("--config", type=str, help="Path to a YAML config file.")
    parser.add_argument("query", type=str, help="User query to send to the agent.")
    args = parser.parse_args()

    if args.config:
        from opal.config import load_config

        experiment = load_config(args.config)
        config = experiment.agent
        runner_config = experiment.runner
        print(f"Experiment: {experiment.name}")
        print(f"Runner: max_steps={runner_config.max_steps}, parallelism={runner_config.parallelism}")
        print(f"Semantic retrieval: top_k={experiment.semantic_retrieval.top_k}, model={experiment.semantic_retrieval.model_name}")
    else:
        config = AgentConfig(
            model_name="gpt-4o-2024-11-20",
            agent_name="react",
            log_llm_calls=True,
            tools=[CALCULATOR_TOOL, SEARCH_WEB_TOOL, READ_PDF_TOOL],
        )
        runner_config = RunnerConfig(max_steps=10)

    runner = Runner(config=config, runner_config=runner_config)
    answer = runner.run(args.query)

    print("\n\nFinal answer:", answer)
    runner.print_trajectory()


if __name__ == "__main__":
    main()
