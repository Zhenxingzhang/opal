"""YAML config file support for experiment configuration.

The config is organized into sections:
- name: experiment name
- agent: agent configuration (model, agent type, tools, etc.)
- runner: execution settings (max_steps, parallelism)
- semantic_retrieval: retrieval settings (top_k, ranking model)
"""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from opal.environment.tool import Tool
from opal.environment.tool_environment import (
    CALCULATOR_TOOL,
    SEARCH_PDF_TOOL,
    READ_PDF_TOOL,
    SEARCH_WEB_TOOL,
)
from opal.runner import AgentConfig, RunnerConfig

TOOL_REGISTRY: dict[str, Tool] = {
    "calculator": CALCULATOR_TOOL,
    "search_pdf": SEARCH_PDF_TOOL,
    "read_pdf": READ_PDF_TOOL,
    "search_web": SEARCH_WEB_TOOL,
}


@dataclass
class SemanticRetrievalConfig:
    """Configuration for the semantic retrieval system."""

    top_k: int = 5
    model_name: str = "all-MiniLM-L6-v2"


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration.

    Sections:
        name: Human-readable experiment name for tracking.
        agent: Agent configuration (model, agent type, tools, etc.).
        runner: Execution settings (max_steps, parallelism).
        semantic_retrieval: Retrieval settings (top_k, ranking model).
    """

    name: str = "default"
    agent: AgentConfig = field(default_factory=AgentConfig)
    runner: RunnerConfig = field(default_factory=RunnerConfig)
    semantic_retrieval: SemanticRetrievalConfig = field(
        default_factory=SemanticRetrievalConfig
    )


def _resolve_tools(tool_names: list[str], source: str | Path) -> list[Tool]:
    """Resolve tool name strings to Tool objects via TOOL_REGISTRY."""
    tools: list[Tool] = []
    for name in tool_names:
        if name not in TOOL_REGISTRY:
            available = ", ".join(sorted(TOOL_REGISTRY))
            raise ValueError(
                f"Unknown tool '{name}' in {source}. Available tools: {available}"
            )
        tools.append(TOOL_REGISTRY[name])
    return tools


def load_config(path: str | Path) -> ExperimentConfig:
    """Load an ExperimentConfig from a YAML file.

    Expected YAML structure::

        name: my-experiment

        agent:
          model_name: gpt-4o-2024-11-20
          agent_name: react
          temperature: 0.0
          system_prompt_name: react_prompt
          log_llm_calls: true
          tools:
            - calculator
            - search_web

        runner:
          max_steps: 10
          parallelism: 10

        semantic_retrieval:
          top_k: 5
          model_name: all-MiniLM-L6-v2

    Args:
        path: Path to the YAML config file.

    Returns:
        A populated ExperimentConfig instance.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        ValueError: If the config contains unknown tool names.
    """
    path = Path(path)
    with open(path) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path}, got {type(data).__name__}")

    # Parse experiment name
    name = data.get("name", "default")

    # Parse runner section — support legacy 'max_turns' key
    runner_data = data.get("runner", {})
    if "max_turns" in runner_data and "max_steps" not in runner_data:
        runner_data["max_steps"] = runner_data.pop("max_turns")
    runner_config = RunnerConfig(**runner_data)

    # Parse semantic_retrieval section (before agent, since agent bridges top_k)
    retrieval_data = data.get("semantic_retrieval", {})
    retrieval_config = SemanticRetrievalConfig(**retrieval_data)

    # Parse agent section
    agent_data = data.get("agent", {})
    tool_names = agent_data.pop("tools", [])
    tools = _resolve_tools(tool_names, path)
    # Bridge semantic_retrieval.top_k into AgentConfig.retriever_top_k
    agent_data.setdefault("retriever_top_k", retrieval_config.top_k)
    agent_config = AgentConfig(tools=tools, **agent_data)

    return ExperimentConfig(
        name=name,
        agent=agent_config,
        runner=runner_config,
        semantic_retrieval=retrieval_config,
    )
