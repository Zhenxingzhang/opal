"""Configuration dataclasses and YAML config file support.

Config classes:
- SessionConfig: execution settings (max_steps, logging)
- AgentConfig: agent policy configuration (model, agent type, tools, etc.)
- SemanticRetrievalConfig: retrieval settings (top_k, ranking model)
- ExperimentConfig: top-level experiment configuration combining the above
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yaml

from opal.agentic.agent import (
    DefaultAgent,
    ReActAgent,
)
from opal.environment.tool import Tool
from opal.environment.tool_environment import (
    CALCULATOR_TOOL,
    SEARCH_PDF_TOOL,
    READ_PDF_TOOL,
    SEARCH_WEB_TOOL,
    THINK_TOOL,
)


# Maps agent_name to (AgentClass, prompt_file_name)
AGENT_REGISTRY = {
    "default": (DefaultAgent, "default_prompt"),
    "react": (ReActAgent, "react_prompt"),
}

TOOL_REGISTRY: dict[str, Tool] = {
    "calculator": CALCULATOR_TOOL,
    "search_pdf": SEARCH_PDF_TOOL,
    "read_pdf": READ_PDF_TOOL,
    "search_web": SEARCH_WEB_TOOL,
    "think": THINK_TOOL,
}


@dataclass
class SessionConfig:
    """Configuration for the session execution loop."""

    max_steps: int = 10
    logging_dir_root: Path = field(
        default_factory=lambda: Path(
            f"results/demo_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    )
    enable_search: bool = False
    enable_logging: bool = False


@dataclass
class AgentConfig:
    """Configuration for Agent policy."""

    model_name: str = "gpt-4o-2024-11-20"
    agent_name: str = "react"  # "react" or "default" - determines agent type and prompt
    temperature: float = 0.0
    tools: list[Tool] = field(default_factory=list)
    system_prompt_name: str | None = None  # Optional custom prompt name
    verbose: bool = True
    retriever_top_k: int = 5  # Number of chunks for RAGAgent retrieval

    def get_system_prompt_name(self) -> str:
        """Get the prompt name for this agent.

        If system_prompt_name is set, use it. Otherwise, use the default for the agent type.
        """
        if self.system_prompt_name:
            return self.system_prompt_name
        if self.agent_name in AGENT_REGISTRY:
            return AGENT_REGISTRY[self.agent_name][1]
        return "default_prompt"


@dataclass
class SemanticRetrievalConfig:
    """Configuration for the semantic retrieval system."""

    top_k: int = 5
    model_name: str = "all-MiniLM-L6-v2"
    reranker_model_name: str | None = None
    chunk_size: int = 1024
    chunk_overlap: int = 30


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration.

    Sections:
        name: Human-readable experiment name for tracking.
        agent: Agent configuration (model, agent type, tools, etc.).
        runner: Execution settings (max_steps).
        semantic_retrieval: Retrieval settings (top_k, ranking model).
    """

    name: str = "default"
    parallelism: int = 1
    agent_config: AgentConfig = field(default_factory=AgentConfig)
    session_config: SessionConfig = field(default_factory=SessionConfig)
    sem_retrieval_config: SemanticRetrievalConfig = field(
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
        parallelism: 10

        session_runner:
          max_steps: 10
          enable_logging: true

        agent:
          model_name: gpt-4o-2024-11-20
          agent_name: react
          temperature: 0.0
          system_prompt_name: react_prompt
          tools:
            - calculator
            - search_web

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
        raise ValueError(
            f"Expected a YAML mapping in {path}, got {type(data).__name__}"
        )

    # Parse experiment name
    name = data.get("name", "default")

    # Parse session_runner section — support legacy 'max_turns' and 'runner' keys
    runner_data = data.get("session_runner", data.get("runner", {}))
    if "max_turns" in runner_data and "max_steps" not in runner_data:
        runner_data["max_steps"] = runner_data.pop("max_turns")
    parallelism = data.get("parallelism", runner_data.pop("parallelism", 1))

    # Backward compat: accept log_llm_calls from agent section as enable_logging
    agent_data = data.get("agent", {})
    if "enable_logging" not in runner_data and agent_data.get("log_llm_calls"):
        runner_data["enable_logging"] = agent_data.pop("log_llm_calls")
    else:
        agent_data.pop("log_llm_calls", None)

    session_config = SessionConfig(**runner_data)

    # Default logging_dir to <experiment_name>/<timestamp> if not explicitly set
    if "logging_dir" not in runner_data:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_config.logging_dir_root = Path(f"results/{name}_{timestamp}")

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
        parallelism=parallelism,
        agent_config=agent_config,
        session_config=session_config,
        sem_retrieval_config=retrieval_config,
    )
