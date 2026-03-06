"""Opal - Open Platform for Agentic Learning."""

import dotenv

from opal.agentic.agent import (
    Agent,
    ReActAgent,
    DefaultAgent,
    AdvancedReActAgent,
)
from opal.agentic.llm_model import LLMModel, OpenAIModel, AnthropicModel
from opal.environment.tool_environment import ToolEnvironment
from opal.session.session_runner import SessionRunner
from opal.config import (
    AgentConfig,
    SessionConfig,
    ExperimentConfig,
    SemanticRetrievalConfig,
    load_config,
)

__all__ = [
    "Agent",
    "ReActAgent",
    "DefaultAgent",
    "AdvancedReActAgent",
    "LLMModel",
    "OpenAIModel",
    "AnthropicModel",
    "SessionRunner",
    "AgentConfig",
    "ExperimentConfig",
    "SessionConfig",
    "SemanticRetrievalConfig",
    "load_config",
    "ToolEnvironment",
]

dotenv.load_dotenv()
