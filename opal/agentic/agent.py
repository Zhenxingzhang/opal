"""
Agent interface for agentic design.

The Agent class defines the interface for agents that act as policy objects.
Each agent provides hooks for building messages and pre-loop setup.
The execution loop and tool cycle recording live in the SessionRunner.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from opal.config import AgentConfig

from opal.agentic.llm_model import (
    LLMModel,
    LLMCallMetrics,
    ModelResponse,
    OpenAIModel,
    AnthropicModel,
)
from opal.environment.tool import Tool
from opal.session.session import SessionState


PROMPT_DIR = Path(__file__).parent.parent / "prompt"

logger = logging.getLogger(__name__)


def load_prompt(prompt_name: str) -> str:
    """Load a prompt from the prompt directory.

    Args:
        prompt_name: Name of the prompt file (with or without .txt extension).

    Returns:
        The prompt content as a string.
    """
    if not prompt_name.endswith(".txt"):
        prompt_name = f"{prompt_name}.txt"

    prompt_path = PROMPT_DIR / prompt_name
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    return prompt_path.read_text().strip()


def build_model(model_name: str, temperature: float = 0.0) -> LLMModel:
    """Build an LLM model based on the model name.

    Args:
        model_name: Name of the model (e.g., "gpt-4o", "claude-sonnet-4").
        temperature: Sampling temperature for the model.

    Returns:
        An LLMModel instance.
    """
    if model_name.startswith("claude"):
        return AnthropicModel(model_name=model_name, temperature=temperature)
    else:
        return OpenAIModel(model_name=model_name, temperature=temperature)


class Agent:
    """Base class for agents (policy objects).

    Agents are pure policy objects — they decide what to say/do given the
    current session state but never execute tools or own the loop.  The
    SessionRunner owns the execution loop and mediates between Agent and
    ToolEnvironment.

    Subclasses customise behaviour by overriding:
    - ``build_messages`` — how the message list is constructed
    """

    system_prompt_name = ""
    system_prompt: str = ""
    model: LLMModel
    tools: list[Tool] = []

    def _init_common(self, config: AgentConfig) -> None:
        """Shared initialization for all agent subclasses."""
        self.system_prompt_name = config.get_system_prompt_name()
        self.system_prompt = load_prompt(self.system_prompt_name)
        self.model = build_model(
            config.model_name,
            temperature=getattr(config, "temperature", 0.0),
        )

    # ------------------------------------------------------------------
    # Policy hooks (overridable)
    # ------------------------------------------------------------------

    def build_messages(self, session: SessionState) -> list[dict]:
        """Build the message list for the LLM API call.

        The trajectory stores Thought and Action as separate ``assistant``
        steps, but the LLM API rejects consecutive ``assistant`` messages.
        This method detects Thought+Action pairs and merges them into a
        single ``assistant`` message with both ``content`` and ``tool_calls``.
        """
        messages: list[dict] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        steps = session.trajectory
        i = 0
        while i < len(steps):
            step = steps[i]

            # Detect Thought+Action pair: assistant-with-content followed by
            # assistant-with-tool_call (no content).
            if (
                i + 1 < len(steps)
                and step.role == "assistant"
                and step.content
                and step.tool_call is None
                and steps[i + 1].role == "assistant"
                and steps[i + 1].tool_call is not None
                and steps[i + 1].content is None
            ):
                # Merge into a single assistant message
                tc = steps[i + 1].tool_call
                messages.append(
                    {
                        "role": "assistant",
                        "content": step.content,
                        "tool_calls": [
                            {
                                "id": tc["id"],
                                "type": "function",
                                "function": {
                                    "name": tc["name"],
                                    "arguments": tc["arguments"],
                                },
                            }
                        ],
                    }
                )
                i += 2  # skip both steps
                continue

            # Default: delegate to Step.to_message()
            messages.append(step.to_message())
            i += 1

        return messages

    # ------------------------------------------------------------------
    # LLM interaction (not overridden by subclasses)
    # ------------------------------------------------------------------

    async def act(
        self, messages: list[dict], call_number: int = 0
    ) -> tuple[ModelResponse, LLMCallMetrics]:
        """Call the LLM with the message history and return the response."""
        return await self.model.call(
            messages, self.tools if self.tools else None, call_number
        )


class DefaultAgent(Agent):
    """General-purpose agent. Works with or without tools.

    With no tools configured, makes a single LLM call and returns the response.
    With tools, the SessionRunner drives a loop using ``build_messages``.
    """

    def __init__(self, config):
        self._init_common(config)
        self.tools = config.tools or []


class ReActAgent(Agent):
    """ReAct agent with explicit Thought/Action/Observation separation.

    Uses a ReAct-style prompt that encourages the LLM to emit explicit
    reasoning before each tool call. The SessionRunner records each cycle
    as separate Thought/Action/Observation trajectory steps for cleaner
    downstream RL analysis.
    """

    def __init__(self, config):
        self._init_common(config)
        self.tools = config.tools or []
