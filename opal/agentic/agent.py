"""
Agent interface for agentic design.

The Agent class defines the interface for agents that act as policy objects.
Each agent provides hooks for building messages, recording tool cycles, and
pre-loop setup. The execution loop itself lives in the SessionRunner.
"""

import logging
from pathlib import Path

from opal.agentic.llm_model import (
    LLMModel,
    ModelResponse,
    OpenAIModel,
    AnthropicModel,
)
from opal.environment.tool import Tool
from opal.session.session import SessionState
from opal.environment.step import Step


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


def build_model(
    model_name: str, log_llm_calls: bool = False, temperature: float = 0.0
) -> LLMModel:
    """Build an LLM model based on the model name.

    Args:
        model_name: Name of the model (e.g., "gpt-4o", "claude-sonnet-4").
        log_llm_calls: Whether to enable LLM call logging.
        temperature: Sampling temperature for the model.

    Returns:
        An LLMModel instance.
    """
    if model_name.startswith("claude"):
        model = AnthropicModel(model_name=model_name, temperature=temperature)
    else:
        model = OpenAIModel(model_name=model_name, temperature=temperature)

    if log_llm_calls:
        model.log_llm_calls = True

    return model


class Agent:
    """Base class for agents (policy objects).

    Agents are pure policy objects — they decide what to say/do given the
    current session state but never execute tools or own the loop.  The
    SessionRunner owns the execution loop and mediates between Agent and
    ToolEnvironment.

    Subclasses customise behaviour by overriding:
    - ``build_messages``   — how the message list is constructed
    - ``record_tool_cycle`` — how a tool call + observation is recorded
    """

    system_prompt_name = ""
    system_prompt: str = ""
    model: LLMModel
    tools: list[Tool] = []
    verbose: bool = True

    def _init_common(self, config) -> None:
        """Shared initialization for all agent subclasses."""
        self.system_prompt_name = config.get_system_prompt_name()
        self.system_prompt = load_prompt(self.system_prompt_name)
        self.model = build_model(
            config.model_name,
            getattr(config, "log_llm_calls", False),
            temperature=getattr(config, "temperature", 0.0),
        )
        self.verbose = getattr(config, "verbose", True)

    # ------------------------------------------------------------------
    # Policy hooks (overridable)
    # ------------------------------------------------------------------

    def build_messages(self, session: SessionState) -> list[dict]:
        """Build the message list for the LLM API call."""
        return session.build_messages(self.system_prompt)

    def record_tool_cycle(
        self,
        response: ModelResponse,
        observation: str,
        session: SessionState,
        step_idx: int,
    ) -> None:
        """Record a tool call and its observation in the session trajectory."""
        tc = response.tool_calls[0]
        tool_call_record = {
            "id": tc.id,
            "name": tc.name,
            "arguments": tc.arguments,
        }
        session.add_step(
            Step(
                role="assistant",
                content=response.content,
                tool_call=tool_call_record,
            )
        )
        session.add_step(
            Step(
                role="tool",
                tool_call=tool_call_record,
                tool_result=observation,
            )
        )

        if self.verbose:
            print(f"[Step {step_idx + 1}] Tool: {tc.name}")
            print(f"  Args: {tc.arguments}")
            print(
                f"  Obs:  {observation[:200]}{'...' if len(observation) > 200 else ''}"
            )

    # ------------------------------------------------------------------
    # LLM interaction (not overridden by subclasses)
    # ------------------------------------------------------------------

    def act(
        self, messages: list[dict], call_number: int | None = None
    ) -> ModelResponse:
        """Call the LLM with the message history and return the response."""
        return self.model.call(
            messages, self.tools if self.tools else None, call_number
        )

    async def act_async(
        self, messages: list[dict], call_number: int | None = None
    ) -> ModelResponse:
        """Async version of act()."""
        return await self.model.call_async(
            messages, self.tools if self.tools else None, call_number
        )



class DefaultAgent(Agent):
    """A simple agent that uses the default prompt.

    Makes a single LLM call with no tools.  All default hooks work as-is.
    """

    def __init__(self, config, **kwargs):
        self._init_common(config)
        self.tools = []


class ReActAgent(Agent):
    """A ReAct-style agent that uses reasoning and acting.

    Loop (driven by SessionRunner):
    1. Call LLM with message history
    2. If tool call -> execute tool, add observation, continue
    3. If plain text -> return as final answer
    4. Repeat until done or max_steps

    All default hooks (build_messages, record_tool_cycle, finish,
    max_steps_exceeded) match exactly, so no overrides needed.
    """

    def __init__(self, config, **kwargs):
        self._init_common(config)
        self.tools = config.tools or []


class AdvancedReActAgent(Agent):
    """A ReAct-style agent with explicit Thought/Action/Observation separation.

    Each cycle's response is split into three distinct trajectory steps:

        Step(role="assistant", content="I need to...")           # Thought
        Step(role="assistant", content=None, tool_call={...})    # Action
        Step(role="tool", tool_call={...}, tool_result="...")     # Observation

    This gives a cleaner signal for downstream RL analysis compared to
    ReActAgent which merges Thought+Action into a single step.

    Overrides ``build_messages`` to merge consecutive Thought+Action pairs
    back into one assistant message (APIs reject consecutive assistant
    messages), and ``record_tool_cycle`` for the 3-step split.
    """

    def __init__(self, config, **kwargs):
        self._init_common(config)
        self.tools = config.tools or []

    def build_messages(self, session: SessionState) -> list[dict]:
        """Build LLM messages, merging consecutive Thought+Action step pairs.

        The trajectory stores Thought and Action as two separate ``assistant``
        steps, but the LLM API rejects consecutive ``assistant`` messages.
        This method detects the pattern and merges them into a single
        ``assistant`` message with both ``content`` and ``tool_calls``.
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

    def record_tool_cycle(
        self,
        response: ModelResponse,
        observation: str,
        session: SessionState,
        step_idx: int,
    ) -> None:
        """Split one LLM response into three separate trajectory steps."""
        tc = response.tool_calls[0]
        tool_call_record = {
            "id": tc.id,
            "name": tc.name,
            "arguments": tc.arguments,
        }

        # Step 1 — Thought (pure reasoning, no tool_call)
        thought_text = response.content or ""
        session.add_step(Step(role="assistant", content=thought_text))

        # Step 2 — Action (pure tool call, no content)
        session.add_step(
            Step(role="assistant", content=None, tool_call=tool_call_record)
        )

        # Step 3 — Observation (tool result)
        session.add_step(
            Step(role="tool", tool_call=tool_call_record, tool_result=observation)
        )

        if self.verbose:
            print(f"--- Cycle {step_idx + 1} ---")
            print(
                f"[Thought] {thought_text[:300]}{'...' if len(thought_text) > 300 else ''}"
            )
            print(f"[Action]  {tc.name}({tc.arguments})")
            print(
                f"[Observation] {observation[:200]}{'...' if len(observation) > 200 else ''}"
            )


