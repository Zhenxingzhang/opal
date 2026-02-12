"""
Agent interface for agentic design.

The Agent class defines the interface for agents that can process queries
using tools and LLM models. Each agent has its own execution loop.
"""

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path

from llm_agents.agentic.llm_model import (
    LLMModel,
    ModelResponse,
    OpenAIModel,
    AnthropicModel,
)
from llm_agents.agentic.tool import Tool
from llm_agents.environment.session import Session
from llm_agents.environment.step import Step


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


def build_model(model_name: str, log_llm_calls: bool = False) -> LLMModel:
    """Build an LLM model based on the model name.

    Args:
        model_name: Name of the model (e.g., "gpt-4o", "claude-sonnet-4").
        log_llm_calls: Whether to enable LLM call logging.

    Returns:
        An LLMModel instance.
    """
    if model_name.startswith("claude"):
        model = AnthropicModel(model_name=model_name)
    else:
        model = OpenAIModel(model_name=model_name)

    if log_llm_calls:
        model.log_llm_calls = True

    return model


class Agent(ABC):
    """Abstract base class for agents."""

    system_prompt_name = ""
    system_prompt: str = ""
    model: LLMModel
    tools: list[Tool] = []
    max_steps: int = 10
    verbose: bool = True
    _tool_map: dict[str, Tool] = {}

    @abstractmethod
    def run(self, user_query: str, session: Session) -> str:
        """Run the agent's loop on a user query. Returns the final answer."""
        pass

    @abstractmethod
    async def run_async(self, user_query: str, session: Session) -> str:
        """Async version of run()."""
        pass

    def act(self, messages: list[dict], session: Session) -> ModelResponse:
        """Call the LLM with the message history and return the response."""
        # Set session ID for logging before the call
        if self.model._session_id != session.id:
            self.model._session_id = session.id
        call_number = session.increment_call_counter()
        return self.model.call(
            messages, self.tools if self.tools else None, call_number
        )

    async def act_async(self, messages: list[dict], session: Session) -> ModelResponse:
        """Async version of act()."""
        if self.model._session_id != session.id:
            self.model._session_id = session.id
        call_number = session.increment_call_counter()
        return await self.model.call_async(
            messages, self.tools if self.tools else None, call_number
        )

    def execute_tool(self, name: str, arguments_json: str) -> str:
        """Execute a tool by name with JSON arguments.

        Args:
            name: Name of the tool to execute.
            arguments_json: JSON string of arguments.

        Returns:
            Tool execution result as a string.
        """
        tool = self._tool_map.get(name)
        if not tool:
            return f"Error: unknown tool '{name}'"
        try:
            args = json.loads(arguments_json)
            result = tool.function(**args)
            return str(result)
        except json.JSONDecodeError as e:
            return f"Error parsing arguments for {name}: {e}"
        except Exception as e:
            return f"Error executing {name}: {e}"


class DefaultAgent(Agent):
    """
    A simple agent that uses the default prompt.

    This agent makes a single LLM call with no tools.
    """

    def __init__(
        self,
        config,
        verbose: bool = True,
    ):
        self.system_prompt_name = config.get_system_prompt_name()
        self.system_prompt = load_prompt(self.system_prompt_name)
        self.model = build_model(
            config.model_name, getattr(config, "log_llm_calls", False)
        )
        self.tools = []
        self._tool_map = {}
        self.max_steps = 1  # Single call, no loop
        self.verbose = verbose

    def run(self, user_query: str, session: Session) -> str:
        """Run a single LLM call and return the response."""
        session.add_step(Step(role="user", content=user_query))

        messages = session.build_messages(self.system_prompt)
        response = self.act(messages, session)

        answer = response.content or ""
        session.add_step(Step(role="assistant", content=answer))
        session.metadata["steps"] = 1
        session.metadata["status"] = "success"

        if self.verbose:
            print(f"[Answer] {answer[:500]}{'...' if len(answer) > 500 else ''}")

        return answer

    async def run_async(self, user_query: str, session: Session) -> str:
        """Async version of run()."""
        session.add_step(Step(role="user", content=user_query))

        messages = session.build_messages(self.system_prompt)
        response = await self.act_async(messages, session)

        answer = response.content or ""
        session.add_step(Step(role="assistant", content=answer))
        session.metadata["steps"] = 1
        session.metadata["status"] = "success"

        if self.verbose:
            print(f"[Answer] {answer[:500]}{'...' if len(answer) > 500 else ''}")

        return answer


class ReActAgent(Agent):
    """
    A ReAct-style agent that uses reasoning and acting.

    This agent runs a loop:
    1. Call LLM with message history
    2. If tool call -> execute tool, add observation, continue
    3. If plain text -> return as final answer
    4. Repeat until done or max_steps
    """

    def __init__(
        self,
        config,
        verbose: bool = True,
    ):
        self.system_prompt_name = config.get_system_prompt_name()
        self.system_prompt = load_prompt(self.system_prompt_name)
        self.model = build_model(
            config.model_name, getattr(config, "log_llm_calls", False)
        )
        self.tools = config.tools or []
        self._tool_map = {t.name: t for t in self.tools}
        self.max_steps = config.max_steps
        self.verbose = verbose

    def run(self, user_query: str, session: Session) -> str:
        """Run the ReAct loop on a user query."""
        session.add_step(Step(role="user", content=user_query))

        for step_idx in range(self.max_steps):
            messages = session.build_messages(self.system_prompt)
            response = self.act(messages, session)

            # --- tool call ---
            if response.tool_calls:
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

                # Execute the tool
                observation = self.execute_tool(tc.name, tc.arguments)
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
                continue

            # --- final answer ---
            answer = response.content or ""
            session.add_step(Step(role="assistant", content=answer))
            session.metadata["steps"] = step_idx + 1
            session.metadata["status"] = "success"

            if self.verbose:
                print(f"[Answer] {answer[:500]}{'...' if len(answer) > 500 else ''}")

            return answer

        # max steps exceeded
        session.metadata["steps"] = self.max_steps
        session.metadata["status"] = "max_steps_exceeded"
        last = (
            session.trajectory[-1].content or session.trajectory[-1].tool_result or ""
        )

        if self.verbose:
            print(f"[Max steps reached] Last output: {last[:200]}")

        return last

    async def run_async(self, user_query: str, session: Session) -> str:
        """Async version of run()."""
        session.add_step(Step(role="user", content=user_query))

        for step_idx in range(self.max_steps):
            messages = session.build_messages(self.system_prompt)
            response = await self.act_async(messages, session)

            # --- tool call ---
            if response.tool_calls:
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

                # Execute the tool
                observation = self.execute_tool(tc.name, tc.arguments)
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
                continue

            # --- final answer ---
            answer = response.content or ""
            session.add_step(Step(role="assistant", content=answer))
            session.metadata["steps"] = step_idx + 1
            session.metadata["status"] = "success"

            if self.verbose:
                print(f"[Answer] {answer[:500]}{'...' if len(answer) > 500 else ''}")

            return answer

        # max steps exceeded
        session.metadata["steps"] = self.max_steps
        session.metadata["status"] = "max_steps_exceeded"
        last = (
            session.trajectory[-1].content or session.trajectory[-1].tool_result or ""
        )

        if self.verbose:
            print(f"[Max steps reached] Last output: {last[:200]}")

        return last
