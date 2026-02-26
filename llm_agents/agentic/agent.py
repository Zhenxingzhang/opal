"""
Agent interface for agentic design.

The Agent class defines the interface for agents that can process queries
using tools and LLM models. Each agent has its own execution loop.
"""

import json
import logging
import uuid
from abc import ABC, abstractmethod
from pathlib import Path

from llm_agents.agentic.llm_model import (
    LLMModel,
    ModelResponse,
    OpenAIModel,
    AnthropicModel,
)
from llm_agents.environment.tool import Tool
from llm_agents.environment.session import Session
from llm_agents.environment.step import Step
from llm_agents.environment.tool_environment import ToolEnvironment


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
    env: ToolEnvironment

    def _init_common(self, config, verbose: bool = True) -> None:
        """Shared initialisation for all agent subclasses."""
        self.system_prompt_name = config.get_system_prompt_name()
        self.system_prompt = load_prompt(self.system_prompt_name)
        self.model = build_model(
            config.model_name, getattr(config, "log_llm_calls", False)
        )
        self.verbose = verbose

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

    def _log_answer(self, answer: str) -> None:
        if self.verbose:
            print(f"[Answer] {answer[:500]}{'...' if len(answer) > 500 else ''}")


class DefaultAgent(Agent):
    """
    A simple agent that uses the default prompt.

    This agent makes a single LLM call with no tools.
    """

    def __init__(self, config, verbose: bool = True, **kwargs):
        self._init_common(config, verbose)
        self.tools = []
        self.max_steps = 1

    def _run_impl(
        self, user_query: str, session: Session, response: ModelResponse
    ) -> str:
        answer = response.content or ""
        session.add_step(Step(role="assistant", content=answer))
        session.metadata["steps"] = 1
        session.metadata["status"] = "success"
        self._log_answer(answer)
        return answer

    def run(self, user_query: str, session: Session) -> str:
        """Run a single LLM call and return the response."""
        session.add_step(Step(role="user", content=user_query))
        messages = session.build_messages(self.system_prompt)
        response = self.act(messages, session)
        return self._run_impl(user_query, session, response)

    async def run_async(self, user_query: str, session: Session) -> str:
        """Async version of run()."""
        session.add_step(Step(role="user", content=user_query))
        messages = session.build_messages(self.system_prompt)
        response = await self.act_async(messages, session)
        return self._run_impl(user_query, session, response)


class ReActAgent(Agent):
    """
    A ReAct-style agent that uses reasoning and acting.

    This agent runs a loop:
    1. Call LLM with message history
    2. If tool call -> execute tool, add observation, continue
    3. If plain text -> return as final answer
    4. Repeat until done or max_steps
    """

    def __init__(self, config, verbose: bool = True, **kwargs):
        self._init_common(config, verbose)
        self.tools = config.tools or []
        self.max_steps = config.max_steps

    def _handle_tool_call(
        self, response: ModelResponse, session: Session, step_idx: int
    ) -> None:
        """Record a tool call and its result in the session."""
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

        observation = self.env.execute_tool(tc.name, tc.arguments)
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

    def _finish(self, response: ModelResponse, session: Session, step_idx: int) -> str:
        """Record the final answer and return it."""
        answer = response.content or ""
        session.add_step(Step(role="assistant", content=answer))
        session.metadata["steps"] = step_idx + 1
        session.metadata["status"] = "success"
        self._log_answer(answer)
        return answer

    def _max_steps_exceeded(self, session: Session) -> str:
        session.metadata["steps"] = self.max_steps
        session.metadata["status"] = "max_steps_exceeded"
        last = (
            session.trajectory[-1].content or session.trajectory[-1].tool_result or ""
        )
        if self.verbose:
            print(f"[Max steps reached] Last output: {last[:200]}")
        return last

    def run(self, user_query: str, session: Session) -> str:
        """Run the ReAct loop on a user query."""
        session.add_step(Step(role="user", content=user_query))

        for step_idx in range(self.max_steps):
            messages = session.build_messages(self.system_prompt)
            response = self.act(messages, session)

            if response.tool_calls:
                self._handle_tool_call(response, session, step_idx)
                continue

            return self._finish(response, session, step_idx)

        return self._max_steps_exceeded(session)

    async def run_async(self, user_query: str, session: Session) -> str:
        """Async version of run()."""
        session.add_step(Step(role="user", content=user_query))

        for step_idx in range(self.max_steps):
            messages = session.build_messages(self.system_prompt)
            response = await self.act_async(messages, session)

            if response.tool_calls:
                self._handle_tool_call(response, session, step_idx)
                continue

            return self._finish(response, session, step_idx)

        return self._max_steps_exceeded(session)


class AdvancedReActAgent(Agent):
    """
    A ReAct-style agent with explicit Thought/Action/Observation separation.

    Like ReActAgent, this agent makes one LLM call per cycle.  However, each
    cycle's response is split into three distinct trajectory steps:

        Step(role="assistant", content="I need to...")           # Thought
        Step(role="assistant", content=None, tool_call={...})    # Action
        Step(role="tool", tool_call={...}, tool_result="...")     # Observation

    This gives a cleaner signal for downstream RL analysis compared to
    ReActAgent which merges Thought+Action into a single step.

    When building messages for the LLM API, consecutive Thought+Action step
    pairs are merged back into one ``assistant`` message (APIs reject
    consecutive ``assistant`` messages).
    """

    def __init__(self, config, verbose: bool = True, **kwargs):
        self._init_common(config, verbose)
        self.tools = config.tools or []
        self.max_steps = config.max_steps

    # ------------------------------------------------------------------
    # Trajectory helpers
    # ------------------------------------------------------------------

    def _handle_thought_action_observation(
        self, response: ModelResponse, session: Session, cycle_idx: int
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
        observation = self.env.execute_tool(tc.name, tc.arguments)
        session.add_step(
            Step(role="tool", tool_call=tool_call_record, tool_result=observation)
        )

        if self.verbose:
            print(f"--- Cycle {cycle_idx + 1} ---")
            print(
                f"[Thought] {thought_text[:300]}{'...' if len(thought_text) > 300 else ''}"
            )
            print(f"[Action]  {tc.name}({tc.arguments})")
            print(
                f"[Observation] {observation[:200]}{'...' if len(observation) > 200 else ''}"
            )

    def _build_messages_for_api(self, session: Session) -> list[dict]:
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

    # ------------------------------------------------------------------
    # Finish / overflow helpers
    # ------------------------------------------------------------------

    def _finish(self, response: ModelResponse, session: Session, cycle_idx: int) -> str:
        """Record the final answer and return it."""
        answer = response.content or ""
        session.add_step(Step(role="assistant", content=answer))
        session.metadata["steps"] = cycle_idx + 1
        session.metadata["status"] = "success"
        self._log_answer(answer)
        return answer

    def _max_steps_exceeded(self, session: Session) -> str:
        session.metadata["steps"] = self.max_steps
        session.metadata["status"] = "max_steps_exceeded"
        last = (
            session.trajectory[-1].content or session.trajectory[-1].tool_result or ""
        )
        if self.verbose:
            print(f"[Max steps reached] Last output: {last[:200]}")
        return last

    # ------------------------------------------------------------------
    # Main loops
    # ------------------------------------------------------------------

    def run(self, user_query: str, session: Session) -> str:
        """Run the AdvancedReAct loop on a user query."""
        session.add_step(Step(role="user", content=user_query))

        for cycle_idx in range(self.max_steps):
            messages = self._build_messages_for_api(session)
            response = self.act(messages, session)

            if response.tool_calls:
                self._handle_thought_action_observation(response, session, cycle_idx)
                continue

            return self._finish(response, session, cycle_idx)

        return self._max_steps_exceeded(session)

    async def run_async(self, user_query: str, session: Session) -> str:
        """Async version of run()."""
        session.add_step(Step(role="user", content=user_query))

        for cycle_idx in range(self.max_steps):
            messages = self._build_messages_for_api(session)
            response = await self.act_async(messages, session)

            if response.tool_calls:
                self._handle_thought_action_observation(response, session, cycle_idx)
                continue

            return self._finish(response, session, cycle_idx)

        return self._max_steps_exceeded(session)


class RAGAgent(Agent):
    """
    Retrieval-Augmented Generation agent.

    Before calling the LLM, this agent programmatically runs a semantic search
    over its indexed documents using the ``search_pdf`` tool via ToolEnvironment.
    The retrieval results are injected into the session as a tool-call step so
    that the LLM sees them as context.  Then a single LLM call produces the
    final answer.
    """

    def __init__(
        self,
        config,
        verbose: bool = True,
        retriever_top_k: int = 5,
        **kwargs,
    ):
        self._init_common(config, verbose)
        from llm_agents.environment.tool_environment import SEARCH_PDF_TOOL

        self.tools = [SEARCH_PDF_TOOL]
        self.max_steps = 1
        self.retriever_top_k = retriever_top_k

    def _inject_search_step(self, query: str, session: Session) -> str:
        """Execute ``search_pdf`` via the ToolEnvironment and record the
        tool-call / tool-result step pair in the session.

        Returns the retrieval text.
        """
        arguments_json = json.dumps({"query": query, "top_k": self.retriever_top_k})
        tool_call_id = f"retrieval_{uuid.uuid4().hex[:8]}"
        tool_call_record = {
            "id": tool_call_id,
            "name": "search_pdf",
            "arguments": arguments_json,
        }

        session.add_step(
            Step(
                role="assistant",
                content="Searching indexed documents for relevant context.",
                tool_call=tool_call_record,
            )
        )

        retrieval_text = self.env.execute_tool("search_pdf", arguments_json)

        session.add_step(
            Step(
                role="tool",
                tool_call=tool_call_record,
                tool_result=retrieval_text,
            )
        )

        if self.verbose:
            print(f"[RAG] search_pdf returned {len(retrieval_text)} chars")

        return retrieval_text

    def _run_impl(
        self, user_query: str, session: Session, response: ModelResponse
    ) -> str:
        answer = response.content or ""
        session.add_step(Step(role="assistant", content=answer))
        session.metadata["steps"] = 1
        session.metadata["status"] = "success"
        self._log_answer(answer)
        return answer

    def run(self, user_query: str, session: Session) -> str:
        """Run the RAG agent: retrieve context, then single LLM call."""
        session.add_step(Step(role="user", content=user_query))

        if self.env.retriever and self.env.retriever.document_count > 0:
            self._inject_search_step(user_query, session)

        messages = session.build_messages(self.system_prompt)
        response = self.act(messages, session)
        return self._run_impl(user_query, session, response)

    async def run_async(self, user_query: str, session: Session) -> str:
        """Async version of run()."""
        session.add_step(Step(role="user", content=user_query))

        if self.env.retriever and self.env.retriever.document_count > 0:
            self._inject_search_step(user_query, session)

        messages = session.build_messages(self.system_prompt)
        response = await self.act_async(messages, session)
        return self._run_impl(user_query, session, response)
