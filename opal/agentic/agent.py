"""
Agent interface for agentic design.

The Agent class defines the interface for agents that act as policy objects.
Each agent provides hooks for building messages, recording tool cycles, and
pre-loop setup. The execution loop itself lives in the Runner.
"""

import json
import logging
import uuid
from pathlib import Path

from opal.agentic.llm_model import (
    LLMModel,
    ModelResponse,
    OpenAIModel,
    AnthropicModel,
)
from opal.environment.tool import Tool
from opal.environment.session import Session
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
    Runner owns the execution loop and mediates between Agent and
    ToolEnvironment.

    Subclasses customise behaviour by overriding:
    - ``build_messages``   — how the message list is constructed
    - ``record_tool_cycle`` — how a tool call + observation is recorded
    - ``pre_loop``          — one-time setup before the loop starts
    """

    system_prompt_name = ""
    system_prompt: str = ""
    model: LLMModel
    tools: list[Tool] = []
    max_steps: int = 10
    verbose: bool = True

    def _init_common(self, config) -> None:
        """Shared initialisation for all agent subclasses."""
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

    def build_messages(self, session: Session) -> list[dict]:
        """Build the message list for the LLM API call."""
        return session.build_messages(self.system_prompt)

    def record_tool_cycle(
        self,
        response: ModelResponse,
        observation: str,
        session: Session,
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

    def pre_loop(self, session: Session, env) -> None:
        """Called once before the execution loop starts.

        Override to inject retrieval steps or other setup.  Receives the
        ToolEnvironment so subclasses can call ``env.execute_tool()`` for
        programmatic tool use without owning the environment.
        """

    # ------------------------------------------------------------------
    # Finish helpers (used by Runner)
    # ------------------------------------------------------------------

    def finish(
        self, response: ModelResponse, session: Session, step_idx: int
    ) -> str:
        """Record the final answer and return it."""
        answer = response.content or ""
        session.add_step(Step(role="assistant", content=answer))
        session.metadata["steps"] = step_idx + 1
        session.metadata["status"] = "success"
        self._log_answer(answer)
        return answer

    def max_steps_exceeded(self, session: Session) -> str:
        """Handle the case where the agent exhausted its step budget."""
        session.metadata["steps"] = self.max_steps
        session.metadata["status"] = "max_steps_exceeded"
        last = (
            session.trajectory[-1].content or session.trajectory[-1].tool_result or ""
        )
        if self.verbose:
            print(f"[Max steps reached] Last output: {last[:200]}")
        return last

    # ------------------------------------------------------------------
    # LLM interaction (not overridden by subclasses)
    # ------------------------------------------------------------------

    def act(self, messages: list[dict], session: Session) -> ModelResponse:
        """Call the LLM with the message history and return the response."""
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
    """A simple agent that uses the default prompt.

    Makes a single LLM call with no tools.  All default hooks work as-is.
    """

    def __init__(self, config, **kwargs):
        self._init_common(config)
        self.tools = []
        self.max_steps = 1


class ReActAgent(Agent):
    """A ReAct-style agent that uses reasoning and acting.

    Loop (driven by Runner):
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
        self.max_steps = config.max_steps


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
        self.max_steps = config.max_steps

    def build_messages(self, session: Session) -> list[dict]:
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
        session: Session,
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


class RAGAgent(Agent):
    """Retrieval-Augmented Generation agent.

    Before calling the LLM, this agent programmatically runs a semantic search
    over its indexed documents using the ``search_pdf`` tool via ToolEnvironment.
    The retrieval results are injected into the session as a tool-call step so
    that the LLM sees them as context.  Then a single LLM call produces the
    final answer.

    Overrides ``pre_loop`` to perform the retrieval injection.
    """

    def __init__(
        self,
        config,
        **kwargs,
    ):
        self._init_common(config)
        from opal.environment.tool_environment import SEARCH_PDF_TOOL

        self.tools = [SEARCH_PDF_TOOL]
        self.max_steps = 1
        self.retriever_top_k = getattr(config, "retriever_top_k", 5)

    def pre_loop(self, session: Session, env) -> None:
        """Inject semantic search results before the LLM call."""
        if env.retriever and env.retriever.document_count > 0:
            query = session.trajectory[0].content  # user query from first step
            arguments_json = json.dumps(
                {"query": query, "top_k": self.retriever_top_k}
            )
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

            retrieval_text = env.execute_tool("search_pdf", arguments_json)

            session.add_step(
                Step(
                    role="tool",
                    tool_call=tool_call_record,
                    tool_result=retrieval_text,
                )
            )

            if self.verbose:
                print(f"[RAG] search_pdf returned {len(retrieval_text)} chars")
