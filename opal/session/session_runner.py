"""
SessionRunner orchestrates agent execution.

The SessionRunner:
- Creates and manages the SessionState
- Owns the canonical execution loop (mediates between Agent and ToolEnvironment)
- Provides utilities for debugging/inspection
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Any

from opal.config import AgentConfig, SessionConfig, AGENT_REGISTRY, TOOL_REGISTRY
from opal.environment.step import Step
from opal.environment.tool_environment import ToolEnvironment
from opal.session.session import SessionState
from opal.agentic.agent import Agent
from opal.agentic.llm_model import ModelResponse

logger = logging.getLogger(__name__)


class SessionRunner:
    """
    Orchestrates agent execution.

    The SessionRunner creates a session state, owns the canonical execution loop that
    mediates between Agent (policy) and ToolEnvironment (MDP), and
    provides access to results.
    """

    def __init__(
        self,
        session_config: SessionConfig,
        agent_config: AgentConfig,
        env: ToolEnvironment| None = None,
        session_timestamp: str | None = None,
    ):
        self.session_config = session_config
        self.agent_config = agent_config
        self.session_timestamp = session_timestamp or datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
        self.env = env
        self.agent = self._build_agent()
        self.session_state = SessionState()

    def _build_agent(self) -> Agent:
        """Build the appropriate agent based on config."""
        agent_name = self.agent_config.agent_name

        assert agent_name in AGENT_REGISTRY, (
            f"Error: agent type {agent_name} is not supported!"
        )

        agent_class, _ = AGENT_REGISTRY[agent_name]

        agent = agent_class(config=self.agent_config)

        # Build ToolEnvironment: merge caller-supplied env with the tool map
        # derived from the agent's declared tools.
        tool_map = {t.name: t for t in agent.tools}
        # Ensure search_pdf is available when enable_search is on,
        # even if the agent doesn't declare it as a tool.
        if self.session_config.enable_search and "search_pdf" not in tool_map:
            tool_map["search_pdf"] = TOOL_REGISTRY["search_pdf"]
        if self.env is not None:
            self.env.tool_map = tool_map
        else:
            self.env = ToolEnvironment(tool_map=tool_map)
        return agent

    # ------------------------------------------------------------------
    # Finish helpers
    # ------------------------------------------------------------------

    def _prepare_call(self, session: SessionState) -> int:
        """Sync model session-id and increment call counter before an LLM call."""
        if self.agent.model._session_id != session.id:
            self.agent.model._session_id = session.id
        return session.increment_call_counter()

    def _finish(
        self, response: ModelResponse, session: SessionState, step_idx: int
    ) -> str:
        """Record the final answer and return it."""
        answer = response.content or ""
        session.add_step(Step(role="assistant", content=answer))
        session.metadata["steps"] = step_idx + 1
        session.metadata["status"] = "success"
        if self.agent.verbose:
            print(f"[Answer] {answer[:500]}{'...' if len(answer) > 500 else ''}")
        return answer

    def _max_steps_exceeded(self, session: SessionState, max_steps: int) -> str:
        """Handle the case where the agent exhausted its step budget."""
        session.metadata["steps"] = max_steps
        session.metadata["status"] = "max_steps_exceeded"
        last = (
            session.trajectory[-1].content or session.trajectory[-1].tool_result or ""
        )
        if self.agent.verbose:
            print(f"[Max steps reached] Last output: {last[:200]}")
        return last



    # ------------------------------------------------------------------
    # Search injection
    # ------------------------------------------------------------------

    def _inject_search(self, user_query: str) -> None:
        """Manually call search_pdf with the user query and inject results into the trajectory.

        This simulates the agent having called search_pdf itself, so the LLM
        sees the retrieved context without spending an agentic step.
        """
        tool_call_record = {
            "id": f"injected_{uuid.uuid4().hex[:8]}",
            "name": "search_pdf",
            "arguments": json.dumps({"query": user_query}),
        }
        observation = self.env.execute_tool(
            tool_call_record["name"], tool_call_record["arguments"]
        )
        self.session_state.add_step(
            Step(role="assistant", content=None, tool_call=tool_call_record)
        )
        self.session_state.add_step(
            Step(role="tool", tool_call=tool_call_record, tool_result=observation)
        )
        if self.agent.verbose:
            print(f"[Injected search_pdf] query={user_query[:100]}")
            print(
                f"  Result: {observation[:200]}{'...' if len(observation) > 200 else ''}"
            )

    # ------------------------------------------------------------------
    # Canonical execution loop
    # ------------------------------------------------------------------

    def _execute_loop(self, user_query: str) -> str:
        """Run the canonical agent loop (sync)."""
        session = self.session_state
        max_steps = self.session_config.max_steps
        session.add_step(Step(role="user", content=user_query))
        if self.session_config.enable_search:
            self._inject_search(user_query)

        for step_idx in range(max_steps):
            messages = self.agent.build_messages(session)
            call_number = self._prepare_call(session)
            response = self.agent.act(messages, call_number)

            if response.tool_calls:
                tc = response.tool_calls[0]
                observation = self.env.execute_tool(tc.name, tc.arguments)
                self.agent.record_tool_cycle(response, observation, session, step_idx)
                continue

            return self._finish(response, session, step_idx)

        return self._max_steps_exceeded(session, max_steps)

    async def _execute_loop_async(self, user_query: str) -> str:
        """Run the canonical agent loop (async)."""
        session = self.session_state
        max_steps = self.session_config.max_steps
        session.add_step(Step(role="user", content=user_query))
        if self.session_config.enable_search:
            self._inject_search(user_query)

        for step_idx in range(max_steps):
            messages = self.agent.build_messages(session)
            call_number = self._prepare_call(session)
            response = await self.agent.act_async(messages, call_number)

            if response.tool_calls:
                tc = response.tool_calls[0]
                observation = self.env.execute_tool(tc.name, tc.arguments)
                self.agent.record_tool_cycle(response, observation, session, step_idx)
                continue

            return self._finish(response, session, step_idx)

        return self._max_steps_exceeded(session, max_steps)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self):
        """Clear session for a fresh run."""
        self.session_state.reset()

    def run(self, user_query: str) -> str:
        """
        Run the agent on a user query. Returns the final text answer.
        The full trajectory is stored in self.session.trajectory.
        """
        self.session_state.reset()
        self.session_state.metadata = {
            "timestamp": datetime.now().isoformat(),
            "query": user_query,
            "agent": self.agent.system_prompt_name,
            "model": self.agent.model.get_name(),
            "retriever": self.env.retriever.summary() if self.env.retriever else None,
        }

        if self.agent_config.log_llm_calls:
            self.agent.model.set_logging(
                True,
                session_id=self.session_state.id,
                timestamp=self.session_timestamp,
                logging_dir=self.session_config.logging_dir,
            )

        result = self._execute_loop(user_query)
        self._log_trajectory()
        return result

    async def run_async(self, user_query: str) -> str:
        """
        Async version of run(). Returns the final text answer.
        The full trajectory is stored in self.session.trajectory.
        """
        self.session_state.reset()
        self.session_state.metadata = {
            "timestamp": datetime.now().isoformat(),
            "query": user_query,
            "agent": self.agent.system_prompt_name,
            "model": self.agent.model.get_name(),
            "retriever": self.env.retriever.summary() if self.env.retriever else None,
        }

        if self.agent_config.log_llm_calls:
            self.agent.model.set_logging(
                True,
                session_id=self.session_state.id,
                timestamp=self.session_timestamp,
                logging_dir=self.session_config.logging_dir,
            )

        result = await self._execute_loop_async(user_query)
        self._log_trajectory()
        return result

    def _log_trajectory(self):
        """Save the full trajectory to a JSON file in the same folder as LLM call logs."""
        if not self.agent_config.log_llm_calls:
            return

        session_suffix = self.session_state.id[:8]
        output_dir = (
            self.session_config.logging_dir
            / f"{session_suffix}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        trajectory_data = {
            "session_id": self.session_state.id,
            "metadata": self.session_state.metadata,
            "trajectory": self.session_state.get_trajectory_as_dicts(),
        }

        output_file = output_dir / f"{session_suffix}_trajectory.json"
        with open(output_file, "w") as f:
            json.dump(trajectory_data, f, indent=2)

        logger.info(f"Trajectory logged to {output_file}")

    def print_trajectory(self):
        """Pretty-print the trajectory for debugging."""
        print("=" * 60)
        print("TRAJECTORY")
        print("=" * 60)
        for i, step in enumerate(self.session_state.trajectory):
            print(f"\n--- Step {i} ({step.role}) ---")
            if step.content:
                print(f"Content: {step.content[:300]}")
            if step.tool_call:
                print(
                    f"Tool:    {step.tool_call['name']}({step.tool_call['arguments']})"
                )
            if step.tool_result:
                print(f"Result:  {step.tool_result[:300]}")
        print("\n" + "=" * 60)
        print(f"Metadata: {json.dumps(self.session_state.metadata, indent=2)}")

    @property
    def trajectory(self) -> list[Step]:
        """Access trajectory from session."""
        return self.session_state.trajectory

    @property
    def metadata(self) -> dict[str, Any]:
        """Access metadata from session."""
        return self.session_state.metadata
