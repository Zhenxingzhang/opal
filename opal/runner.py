"""
Runner orchestrates agent execution.

The Runner:
- Creates and manages the Session
- Owns the canonical execution loop (mediates between Agent and ToolEnvironment)
- Provides utilities for debugging/inspection
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from opal.environment.step import Step
from opal.environment.tool_environment import ToolEnvironment
from opal.environment.tool import Tool
from opal.environment.session import Session
from opal.agentic.agent import (
    Agent,
    ReActAgent,
    DefaultAgent,
    AdvancedReActAgent,
    RAGAgent,
)
from opal.agentic.llm_model import RESULTS_DIR

logger = logging.getLogger(__name__)


# Maps agent_name to (AgentClass, prompt_file_name)
AGENT_REGISTRY = {
    "default": (DefaultAgent, "default_prompt"),
    "react": (ReActAgent, "react_prompt"),
    "advanced_react": (AdvancedReActAgent, "advanced_react_prompt"),
    "rag": (RAGAgent, "rag_prompt"),
}


@dataclass
class RunnerConfig:
    """Configuration for the Runner execution loop."""

    max_steps: int = 10
    parallelism: int = 1


@dataclass
class AgentConfig:
    """Configuration for Agent policy."""

    model_name: str = "gpt-4o-2024-11-20"
    agent_name: str = "react"  # "react" or "default" - determines agent type and prompt
    temperature: float = 0.0
    log_llm_calls: bool = False
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


class Runner:
    """
    Orchestrates agent execution.

    The Runner creates a session, owns the canonical execution loop that
    mediates between Agent (policy) and ToolEnvironment (MDP), and
    provides access to results.
    """

    def __init__(
        self,
        config: AgentConfig | None = None,
        runner_config: RunnerConfig | None = None,
        run_timestamp: str | None = None,
        env: ToolEnvironment | None = None,
    ):
        self.config = config or AgentConfig()
        self.runner_config = runner_config or RunnerConfig()
        self.run_timestamp = run_timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.env = env
        self.agent = self._build_agent()
        self.session = Session()

    def _build_agent(self) -> Agent:
        """Build the appropriate agent based on config."""
        agent_name = self.config.agent_name

        assert agent_name in AGENT_REGISTRY, (
            f"Error: agent type {agent_name} is not supported!"
        )

        agent_class, _ = AGENT_REGISTRY[agent_name]

        agent = agent_class(config=self.config)

        # Build ToolEnvironment: merge caller-supplied env with the tool map
        # derived from the agent's declared tools.
        tool_map = {t.name: t for t in agent.tools}
        if self.env is not None:
            self.env.tool_map = tool_map
        else:
            self.env = ToolEnvironment(tool_map=tool_map)
        return agent

    # ------------------------------------------------------------------
    # Canonical execution loop
    # ------------------------------------------------------------------

    def _execute_loop(self, user_query: str) -> str:
        """Run the canonical agent loop (sync)."""
        session = self.session
        max_steps = self.runner_config.max_steps
        session.add_step(Step(role="user", content=user_query))
        self.agent.pre_loop(session, self.env)

        for step_idx in range(max_steps):
            messages = self.agent.build_messages(session)
            response = self.agent.act(messages, session)

            if response.tool_calls:
                tc = response.tool_calls[0]
                observation = self.env.execute_tool(tc.name, tc.arguments)
                self.agent.record_tool_cycle(response, observation, session, step_idx)
                continue

            return self.agent.finish(response, session, step_idx)

        return self.agent.max_steps_exceeded(session, max_steps)

    async def _execute_loop_async(self, user_query: str) -> str:
        """Run the canonical agent loop (async)."""
        session = self.session
        max_steps = self.runner_config.max_steps
        session.add_step(Step(role="user", content=user_query))
        self.agent.pre_loop(session, self.env)

        for step_idx in range(max_steps):
            messages = self.agent.build_messages(session)
            response = await self.agent.act_async(messages, session)

            if response.tool_calls:
                tc = response.tool_calls[0]
                observation = self.env.execute_tool(tc.name, tc.arguments)
                self.agent.record_tool_cycle(response, observation, session, step_idx)
                continue

            return self.agent.finish(response, session, step_idx)

        return self.agent.max_steps_exceeded(session, max_steps)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self):
        """Clear session for a fresh run."""
        self.session.reset()

    def run(self, user_query: str) -> str:
        """
        Run the agent on a user query. Returns the final text answer.
        The full trajectory is stored in self.session.trajectory.
        """
        self.session.reset()
        self.session.metadata = {
            "timestamp": datetime.now().isoformat(),
            "query": user_query,
            "agent": self.agent.system_prompt_name,
            "model": self.agent.model.get_name(),
            "retriever": self.env.retriever.summary() if self.env.retriever else None,
        }

        if self.config.log_llm_calls:
            self.agent.model.set_logging(
                True, session_id=self.session.id, timestamp=self.run_timestamp
            )

        result = self._execute_loop(user_query)
        self._log_trajectory()
        return result

    async def run_async(self, user_query: str) -> str:
        """
        Async version of run(). Returns the final text answer.
        The full trajectory is stored in self.session.trajectory.
        """
        self.session.reset()
        self.session.metadata = {
            "timestamp": datetime.now().isoformat(),
            "query": user_query,
            "agent": self.agent.system_prompt_name,
            "model": self.agent.model.get_name(),
            "retriever": self.env.retriever.summary() if self.env.retriever else None,
        }

        if self.config.log_llm_calls:
            self.agent.model.set_logging(
                True, session_id=self.session.id, timestamp=self.run_timestamp
            )

        result = await self._execute_loop_async(user_query)
        self._log_trajectory()
        return result

    def _log_trajectory(self):
        """Save the full trajectory to a JSON file in the same folder as LLM call logs."""
        if not self.config.log_llm_calls:
            return

        session_suffix = self.session.id[:8] if self.session.id else "unknown"
        output_dir = (
            RESULTS_DIR / "llm_calls" / f"{self.run_timestamp}/{session_suffix}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        trajectory_data = {
            "session_id": self.session.id,
            "metadata": self.session.metadata,
            "trajectory": self.session.get_trajectory_as_dicts(),
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
        for i, step in enumerate(self.session.trajectory):
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
        print(f"Metadata: {json.dumps(self.session.metadata, indent=2)}")

    @property
    def trajectory(self) -> list[Step]:
        """Access trajectory from session."""
        return self.session.trajectory

    @property
    def metadata(self) -> dict[str, Any]:
        """Access metadata from session."""
        return self.session.metadata
