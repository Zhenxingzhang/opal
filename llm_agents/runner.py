"""
Runner orchestrates agent execution.

The Runner:
- Creates and manages the Session
- Delegates execution to the Agent
- Provides utilities for debugging/inspection
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from llm_agents.environment.step import Step
from llm_agents.agentic.tool import Tool
from llm_agents.environment.session import Session
from llm_agents.agentic.agent import Agent, ReActAgent, DefaultAgent


# Maps agent_name to (AgentClass, prompt_file_name)
AGENT_REGISTRY = {
    "default": (DefaultAgent, "default_prompt"),
    "react": (ReActAgent, "react_prompt"),
}


@dataclass
class AgentConfig:
    """Configuration for Runner and Agent."""

    model_name: str = "gpt-4o-2024-11-20"
    agent_name: str = "react"  # "react" or "default" - determines agent type and prompt
    max_steps: int = 10
    log_llm_calls: bool = False
    tools: list[Tool] = field(default_factory=list)
    system_prompt_name: str | None = None  # Optional custom prompt name

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

    The Runner creates a session, delegates to the appropriate agent,
    and provides access to results.
    """

    def __init__(
        self,
        config: AgentConfig | None = None,
        verbose: bool = False,
    ):
        self.config = config or AgentConfig()
        self.agent = self._build_agent(verbose)
        self.session = Session()
        self.verbose = verbose

    def _build_agent(self, verbose: bool) -> Agent:
        """Build the appropriate agent based on config."""
        agent_name = self.config.agent_name
        if agent_name not in AGENT_REGISTRY:
            agent_name = "react"

        agent_class, _ = AGENT_REGISTRY[agent_name]

        if agent_class == DefaultAgent:
            return DefaultAgent(
                config=self.config,
                verbose=verbose,
            )
        else:
            return ReActAgent(
                config=self.config,
                verbose=verbose,
            )

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
        }

        return self.agent.run(user_query, self.session)

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
        }

        return await self.agent.run_async(user_query, self.session)

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
