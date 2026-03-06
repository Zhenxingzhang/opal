"""SessionState manages the conversation state for an agent."""

import uuid
from dataclasses import dataclass, field
from typing import Any

from opal.environment.step import Step


@dataclass
class SessionState:
    """Manages the state of a conversation/trajectory."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trajectory: list[Step] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    call_counter: int = 0

    def add_step(self, step: Step):
        """Add a step to the trajectory."""
        self.trajectory.append(step)

    def increment_call_counter(self) -> int:
        """Increment and return the call counter."""
        self.call_counter += 1
        return self.call_counter

    def reset(self):
        """Clear trajectory and metadata for a fresh run, generate new id."""
        self.id = str(uuid.uuid4())
        self.trajectory = []
        self.metadata = {}
        self.call_counter = 0

    def build_messages(self, system_prompt: str | None = None) -> list[dict]:
        """Build message list for LLM API from optional system prompt + trajectory."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        for step in self.trajectory:
            messages.append(step.to_message())
        return messages

    def get_trajectory_as_dicts(self) -> list[dict]:
        """Return trajectory as a list of serializable dicts (for logging/RL)."""
        return [
            {
                "role": s.role,
                "content": s.content,
                "tool_call": s.tool_call,
                "tool_result": s.tool_result,
            }
            for s in self.trajectory
        ]
