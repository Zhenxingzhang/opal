"""SessionState manages the conversation state for an agent."""

import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from opal.agentic.llm_model import LLMCallMetrics
from opal.environment.step import Step


@dataclass
class SessionState:
    """Manages the state of a conversation/trajectory."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trajectory: list[Step] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    llm_calls: list[LLMCallMetrics] = field(default_factory=list)

    def add_step(self, step: Step):
        """Add a step to the trajectory."""
        self.trajectory.append(step)

    @property
    def tool_call_count(self) -> int:
        return sum(1 for s in self.trajectory if s.role == "assistant" and s.tool_call)

    @property
    def tool_usage(self) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for s in self.trajectory:
            if s.role == "assistant" and s.tool_call:
                counts[s.tool_call["name"]] += 1
        return dict(counts)

    def reset(self):
        """Clear trajectory and metadata for a fresh run, generate new id."""
        self.id = str(uuid.uuid4())
        self.trajectory = []
        self.metadata = {}
        self.llm_calls = []

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
