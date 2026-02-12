"""Tool definition for the agent environment."""

from dataclasses import dataclass
from typing import Callable


@dataclass
class Tool:
    """A tool the agent can invoke."""

    name: str
    description: str
    parameters: dict  # JSON-schema style parameter spec
    function: Callable  # the actual callable

    def schema(self) -> dict:
        """Return OpenAI-compatible function schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
