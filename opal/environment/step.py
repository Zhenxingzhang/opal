"""Trajectory step recording for the agent environment."""

from dataclasses import dataclass


@dataclass
class Step:
    """One step in the agent trajectory."""

    role: str  # "assistant" | "tool" | "user"
    content: str | None = None
    tool_call: dict | None = None  # {name, arguments, id}
    tool_result: str | None = None

    def to_message(self) -> dict:
        if self.role == "assistant" and self.tool_call:
            return {
                "role": "assistant",
                "content": self.content,
                "tool_calls": [
                    {
                        "id": self.tool_call["id"],
                        "type": "function",
                        "function": {
                            "name": self.tool_call["name"],
                            "arguments": self.tool_call["arguments"],
                        },
                    }
                ],
            }
        if self.role == "tool":
            return {
                "role": "tool",
                "tool_call_id": self.tool_call["id"],
                "content": self.tool_result,
            }
        return {"role": self.role, "content": self.content}
