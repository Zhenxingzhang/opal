"""
Simplified LLM model abstraction.

Inspired by funbench's Model class but simplified for this codebase.

Usage:
    from source.llm_model import LLMModel, OpenAIModel

    model = OpenAIModel(model_name="gpt-4o")
    response = model.call(messages, tools)
"""

import asyncio
import os
import time
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import openai

from llm_agents.agentic.tool import Tool

load_dotenv()

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent.parent / "results"


@dataclass
class ToolCallInfo:
    """Information about a tool call made by the model."""

    id: str
    name: str
    arguments: str  # JSON string of arguments


@dataclass
class ModelResponse:
    """Response from an LLM model."""

    content: str | None
    tool_calls: list[ToolCallInfo] | None
    raw_response: Any = None  # Original response object for debugging


class LLMModel(ABC):
    """Base class for language models."""

    log_llm_calls: bool = False
    _log_timestamp: str | None = None
    _session_id: str | None = None

    @abstractmethod
    def call(
        self,
        messages: list[dict],
        tools: list[Tool] | None = None,
        call_number: int | None = None,
    ) -> ModelResponse:
        """Call the model with messages and optional tools.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            tools: Optional list of Tool objects available for the model.
            call_number: Optional call number for logging.

        Returns:
            ModelResponse with content and/or tool_calls.
        """
        pass

    @abstractmethod
    async def call_async(
        self,
        messages: list[dict],
        tools: list[Tool] | None = None,
        call_number: int | None = None,
    ) -> ModelResponse:
        """Async version of call().

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            tools: Optional list of Tool objects available for the model.
            call_number: Optional call number for logging.

        Returns:
            ModelResponse with content and/or tool_calls.
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get the model name/identifier."""
        pass

    def set_logging(self, enabled: bool, session_id: str | None = None, timestamp: str | None = None):
        """Enable or disable LLM call logging.

        Args:
            enabled: Whether to log LLM calls.
            session_id: Session ID for organizing logs.
            timestamp: Run-level timestamp for the log folder. If not provided, one is generated.
        """
        self.log_llm_calls = enabled
        self._session_id = session_id
        if enabled:
            self._log_timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")

    def _log_llm_call(
        self, kwargs: dict, response: ModelResponse, call_number: int | None = None
    ):
        """Log raw LLM call kwargs and response to file."""
        if not self.log_llm_calls:
            return

        # Use provided call number or default to 0
        call_num = call_number if call_number is not None else 0

        # Ensure timestamp is set
        if self._log_timestamp is None:
            self._log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Prepare kwargs for JSON serialization (convert Tool objects to schemas)
        serializable_kwargs = {}
        for k, v in kwargs.items():
            if k == "tools" and v:
                serializable_kwargs[k] = v  # Already converted to schema dicts
            else:
                serializable_kwargs[k] = v

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "call_number": call_num,
            "session_id": self._session_id,
            "model_name": self.get_name(),
            "llm_raw_input": serializable_kwargs,
            "llm_raw_output": {
                "content": response.content,
                "tool_calls": [
                    {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                    for tc in response.tool_calls
                ]
                if response.tool_calls
                else None,
            },
        }

        # Create output directory: results/llm_calls/{timestamp}_{session_id_prefix}/
        session_suffix = self._session_id[:8] if self._session_id else "unknown"
        output_dir = (
            RESULTS_DIR / "llm_calls" / f"{self._log_timestamp}/{session_suffix}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        # Write to file
        output_file = output_dir / f"llm_call_{call_num}.json"
        with open(output_file, "w") as f:
            json.dump(log_entry, f, indent=2)

        logger.info(f"LLM call logged to {output_file}")


class OpenAIModel(LLMModel):
    """OpenAI-compatible model implementation."""

    def __init__(
        self,
        model_name: str = "gpt-4o-2024-11-20",
        api_key: str | None = None,
        temperature: float = 0,
        max_retries: int = 5,
        base_url: str | None = None,
    ):
        self.model_name = model_name
        self.api_key = (
            api_key or os.getenv("CHATGPT_API_KEY") or os.getenv("OPENAI_API_KEY")
        )
        self.temperature = temperature
        self.max_retries = max_retries
        self.log_llm_calls = False
        self._log_timestamp = None
        self._session_id = None

        client_kwargs = {"api_key": self.api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = openai.OpenAI(**client_kwargs)
        self._async_client = openai.AsyncOpenAI(**client_kwargs)

    def get_name(self) -> str:
        return self.model_name

    def call(
        self,
        messages: list[dict],
        tools: list[Tool] | None = None,
        call_number: int | None = None,
    ) -> ModelResponse:
        """Call the OpenAI API."""
        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
        }
        if tools:
            kwargs["tools"] = [t.schema() for t in tools]
            kwargs["tool_choice"] = "auto"

        for attempt in range(self.max_retries):
            try:
                response = self._client.chat.completions.create(**kwargs)
                parsed_response = self._parse_response(response)
                self._log_llm_call(kwargs, parsed_response, call_number)
                return parsed_response
            except Exception as e:
                logger.error(f"LLM call failed (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2**attempt)
                else:
                    raise

    async def call_async(
        self,
        messages: list[dict],
        tools: list[Tool] | None = None,
        call_number: int | None = None,
    ) -> ModelResponse:
        """Async call to the OpenAI API."""
        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
        }
        if tools:
            kwargs["tools"] = [t.schema() for t in tools]
            kwargs["tool_choice"] = "auto"

        for attempt in range(self.max_retries):
            try:
                response = await self._async_client.chat.completions.create(**kwargs)
                parsed_response = self._parse_response(response)
                self._log_llm_call(kwargs, parsed_response, call_number)
                return parsed_response
            except Exception as e:
                logger.error(f"LLM call failed (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2**attempt)
                else:
                    raise

    def _parse_response(self, response) -> ModelResponse:
        """Parse OpenAI response into ModelResponse."""
        choice = response.choices[0]
        message = choice.message

        tool_calls = None
        if message.tool_calls:
            tool_calls = [
                ToolCallInfo(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=tc.function.arguments,
                )
                for tc in message.tool_calls
            ]

        return ModelResponse(
            content=message.content,
            tool_calls=tool_calls,
            raw_response=response,
        )


class AnthropicModel(LLMModel):
    """Anthropic Claude model implementation."""

    def __init__(
        self,
        model_name: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        temperature: float = 0,
        max_retries: int = 5,
        max_tokens: int = 4096,
    ):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.temperature = temperature
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self.log_llm_calls = False
        self._log_timestamp = None
        self._session_id = None

        try:
            import anthropic

            self._client = anthropic.Anthropic(api_key=self.api_key)
            self._async_client = anthropic.AsyncAnthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("anthropic package required: pip install anthropic")

    def get_name(self) -> str:
        return self.model_name

    def _prepare_kwargs(
        self, messages: list[dict], tools: list[Tool] | None
    ) -> tuple[dict, str | None]:
        """Prepare kwargs for Anthropic API call."""
        system_content = None
        filtered_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            else:
                filtered_messages.append(msg)

        kwargs = {
            "model": self.model_name,
            "messages": filtered_messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if system_content:
            kwargs["system"] = system_content

        if tools:
            kwargs["tools"] = [self._convert_tool(t) for t in tools]

        return kwargs

    def call(
        self,
        messages: list[dict],
        tools: list[Tool] | None = None,
        call_number: int | None = None,
    ) -> ModelResponse:
        """Call the Anthropic API."""
        kwargs = self._prepare_kwargs(messages, tools)

        for attempt in range(self.max_retries):
            try:
                response = self._client.messages.create(**kwargs)
                parsed_response = self._parse_response(response)
                self._log_llm_call(kwargs, parsed_response, call_number)
                return parsed_response
            except Exception as e:
                logger.error(f"Anthropic call failed (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2**attempt)
                else:
                    raise

    async def call_async(
        self,
        messages: list[dict],
        tools: list[Tool] | None = None,
        call_number: int | None = None,
    ) -> ModelResponse:
        """Async call to the Anthropic API."""
        kwargs = self._prepare_kwargs(messages, tools)

        for attempt in range(self.max_retries):
            try:
                response = await self._async_client.messages.create(**kwargs)
                parsed_response = self._parse_response(response)
                self._log_llm_call(kwargs, parsed_response, call_number)
                return parsed_response
            except Exception as e:
                logger.error(f"Anthropic call failed (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2**attempt)
                else:
                    raise

    def _convert_tool(self, tool: Tool) -> dict:
        """Convert Tool to Anthropic tool format."""
        return {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.parameters,
        }

    def _parse_response(self, response) -> ModelResponse:
        """Parse Anthropic response into ModelResponse."""
        content = None
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                content = block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCallInfo(
                        id=block.id,
                        name=block.name,
                        arguments=json.dumps(block.input),
                    )
                )

        return ModelResponse(
            content=content,
            tool_calls=tool_calls if tool_calls else None,
            raw_response=response,
        )


if __name__ == "__main__":
    model = OpenAIModel()
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2 + 2?"},
    ]
    response = model.call(messages)
    print(f"Response: {response.content}")
