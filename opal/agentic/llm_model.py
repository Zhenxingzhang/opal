"""
Simplified LLM model abstraction.

Inspired by funbench's Model class but simplified for this codebase.

Usage:
    from source.llm_model import LLMModel, OpenAIModel

    model = OpenAIModel(model_name="gpt-4o")
    response = model.call(messages, tools)
"""

import asyncio
import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

from dotenv import load_dotenv
import openai

from opal.environment.tool import Tool

load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class ToolCallInfo:
    """Information about a tool call made by the model."""

    id: str
    name: str
    arguments: str  # JSON string of arguments


@dataclass(frozen=True)
class LLMCallMetrics:
    """Metrics captured from a single LLM API call."""

    call_number: int
    timestamp: str
    model_name: str
    ai_cache_hit: bool
    raw_request: dict
    raw_response: dict


@dataclass
class ModelResponse:
    """Response from an LLM model."""

    content: str | None
    tool_calls: list[ToolCallInfo] | None


class LLMModel(ABC):
    """Base class for language models."""

    @abstractmethod
    async def call(
        self,
        messages: list[dict],
        tools: list[Tool] | None = None,
        call_number: int = 0,
    ) -> tuple[ModelResponse, LLMCallMetrics]:
        """Call the model with messages and optional tools.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            tools: Optional list of Tool objects available for the model.
            call_number: Sequence number for this call within the session.

        Returns:
            Tuple of (ModelResponse, LLMCallMetrics).
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get the model name/identifier."""
        pass

    def _build_metrics(
        self, kwargs: dict, parsed_response: ModelResponse, call_number: int
    ) -> LLMCallMetrics:
        """Build an immutable LLMCallMetrics from call kwargs and parsed response."""
        tool_calls = (
            [
                {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                for tc in parsed_response.tool_calls
            ]
            if parsed_response.tool_calls
            else None
        )
        return LLMCallMetrics(
            call_number=call_number,
            timestamp=datetime.now().isoformat(),
            model_name=self.get_name(),
            ai_cache_hit=False,
            raw_request=kwargs,
            raw_response={
                "content": parsed_response.content,
                "tool_calls": tool_calls,
            },
        )


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
        client_kwargs = {"api_key": self.api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = openai.AsyncOpenAI(**client_kwargs)

    def get_name(self) -> str:
        return self.model_name

    async def call(
        self,
        messages: list[dict],
        tools: list[Tool] | None = None,
        call_number: int = 0,
    ) -> tuple[ModelResponse, LLMCallMetrics]:
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
                response = await self._client.chat.completions.create(**kwargs)
                parsed_response = self._parse_response(response)
                metrics = self._build_metrics(kwargs, parsed_response, call_number)
                return parsed_response, metrics
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

        try:
            import anthropic

            self._client = anthropic.AsyncAnthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("anthropic package required: pip install anthropic")

    def get_name(self) -> str:
        return self.model_name

    def _prepare_kwargs(self, messages: list[dict], tools: list[Tool] | None) -> dict:
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

    async def call(
        self,
        messages: list[dict],
        tools: list[Tool] | None = None,
        call_number: int = 0,
    ) -> tuple[ModelResponse, LLMCallMetrics]:
        """Call the Anthropic API."""
        kwargs = self._prepare_kwargs(messages, tools)

        for attempt in range(self.max_retries):
            try:
                response = await self._client.messages.create(**kwargs)
                parsed_response = self._parse_response(response)
                metrics = self._build_metrics(kwargs, parsed_response, call_number)
                return parsed_response, metrics
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
        )


if __name__ == "__main__":
    import asyncio

    async def _main():
        model = OpenAIModel()
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2 + 2?"},
        ]
        response, metrics = await model.call(messages)
        print(f"Response: {response.content}")
        print(f"Metrics: {metrics}")

    asyncio.run(_main())
