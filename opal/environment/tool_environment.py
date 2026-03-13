"""Per-run tool execution environment and built-in tool definitions.

ToolEnvironment encapsulates tool execution and per-run resources (e.g. a
SemanticRetriever).  Moving ``execute_tool`` here keeps the Agent focused on
LLM interaction while making tool execution safe under concurrent async runs.

Built-in tool functions and their Tool descriptors (CALCULATOR_TOOL,
SEARCH_PDF_TOOL, READ_PDF_TOOL, SEARCH_WEB_TOOL) are co-located here so
that all tool-related logic lives in one module.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from typing import Any

import fitz

from opal.environment.tool import Tool
from opal.embedding.semantic_retriever import SemanticRetriever


# ---------------------------------------------------------------------------
# Built-in tool function implementations
# ---------------------------------------------------------------------------


def _read_pdf(path: str, page: int | None = None) -> str:
    """Read text from a PDF file.

    Args:
        path: Local file path to the PDF.
        page: 1-indexed page number. If None, reads all pages.
    """
    doc = fitz.open(path)
    if page is not None:
        if page < 1 or page > len(doc):
            return f"Error: page {page} out of range (1-{len(doc)})"
        text = doc[page - 1].get_text()
    else:
        text = "\n\n".join(p.get_text() for p in doc)
    doc.close()
    return text


def _calculator(expression: str) -> str:
    """Evaluate a math expression."""
    try:
        result = eval(expression, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def _search_pdf(query: str, top_k: int | None = None, env=None) -> str:
    """Search the loaded PDF for content relevant to the query using semantic retrieval.

    Args:
        query: The search query.
        top_k: Number of results to return. If not provided, uses the
            configured ``retriever_top_k`` from the ToolEnvironment.
        env: ToolEnvironment providing the per-run SemanticRetriever.
    """
    if env is None or env.retriever is None:
        return (
            "Error: no PDF document loaded. Provide a ToolEnvironment with a retriever."
        )

    if top_k is None:
        top_k = env.retriever_top_k
    results = env.retriever.search(query, top_k=top_k)
    if not results:
        return "No relevant content found."

    output = []
    for i, r in enumerate(results, 1):
        output.append(f"[{i}] (score: {r.score:.3f})\n{r.text}")
    return "\n\n".join(output)


def _think(thought: str) -> str:
    """Use this tool to think about something. It will not obtain new information
    or change any state, but just record the thought. Use it when complex reasoning
    or synthesis of information is needed before taking an action."""
    return "Thought recorded."


def _search_web(query: str, num_results: int = 5) -> str:
    """Search the web using DuckDuckGo.

    Args:
        query: Search query string.
        num_results: Number of results to return (default 5).

    Returns:
        Search results as formatted text.
    """
    try:
        from ddgs import DDGS
    except ImportError:
        return "Error: ddgs package required. Install with: pip install ddgs"

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=num_results))

        if not results:
            return f"No results found for: {query}"

        output = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "No title")
            href = r.get("href", "No URL")
            body = r.get("body", "No description")
            output.append(f"{i}. {title}\n   URL: {href}\n   {body}")

        return "\n\n".join(output)
    except Exception as e:
        return f"Error searching web: {e}"


# ---------------------------------------------------------------------------
# Tool descriptors
# ---------------------------------------------------------------------------

CALCULATOR_TOOL = Tool(
    name="calculator",
    description="Evaluate a mathematical expression. Example: '2 + 3 * 4'",
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The math expression to evaluate.",
            }
        },
        "required": ["expression"],
    },
    function=_calculator,
)

SEARCH_PDF_TOOL = Tool(
    name="search_pdf",
    description="Search the loaded PDF document for content relevant to a query using semantic retrieval. Returns the most relevant text passages.",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query.",
            },
            "top_k": {
                "type": "integer",
                "description": "Number of results to return (default 5).",
            },
        },
        "required": ["query"],
    },
    function=_search_pdf,
)

READ_PDF_TOOL = Tool(
    name="read_pdf",
    description="Read text content from a local PDF file. Returns the extracted text as a string.",
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute path to the PDF file.",
            },
            "page": {
                "type": "integer",
                "description": "1-indexed page number to read. Omit to read all pages.",
            },
        },
        "required": ["path"],
    },
    function=_read_pdf,
)

THINK_TOOL = Tool(
    name="think",
    description="Use this tool to think about something. It will not obtain new information or change any state, but just record the thought. Use it when complex reasoning or synthesis of information is needed before taking an action.",
    parameters={
        "type": "object",
        "properties": {
            "thought": {
                "type": "string",
                "description": "Your thought or reasoning.",
            }
        },
        "required": ["thought"],
    },
    function=_think,
)

SEARCH_WEB_TOOL = Tool(
    name="search_web",
    description="Search the web for information using DuckDuckGo. Returns titles, URLs, and descriptions of search results.",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query.",
            },
            "num_results": {
                "type": "integer",
                "description": "Number of results to return (default 5, max 10).",
            },
        },
        "required": ["query"],
    },
    function=_search_web,
)


# ---------------------------------------------------------------------------
# ToolEnvironment
# ---------------------------------------------------------------------------


@dataclass
class ToolEnvironment:
    """Holds per-run resources and executes tool functions.

    Tool functions that need environment state should accept an ``env``
    keyword argument of this type.  ``execute_tool`` inspects the function
    signature and injects ``self`` automatically when the parameter is present.
    """

    retriever: SemanticRetriever | None = None
    retriever_top_k: int = 5
    tool_map: dict[str, Tool] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)

    async def execute_tool(self, name: str, arguments_json: str) -> str:
        """Execute a tool by name with JSON arguments.

        Supports both sync and async tool functions.  If the tool function
        accepts an ``env`` parameter the environment is passed through
        automatically.

        Args:
            name: Name of the tool to execute.
            arguments_json: JSON string of arguments.

        Returns:
            Tool execution result as a string.
        """
        tool = self.tool_map.get(name)
        if not tool:
            return f"Error: unknown tool '{name}'"
        try:
            args = json.loads(arguments_json)
            # If the tool function accepts an `env` parameter, inject it.
            sig = inspect.signature(tool.function)
            if "env" in sig.parameters:
                args["env"] = self
            result = tool.function(**args)
            if inspect.isawaitable(result):
                result = await result
            return str(result)
        except json.JSONDecodeError as e:
            return f"Error parsing arguments for {name}: {e}"
        except Exception as e:
            return f"Error executing {name}: {e}"
