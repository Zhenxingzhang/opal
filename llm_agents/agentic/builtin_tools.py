"""Built-in example tools for testing the agent environment."""

import fitz

from llm_agents.agentic.tool import Tool


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


def _lookup(query: str) -> str:
    """Stub lookup tool — replace with your own retrieval logic."""
    return f"No results found for: {query}"


def _search_web(query: str, num_results: int = 5) -> str:
    """Search the web using DuckDuckGo.

    Args:
        query: Search query string.
        num_results: Number of results to return (default 5).

    Returns:
        Search results as formatted text.
    """
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        return "Error: duckduckgo_search package required. Install with: pip install duckduckgo-search"

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

LOOKUP_TOOL = Tool(
    name="lookup",
    description="Look up information by query. Returns relevant text.",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query.",
            }
        },
        "required": ["query"],
    },
    function=_lookup,
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
