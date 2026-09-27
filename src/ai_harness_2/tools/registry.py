from dataclasses import dataclass
from typing import Any, Awaitable, Callable


ToolInvoker = Callable[[dict[str, Any]], Awaitable[Any]]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    invoke: ToolInvoker


def to_openai_tools(mcp_tools: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": getattr(tool, "description", "") or "",
                "parameters": getattr(tool, "parameters", None)
                or getattr(tool, "inputSchema", {"type": "object"}),
            },
        }
        for tool in mcp_tools
    ]


def result_to_text(result: Any, limit: int = 30_000) -> str:
    blocks = (
        result.get("content", result)
        if isinstance(result, dict)
        else getattr(result, "content", result)
    ) or []
    if isinstance(blocks, str):
        text = blocks
    else:
        text_blocks = []
        for block in blocks:
            block_text = block if isinstance(block, str) else getattr(block, "text", None)
            resource = (
                block.get("resource")
                if isinstance(block, dict)
                else getattr(block, "resource", None)
            )
            if not block_text and resource:
                block_text = (
                    resource.get("text") or resource.get("uri")
                    if isinstance(resource, dict)
                    else getattr(resource, "text", None)
                    or getattr(resource, "uri", None)
                )
            if block_text:
                text_blocks.append(str(block_text))
        text = "\n".join(text_blocks) or "(no output)"
    is_error = (
        result.get("is_error", False)
        if isinstance(result, dict)
        else getattr(result, "is_error", False)
    )
    if is_error:
        text = f"TOOL ERROR: {text}"
    if len(text) > limit:
        text = text[:limit] + "...[truncated]"
    return text