from __future__ import annotations

import re
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any, AsyncIterator

from ai_harness_2.config.settings import MCPServerSettings
from ai_harness_2.tools.registry import ToolSpec


def _tool_name(server_name: str, tool_name: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_-]", "_", f"{server_name}__{tool_name}")
    return normalized[:64]


@asynccontextmanager
async def discover_mcp_tools(
    servers: tuple[MCPServerSettings, ...],
) -> AsyncIterator[list[ToolSpec]]:
    if not servers:
        yield []
        return

    from fastmcp import Client
    from fastmcp.client.auth import BearerAuth

    async with AsyncExitStack() as stack:
        registered: list[ToolSpec] = []
        for server in servers:
            auth = BearerAuth(server.token) if server.token else None
            client = await stack.enter_async_context(Client(server.url, auth=auth))
            listed = await client.list_tools()
            mcp_tools = getattr(listed, "tools", listed)
            for mcp_tool in mcp_tools:
                name = _tool_name(server.name, mcp_tool.name)
                original_name = mcp_tool.name
                schema = getattr(mcp_tool, "inputSchema", None) or getattr(
                    mcp_tool, "input_schema", {"type": "object"}
                )

                async def invoke(
                    arguments: dict[str, Any],
                    *,
                    _client: Any = client,
                    _name: str = original_name,
                    _server: MCPServerSettings = server,
                ) -> Any:
                    if _server.name == "github" and _server.token:
                        from ai_harness_2.mcp_tools.github import call_github_tool

                        return await call_github_tool(
                            _client, _name, arguments, _server.token
                        )
                    return await _client.call_tool(_name, arguments)

                registered.append(
                    ToolSpec(
                        name=name,
                        description=(
                            f"MCP server {server.name}: "
                            f"{server.description}. Tool {original_name}: "
                            f"{getattr(mcp_tool, 'description', '') or 'No description provided.'}"
                        ),
                        parameters=schema,
                        invoke=invoke,
                    )
                )
        yield registered
