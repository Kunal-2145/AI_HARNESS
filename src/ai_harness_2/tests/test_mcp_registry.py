import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from ai_harness_2.config.settings import MCPServerSettings
from ai_harness_2.mcp_tools.registry import discover_mcp_tools


class MCPRegistryTests(unittest.IsolatedAsyncioTestCase):
    async def test_no_selected_servers_skips_fastmcp_import(self) -> None:
        with patch(
            "builtins.__import__",
            side_effect=AssertionError("Unexpected MCP import"),
        ):
            async with discover_mcp_tools(()) as tools:
                self.assertEqual(tools, [])

    async def test_discovered_tools_are_namespaced_and_dispatch_to_server(self) -> None:
        client = AsyncMock()
        client.__aenter__.return_value = client
        client.list_tools.return_value = [
            SimpleNamespace(
                name="read_file",
                description="Read a remote file",
                inputSchema={"type": "object", "properties": {"path": {"type": "string"}}},
            )
        ]
        server = MCPServerSettings("docs", "https://mcp.example.test", "Docs", None)

        fake_fastmcp = types.ModuleType("fastmcp")
        fake_fastmcp.Client = lambda *_args, **_kwargs: client
        fake_fastmcp_client = types.ModuleType("fastmcp.client")
        fake_fastmcp_auth = types.ModuleType("fastmcp.client.auth")
        fake_fastmcp_auth.BearerAuth = lambda token: token
        with patch.dict(
            sys.modules,
            {
                "fastmcp": fake_fastmcp,
                "fastmcp.client": fake_fastmcp_client,
                "fastmcp.client.auth": fake_fastmcp_auth,
            },
        ):
            async with discover_mcp_tools((server,)) as tools:
                self.assertEqual(tools[0].name, "docs__read_file")
                await tools[0].invoke({"path": "README.md"})

        client.call_tool.assert_awaited_once_with("read_file", {"path": "README.md"})


if __name__ == "__main__":
    unittest.main()
