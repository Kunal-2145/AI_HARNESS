import unittest
from unittest.mock import AsyncMock, patch

from ai_harness_2.mcp_tools.github import call_github_tool


class FailingMcpClient:
    async def call_tool(self, _name, _arguments):
        raise ValueError("invalid MCP resource result")


class GitHubFallbackTests(unittest.IsolatedAsyncioTestCase):
    async def test_failed_file_read_uses_read_only_rest_fallback(self) -> None:
        arguments = {"owner": "owner", "repo": "repo", "path": "README.md"}

        with patch(
            "ai_harness_2.mcp_tools.github.get_file_contents_via_rest",
            new_callable=AsyncMock,
            return_value="README text",
        ) as fallback:
            result = await call_github_tool(
                FailingMcpClient(), "get_file_contents", arguments, "token"
            )

        fallback.assert_awaited_once_with(arguments, "token")
        self.assertEqual(result["content"][0]["text"], "README text")

    async def test_write_tool_failure_is_not_sent_to_rest_fallback(self) -> None:
        with (
            patch(
                "ai_harness_2.mcp_tools.github.get_file_contents_via_rest",
                new_callable=AsyncMock,
            ) as fallback,
            self.assertRaisesRegex(ValueError, "invalid MCP resource result"),
        ):
            await call_github_tool(
                FailingMcpClient(),
                "create_or_update_file",
                {"owner": "owner", "repo": "repo"},
                "token",
            )

        fallback.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()