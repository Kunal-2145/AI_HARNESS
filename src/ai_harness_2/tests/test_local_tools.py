import sys
import tempfile
import unittest
from pathlib import Path

from ai_harness_2.mcp_tools.filesystem import create_filesystem_tools
from ai_harness_2.mcp_tools.terminal import create_terminal_tools


class LocalToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_filesystem_reads_in_workspace_and_rejects_escape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "sample.py").write_text("print('ok')", encoding="utf-8")
            tools = {tool.name: tool for tool in create_filesystem_tools(root)}

            content = await tools["workspace_read_file"].invoke({"path": "sample.py"})
            self.assertEqual(content, "print('ok')")
            with self.assertRaisesRegex(ValueError, "inside the configured workspace"):
                await tools["workspace_read_file"].invoke({"path": "../outside.txt"})

    async def test_workspace_search_returns_paths_and_line_numbers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "src").mkdir()
            (root / "src" / "agent.py").write_text(
                "class Agent:\n    def run(self):  # Agent entry point\n        return 'Ready'\n",
                encoding="utf-8",
            )
            (root / "notes.txt").write_text("agent notes\n", encoding="utf-8")
            tools = {tool.name: tool for tool in create_filesystem_tools(root)}

            result = await tools["workspace_search"].invoke(
                {"query": "AGENT", "include": "*.py"}
            )

        self.assertIn("src/agent.py:1: class Agent:", result)
        self.assertIn("src/agent.py:2: def run(self):  # Agent entry point", result)
        self.assertNotIn("notes.txt", result)

    async def test_workspace_search_narrows_to_path_and_obeys_case_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "src").mkdir()
            (root / "docs").mkdir()
            (root / "src" / "app.py").write_text("needle\n", encoding="utf-8")
            (root / "docs" / "guide.md").write_text("NEEDLE\n", encoding="utf-8")
            tools = {tool.name: tool for tool in create_filesystem_tools(root)}

            result = await tools["workspace_search"].invoke(
                {
                    "query": "needle",
                    "path": "docs",
                    "case_sensitive": True,
                }
            )

        self.assertEqual(result, "No matches found for 'needle' in docs.")

    async def test_workspace_search_hides_credentials_and_rejects_escape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / ".env.local").write_text("needle=secret\n", encoding="utf-8")
            (root / ".git").mkdir()
            (root / ".git" / "config").write_text("needle\n", encoding="utf-8")
            tools = {tool.name: tool for tool in create_filesystem_tools(root)}

            result = await tools["workspace_search"].invoke({"query": "needle"})
            with self.assertRaisesRegex(ValueError, "inside the configured workspace"):
                await tools["workspace_search"].invoke(
                    {"query": "needle", "path": "../outside"}
                )

        self.assertIn("No matches found", result)

    async def test_workspace_search_caps_number_of_results(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "first.txt").write_text("match\nmatch\nmatch\n", encoding="utf-8")
            tools = {tool.name: tool for tool in create_filesystem_tools(root)}

            result = await tools["workspace_search"].invoke(
                {"query": "match", "max_results": 2}
            )

        self.assertEqual(result.count(": match"), 2)
        self.assertIn("Stopped at the 2-result limit", result)

    async def test_workspace_search_snippet_keeps_late_match_visible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "long.py").write_text(
                f"{'x' * 900} unique_symbol\n",
                encoding="utf-8",
            )
            tools = {tool.name: tool for tool in create_filesystem_tools(root)}

            result = await tools["workspace_search"].invoke(
                {"query": "unique_symbol"}
            )

        self.assertIn("unique_symbol", result)
        self.assertIn("...", result)

    async def test_filesystem_write_runs_without_interactive_approval(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tools = {tool.name: tool for tool in create_filesystem_tools(root)}
            result = await tools["workspace_write_file"].invoke(
                {"path": "new.py", "content": "print('new')"}
            )

            self.assertIn("Wrote new.py", result)
            self.assertEqual((root / "new.py").read_text(), "print('new')")

    async def test_terminal_executes_argv_without_interactive_approval(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tool = create_terminal_tools(Path(temporary))[0]
            result = await tool.invoke(
                {"command": ["python", "-c", "print('terminal-ok')"]}
            )

            self.assertIn("exit_code=0", result)
            self.assertIn("terminal-ok", result)

    async def test_terminal_rejects_shell_and_destructive_git_commands(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tool = create_terminal_tools(Path(temporary))[0]
            with self.assertRaisesRegex(ValueError, "not allowed"):
                await tool.invoke({"command": ["sh", "-c", "echo unsafe"]})
            with self.assertRaisesRegex(ValueError, "executable name"):
                await tool.invoke({"command": [sys.executable, "-c", "print('unsafe')"]})
            with self.assertRaisesRegex(ValueError, "blocked"):
                await tool.invoke({"command": ["git", "push"]})


if __name__ == "__main__":
    unittest.main()
