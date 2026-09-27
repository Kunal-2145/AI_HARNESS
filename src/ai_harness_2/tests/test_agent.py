import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from ai_harness_2.agents.coding_agent import run_coding_agent
from ai_harness_2.tools.registry import ToolSpec


class FakeBoundModel:
    def __init__(self) -> None:
        self.messages = None
        self.responses = [
            SimpleNamespace(
                content="",
                tool_calls=[
                    {
                        "name": "github__create_pull_request",
                        "args": {"title": "Test"},
                        "id": "call-1",
                    }
                ],
            ),
            SimpleNamespace(content="Pull request created.", tool_calls=[]),
        ]

    async def ainvoke(self, messages):
        self.messages = messages
        return self.responses.pop(0)


class FakeLlm:
    def __init__(self) -> None:
        self.bound_model = FakeBoundModel()

    def bind_tools(self, _tools):
        return self.bound_model


class AgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_mutating_mcp_tool_runs_without_interactive_approval(self) -> None:
        invoke = AsyncMock(return_value="created")
        tool = ToolSpec(
            name="github__create_pull_request",
            description="Create a pull request",
            parameters={"type": "object"},
            invoke=invoke,
        )

        llm = FakeLlm()
        answer = await run_coding_agent(
            llm=llm,
            tools=[tool],
            question="Open a pull request",
            plan=("Create the requested pull request",),
            history=[
                {"role": "user", "content": "Use the requested title"},
                {"role": "assistant", "content": "I will use that title."},
            ],
        )

        self.assertEqual(answer, "Pull request created.")
        invoke.assert_awaited_once_with({"title": "Test"})
        self.assertEqual(
            [message.content for message in llm.bound_model.messages[2:5]],
            [
                "Use the requested title",
                "I will use that title.",
                "Open a pull request",
            ],
        )

    async def test_workspace_write_forces_test_run_and_reports_result(self) -> None:
        write_tool = ToolSpec(
            name="workspace_write_file",
            description="Write a file",
            parameters={"type": "object"},
            invoke=AsyncMock(return_value="Wrote app.py (12 characters)."),
        )
        terminal_invoke = AsyncMock(return_value="exit_code=0\n2 passed")
        terminal_tool = ToolSpec(
            name="terminal_execute",
            description="Run a command",
            parameters={"type": "object"},
            invoke=terminal_invoke,
        )

        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            (workspace / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
            (workspace / "uv.lock").write_text("version = 1\n", encoding="utf-8")
            (workspace / "tests").mkdir()
            (workspace / "tests" / "test_app.py").write_text("", encoding="utf-8")

            llm = FakeLlm()
            llm.bound_model.responses = [
                SimpleNamespace(
                    content="",
                    tool_calls=[
                        {
                            "name": "workspace_write_file",
                            "args": {"path": "app.py", "content": "print('ok')"},
                            "id": "write-1",
                        }
                    ],
                ),
                SimpleNamespace(content="Implemented the change.", tool_calls=[]),
            ]
            answer = await run_coding_agent(
                llm=llm,
                tools=[write_tool, terminal_tool],
                question="Update the app",
                plan=("Edit the app",),
                workspace_root=workspace,
            )

        self.assertEqual(terminal_invoke.await_count, 1)
        self.assertEqual(
            terminal_invoke.await_args.args[0]["command"],
            ["uv", "run", "--frozen", "python", "-m", "pytest", "-q"],
        )
        self.assertIn("Automatic verification command", answer)
        self.assertIn("2 passed", answer)

    async def test_failed_verification_is_explicitly_reported(self) -> None:
        write_tool = ToolSpec(
            name="workspace_write_file",
            description="Write a file",
            parameters={"type": "object"},
            invoke=AsyncMock(return_value="Wrote app.py (12 characters)."),
        )
        terminal_tool = ToolSpec(
            name="terminal_execute",
            description="Run a command",
            parameters={"type": "object"},
            invoke=AsyncMock(return_value="exit_code=1\n1 failed"),
        )
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            (workspace / "tests").mkdir()
            (workspace / "tests" / "test_app.py").write_text("", encoding="utf-8")
            llm = FakeLlm()
            llm.bound_model.responses = [
                SimpleNamespace(
                    content="",
                    tool_calls=[
                        {
                            "name": "workspace_write_file",
                            "args": {"path": "app.py", "content": "broken"},
                            "id": "write-1",
                        }
                    ],
                ),
                SimpleNamespace(content="Implemented the change.", tool_calls=[]),
            ]
            answer = await run_coding_agent(
                llm=llm,
                tools=[write_tool, terminal_tool],
                question="Update the app",
                plan=("Edit the app",),
                workspace_root=workspace,
            )

        self.assertIn("Automatic verification: FAILED", answer)
        self.assertIn("1 failed", answer)


if __name__ == "__main__":
    unittest.main()
