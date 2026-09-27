import unittest
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from ai_harness_2.main import chat, run
from ai_harness_2.config.settings import MCPServerSettings
from ai_harness_2.orchestration.planner import TaskPlan


class CliTests(unittest.IsolatedAsyncioTestCase):
    async def test_chat_accepts_multiple_prompts_until_quit(self) -> None:
        runner = AsyncMock(side_effect=["Hello!", "Task complete"])
        with tempfile.TemporaryDirectory() as temporary_directory:
            settings = SimpleNamespace(
                workspace_root=Path(temporary_directory) / "workspace",
                history_db_path=Path(temporary_directory) / "history.sqlite3",
            )
            settings.workspace_root.mkdir()

            with (
                patch("builtins.input", side_effect=["hey", "run tests", "quit"]),
                patch("ai_harness_2.main.Settings.from_env", return_value=settings),
                patch("ai_harness_2.main.run", runner),
                patch("builtins.print"),
            ):
                await chat()

        self.assertEqual(
            [call.args[0] for call in runner.await_args_list],
            ["hey", "run tests"],
        )
        self.assertEqual(runner.await_args_list[0].kwargs["history"], [])
        self.assertEqual(
            runner.await_args_list[1].kwargs["history"],
            [
                {"role": "user", "content": "hey"},
                {"role": "assistant", "content": "Hello!"},
            ],
        )

    async def test_new_command_starts_without_prior_context(self) -> None:
        runner = AsyncMock(side_effect=["First answer", "Second answer"])
        with tempfile.TemporaryDirectory() as temporary_directory:
            settings = SimpleNamespace(
                workspace_root=Path(temporary_directory) / "workspace",
                history_db_path=Path(temporary_directory) / "history.sqlite3",
            )
            settings.workspace_root.mkdir()

            with (
                patch(
                    "builtins.input",
                    side_effect=["first topic", "/new", "different topic", "quit"],
                ),
                patch("ai_harness_2.main.Settings.from_env", return_value=settings),
                patch("ai_harness_2.main.run", runner),
                patch("builtins.print"),
            ):
                await chat()

        self.assertEqual(runner.await_args_list[0].kwargs["history"], [])
        self.assertEqual(runner.await_args_list[1].kwargs["history"], [])


class HarnessRunTests(unittest.IsolatedAsyncioTestCase):
    async def test_chat_route_does_not_discover_or_call_tools(self) -> None:
        model = SimpleNamespace(
            ainvoke=AsyncMock(return_value=SimpleNamespace(content="Hello there!"))
        )
        settings = SimpleNamespace(
            workspace_root=Path("."),
            mcp_servers=(),
            create_chat_model=lambda: model,
        )

        with (
            patch("ai_harness_2.main.Settings.from_env", return_value=settings),
            patch(
                "ai_harness_2.main.plan_request",
                new_callable=AsyncMock,
                return_value=TaskPlan("chat", (), ()),
            ),
            patch("ai_harness_2.main.discover_mcp_tools") as discover,
            patch("builtins.print"),
        ):
            answer = await run("hey")

        self.assertEqual(answer, "Hello there!")
        discover.assert_not_called()
        model.ainvoke.assert_awaited_once()

    async def test_chat_route_sends_previous_turns_to_planner_and_model(self) -> None:
        model = SimpleNamespace(
            ainvoke=AsyncMock(return_value=SimpleNamespace(content="The name is Ada."))
        )
        settings = SimpleNamespace(
            workspace_root=Path("."),
            mcp_servers=(),
            create_chat_model=lambda: model,
        )
        history = [
            {"role": "user", "content": "My name is Ada."},
            {"role": "assistant", "content": "Hello, Ada."},
        ]
        planner = AsyncMock(return_value=TaskPlan("chat", (), ()))

        with (
            patch("ai_harness_2.main.Settings.from_env", return_value=settings),
            patch("ai_harness_2.main.plan_request", planner),
            patch("ai_harness_2.main.discover_mcp_tools"),
            patch("builtins.print"),
        ):
            answer = await run("What is my name?", history=history)

        self.assertEqual(answer, "The name is Ada.")
        self.assertEqual(planner.await_args.kwargs["history"], history)
        model_messages = model.ainvoke.await_args.args[0]
        self.assertEqual([message.content for message in model_messages], [
            "My name is Ada.",
            "Hello, Ada.",
            "What is my name?",
        ])

    async def test_unsupported_request_returns_explanation_without_tools(self) -> None:
        settings = SimpleNamespace(
            workspace_root=Path("."),
            mcp_servers=(),
            create_chat_model=lambda: "fake-model",
        )
        explanation = "This request is outside the configured workspace scope."
        with (
            patch("ai_harness_2.main.Settings.from_env", return_value=settings),
            patch(
                "ai_harness_2.main.plan_request",
                new_callable=AsyncMock,
                return_value=TaskPlan("unsupported", (explanation,), ()),
            ),
            patch("ai_harness_2.main.discover_mcp_tools") as discover,
            patch("ai_harness_2.main.run_coding_agent") as agent,
            patch("builtins.print"),
        ):
            answer = await run("Create a file on an unconfigured external path")

        self.assertEqual(answer, explanation)
        discover.assert_not_called()
        agent.assert_not_called()

    async def test_task_route_discovers_only_selected_mcp_server(self) -> None:
        server = MCPServerSettings(
            "docs", "https://docs.example.test/mcp", "Documentation", None
        )
        other_server = MCPServerSettings(
            "github", "https://github.example.test/mcp", "Repositories", "token"
        )
        settings = SimpleNamespace(
            workspace_root=Path("."),
            mcp_servers=(server, other_server),
            create_chat_model=lambda: "fake-model",
        )
        selected_tool = SimpleNamespace(name="docs__search")
        discovered = []

        @asynccontextmanager
        async def fake_discover(servers):
            discovered.extend(servers)
            yield [selected_tool]

        agent = AsyncMock(return_value="Found the documentation.")
        history = [{"role": "user", "content": "This is a Python project."}]
        with (
            patch("ai_harness_2.main.Settings.from_env", return_value=settings),
            patch(
                "ai_harness_2.main.create_filesystem_tools", return_value=[]
            ),
            patch("ai_harness_2.main.create_terminal_tools", return_value=[]),
            patch(
                "ai_harness_2.main.plan_request",
                new_callable=AsyncMock,
                return_value=TaskPlan("task", ("Search the docs",), ("docs",)),
            ),
            patch("ai_harness_2.main.discover_mcp_tools", fake_discover),
            patch("ai_harness_2.main.run_coding_agent", agent),
            patch("builtins.print"),
        ):
            answer = await run("Find the docs for this setting", history=history)

        self.assertEqual(answer, "Found the documentation.")
        self.assertEqual(discovered, [server])
        agent.assert_awaited_once_with(
            llm="fake-model",
            tools=[selected_tool],
            question="Find the docs for this setting",
            plan=("Search the docs",),
            history=history,
            workspace_root=Path("."),
        )


if __name__ == "__main__":
    unittest.main()