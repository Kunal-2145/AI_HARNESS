import unittest
from types import SimpleNamespace

from ai_harness_2.config.settings import MCPServerSettings
from ai_harness_2.orchestration.planner import plan_request
from ai_harness_2.tools.registry import ToolSpec


class FakeLlm:
    def __init__(self, response: str) -> None:
        self.response = response

    async def ainvoke(self, _messages):
        return SimpleNamespace(content=self.response)


class PlannerTests(unittest.IsolatedAsyncioTestCase):
    async def test_chat_plan_does_not_select_mcp_even_if_model_includes_one(self) -> None:
        server = MCPServerSettings("github", "https://example.test", "GitHub", "token")
        plan = await plan_request(
            FakeLlm('{"kind":"chat","steps":[],"server_names":["github"]}'),
            "hey",
            [],
            (server,),
        )

        self.assertEqual(plan.kind, "chat")
        self.assertEqual(plan.server_names, ())

    async def test_task_plan_filters_unconfigured_servers(self) -> None:
        tool = ToolSpec("workspace_read_file", "Read a file", {}, lambda _args: None)
        server = MCPServerSettings("github", "https://example.test", "Repositories", "token")
        plan = await plan_request(
            FakeLlm(
                '{"kind":"task","steps":["Inspect the project"],'
                '"server_names":["github","unknown"]}'
            ),
            "Inspect my repo",
            [tool],
            (server,),
        )

        self.assertEqual(plan.kind, "task")
        self.assertEqual(plan.steps, ("Inspect the project",))
        self.assertEqual(plan.server_names, ("github",))

    async def test_planner_can_mark_unavailable_capability_unsupported(self) -> None:
        plan = await plan_request(
            FakeLlm(
                '{"kind":"unsupported","steps":['
                '"This path is outside the configured workspace."],'
                '"server_names":[]}'
            ),
            "Create a file outside the workspace",
            [],
            (),
        )

        self.assertEqual(plan.kind, "unsupported")
        self.assertEqual(plan.server_names, ())
        self.assertIn("outside the configured workspace", plan.steps[0])


if __name__ == "__main__":
    unittest.main()
