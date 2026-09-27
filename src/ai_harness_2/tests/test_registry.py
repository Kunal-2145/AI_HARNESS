import unittest
from types import SimpleNamespace

from ai_harness_2.tools.registry import result_to_text, to_openai_tools


class RegistryTests(unittest.TestCase):
    def test_converts_mcp_tool_schema(self) -> None:
        tool = SimpleNamespace(
            name="list_issues",
            description="List repository issues",
            inputSchema={"type": "object", "properties": {}},
        )

        self.assertEqual(
            to_openai_tools([tool]),
            [
                {
                    "type": "function",
                    "function": {
                        "name": "list_issues",
                        "description": "List repository issues",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        )

    def test_formats_and_truncates_mcp_result(self) -> None:
        result = SimpleNamespace(
            content=[SimpleNamespace(text="abcdef")], is_error=False
        )

        self.assertEqual(result_to_text(result, limit=3), "abc...[truncated]")

    def test_extracts_embedded_resource_content(self) -> None:
        result = {
            "content": [
                {
                    "type": "resource",
                    "resource": {
                        "uri": "repo://owner/project/README.md",
                        "text": "Project overview",
                    },
                }
            ]
        }

        self.assertEqual(result_to_text(result), "Project overview")


if __name__ == "__main__":
    unittest.main()