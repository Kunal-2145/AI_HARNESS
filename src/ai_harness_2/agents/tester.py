import os
from pathlib import Path
from typing import Any

from ai_harness_2.tools.registry import result_to_text
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

IGNORED_TEST_DIRECTORIES = {
    ".git",
    ".venv",
    ".venv-1",
    "venv",
    ".pytest_cache",
    ".mypy_cache",
    ".tox",
    ".nox",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
}


def find_python_test_command(workspace: Path) -> list[str] | None:
    root = workspace.resolve()
    has_python_tests = False
    for current, directories, filenames in os.walk(root):
        directories[:] = [
            directory
            for directory in directories
            if directory not in IGNORED_TEST_DIRECTORIES
        ]
        if any(
            filename.endswith(".py")
            and (filename.startswith("test_") or filename.endswith("_test.py"))
            for filename in filenames
        ):
            has_python_tests = True
            break

    if not has_python_tests:
        return None
    if (root / "pyproject.toml").is_file() and (root / "uv.lock").is_file():
        return ["uv", "run", "--frozen", "python", "-m", "pytest", "-q"]
    return ["python", "-m", "pytest", "-q"]


async def verify_python_tests(workspace: Path, terminal_tool: Any) -> str:
    command = find_python_test_command(workspace)
    if command is None:
        return "Automatic verification: no Python test files found; tests were not run."
    output = await terminal_tool.invoke({"command": command, "cwd": "."})
    return (
        f"Automatic verification command: {' '.join(command)}\n"
        f"{result_to_text(output)}"
    )


async def suggest_tests(llm: ChatOpenAI, change_description: str) -> str:
    response = await llm.ainvoke(
        [
            SystemMessage(
                content="Propose focused automated tests for the described code change. "
                "Prioritize behavior, edge cases, and failure handling."
            ),
            HumanMessage(content=change_description),
        ]
    )
    return str(response.content)