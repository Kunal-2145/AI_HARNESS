import os
import re
from dataclasses import dataclass
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


@dataclass(frozen=True)
class VerificationResult:
    command: tuple[str, ...] | None
    passed: bool | None
    output: str

    def as_text(self) -> str:
        if self.command is None:
            return f"Automatic verification: {self.output}"
        status = "PASSED" if self.passed else "FAILED"
        return (
            f"Automatic verification: {status}\n"
            f"Automatic verification command: {' '.join(self.command)}\n"
            f"{self.output}"
        )


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


async def verify_python_tests(
    workspace: Path, terminal_tool: Any
) -> VerificationResult:
    command = find_python_test_command(workspace)
    if command is None:
        return VerificationResult(None, None, "no Python test files found; tests were not run.")
    output = await terminal_tool.invoke({"command": command, "cwd": "."})
    text = result_to_text(output)
    exit_code = re.search(r"(?:^|\n)exit_code=(-?\d+)", text)
    passed = bool(exit_code and exit_code.group(1) == "0")
    return VerificationResult(tuple(command), passed, text)


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