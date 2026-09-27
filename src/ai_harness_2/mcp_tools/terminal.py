from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

from ai_harness_2.tools.registry import ToolSpec

ALLOWED_EXECUTABLES = {"python", "python3", "pytest", "uv", "git", "ruff"}
BLOCKED_GIT_COMMANDS = {
    "commit",
    "push",
    "reset",
    "clean",
    "checkout",
    "switch",
    "merge",
    "rebase",
    "cherry-pick",
    "restore",
}
MAX_OUTPUT_CHARS = 20_000
DEFAULT_TIMEOUT_SECONDS = 60


def create_terminal_tools(
    workspace: Path,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> list[ToolSpec]:
    root = workspace.resolve()

    async def execute(arguments: dict[str, Any]) -> str:
        command = arguments.get("command")
        if not isinstance(command, list) or not command or not all(
            isinstance(part, str) for part in command
        ):
            raise ValueError("command must be a non-empty array of strings")
        executable = Path(command[0]).name
        if executable != command[0]:
            raise ValueError("Use an allowlisted executable name, not an executable path")
        if executable not in ALLOWED_EXECUTABLES:
            raise ValueError(
                f"Executable {executable!r} is not allowed. Allowed: "
                f"{', '.join(sorted(ALLOWED_EXECUTABLES))}"
            )
        if executable == "git" and len(command) > 1 and command[1] in BLOCKED_GIT_COMMANDS:
            raise ValueError(f"git {command[1]} is blocked by the terminal safety policy")
        if executable == "git" and len(command) > 2 and command[1] == "branch":
            if "-D" in command[2:] or "--delete" in command[2:]:
                raise ValueError("Deleting Git branches is blocked by the terminal safety policy")

        requested_cwd = arguments.get("cwd", ".")
        cwd = (root / requested_cwd).resolve()
        if not cwd.is_relative_to(root) or not cwd.is_dir():
            raise ValueError("cwd must be an existing directory inside the workspace")
        child_env = {
            key: value
            for key, value in os.environ.items()
            if not any(secret in key.upper() for secret in ("TOKEN", "SECRET", "PASSWORD", "API_KEY"))
        }
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=cwd,
            env=child_env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            output, _ = await asyncio.wait_for(
                process.communicate(), timeout=timeout_seconds
            )
        except TimeoutError:
            process.kill()
            await process.wait()
            return f"Command timed out after {timeout_seconds} seconds."

        text = output.decode("utf-8", errors="replace")
        if len(text) > MAX_OUTPUT_CHARS:
            text = text[:MAX_OUTPUT_CHARS] + "...[output truncated]"
        return f"exit_code={process.returncode}\n{text}"

    return [
        ToolSpec(
            name="terminal_execute",
            description=(
                "Run an allowlisted command without a shell, with cwd inside the configured workspace. "
                "Use argv arrays such as ['python', '-m', 'pytest']. The cwd restriction is not an OS sandbox: "
                "a permitted Python process can access other paths allowed by the operating system. "
                "Git commit, push, reset, clean, checkout, and switch are blocked."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                    },
                    "cwd": {"type": "string", "default": "."},
                },
                "required": ["command"],
                "additionalProperties": False,
            },
            invoke=execute,
        )
    ]
