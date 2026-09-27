from __future__ import annotations

import fnmatch
import os
from pathlib import Path
from typing import Any

from ai_harness_2.tools.registry import ToolSpec

MAX_FILE_BYTES = 500_000
MAX_SEARCH_FILE_BYTES = 1_000_000
MAX_SEARCH_RESULTS = 200
MAX_SEARCHED_FILES = 20_000
MAX_SEARCH_LINE_CHARS = 500
PROTECTED_SEARCH_DIRECTORIES = {
    ".git",
    ".venv",
    ".venv-1",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "node_modules",
    ".ai-harness",
}


def _resolve_workspace_path(workspace: Path, requested: str) -> Path:
    root = workspace.resolve()
    candidate = (root / requested).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError("Path must stay inside the configured workspace")
    if any(
        part in {".git", ".env", ".venv", ".venv-1"}
        for part in candidate.relative_to(root).parts
    ):
        raise ValueError("Access to protected workspace paths is not allowed")
    return candidate


def create_filesystem_tools(workspace: Path) -> list[ToolSpec]:
    async def list_directory(arguments: dict[str, Any]) -> str:
        requested = arguments.get("path", ".")
        directory = _resolve_workspace_path(workspace, requested)
        if not directory.is_dir():
            raise ValueError(f"Not a directory: {requested}")
        entries = []
        for item in sorted(directory.iterdir()):
            if item.name in {".git", ".env", ".venv", ".venv-1"}:
                continue
            kind = "dir" if item.is_dir() else "file"
            entries.append(f"{kind}\t{item.relative_to(workspace.resolve())}")
        return "\n".join(entries) or "(empty directory)"

    async def read_file(arguments: dict[str, Any]) -> str:
        path = _resolve_workspace_path(workspace, arguments["path"])
        if not path.is_file():
            raise ValueError(f"Not a file: {arguments['path']}")
        if path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError(f"File exceeds the {MAX_FILE_BYTES}-byte read limit")
        return path.read_text(encoding="utf-8")

    async def search_workspace(arguments: dict[str, Any]) -> str:
        query = arguments.get("query")
        if not isinstance(query, str) or not query:
            raise ValueError("query must be a non-empty string")
        requested_path = arguments.get("path", ".")
        search_root = _resolve_workspace_path(workspace, requested_path)
        if not search_root.is_file() and not search_root.is_dir():
            raise ValueError(f"Not a file or directory: {requested_path}")

        include = arguments.get("include", "*")
        if not isinstance(include, str) or not include:
            raise ValueError("include must be a non-empty glob pattern")
        case_sensitive = arguments.get("case_sensitive", False)
        if not isinstance(case_sensitive, bool):
            raise ValueError("case_sensitive must be a boolean")
        max_results = arguments.get("max_results", 50)
        if not isinstance(max_results, int) or isinstance(max_results, bool):
            raise ValueError("max_results must be an integer")
        if not 1 <= max_results <= MAX_SEARCH_RESULTS:
            raise ValueError(f"max_results must be between 1 and {MAX_SEARCH_RESULTS}")

        root = workspace.resolve()
        query_value = query if case_sensitive else query.casefold()
        if search_root.is_file():
            candidate_files = [search_root]
            directory_walk_truncated = False
        else:
            candidate_files = []
            directory_walk_truncated = False
            for current, directories, filenames in os.walk(search_root):
                directories[:] = [
                    name
                    for name in directories
                    if name not in PROTECTED_SEARCH_DIRECTORIES
                    and not name.startswith(".env")
                ]
                for filename in filenames:
                    candidate_files.append(Path(current) / filename)
                    if len(candidate_files) >= MAX_SEARCHED_FILES:
                        directory_walk_truncated = True
                        break
                if directory_walk_truncated:
                    break

        matches: list[str] = []
        for path in candidate_files:
            if path.is_symlink() or path.name.startswith(".env"):
                continue
            relative = path.relative_to(root).as_posix()
            if not (
                fnmatch.fnmatchcase(relative, include)
                or fnmatch.fnmatchcase(path.name, include)
                or (
                    include.startswith("**/")
                    and fnmatch.fnmatchcase(relative, include[3:])
                )
            ):
                continue
            try:
                if path.stat().st_size > MAX_SEARCH_FILE_BYTES:
                    continue
                content = path.read_bytes()
            except (OSError, ValueError):
                continue
            if b"\0" in content:
                continue
            text = content.decode("utf-8", errors="replace")
            for line_number, line in enumerate(text.splitlines(), start=1):
                line_value = line if case_sensitive else line.casefold()
                if query_value not in line_value:
                    continue
                shown_line = line.strip()
                if len(shown_line) > MAX_SEARCH_LINE_CHARS:
                    match_index = line_value.find(query_value)
                    start = max(0, match_index - MAX_SEARCH_LINE_CHARS // 4)
                    end = min(len(line), start + MAX_SEARCH_LINE_CHARS)
                    shown_line = (
                        ("..." if start else "")
                        + line[start:end].strip()
                        + ("..." if end < len(line) else "")
                    )
                matches.append(f"{relative}:{line_number}: {shown_line}")
                if len(matches) >= max_results:
                    return "\n".join(matches) + f"\nStopped at the {max_results}-result limit."

        if not matches:
            result = f"No matches found for {query!r} in {requested_path}."
        else:
            result = "\n".join(matches)
        if directory_walk_truncated:
            result += f"\nSearch stopped after {MAX_SEARCHED_FILES} files."
        return result

    async def write_file(arguments: dict[str, Any]) -> str:
        path = _resolve_workspace_path(workspace, arguments["path"])
        relative = path.relative_to(workspace.resolve())
        content = arguments["content"]
        if len(content.encode("utf-8")) > MAX_FILE_BYTES:
            raise ValueError(f"File exceeds the {MAX_FILE_BYTES}-byte write limit")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"Wrote {relative} ({len(content)} characters)."

    return [
        ToolSpec(
            name="workspace_list_directory",
            description=f"List files and directories inside the configured workspace root ({workspace.resolve()}). Paths are relative to that root; protected paths are hidden.",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string", "default": "."}},
                "additionalProperties": False,
            },
            invoke=list_directory,
        ),
        ToolSpec(
            name="workspace_read_file",
            description=f"Read a UTF-8 text file inside the configured workspace root ({workspace.resolve()}), excluding secrets and environment folders. Paths are relative to that root.",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
            invoke=read_file,
        ),
        ToolSpec(
            name="workspace_search",
            description=(
                f"Search text inside files in the configured workspace ({workspace.resolve()}). "
                "Returns matching file paths, line numbers, and lines. Search is literal and "
                "case-insensitive by default; use path and include glob filters to narrow it. "
                "Credential files, generated environments, binary files, and large files are skipped."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "minLength": 1},
                    "path": {"type": "string", "default": "."},
                    "include": {"type": "string", "default": "*"},
                    "case_sensitive": {"type": "boolean", "default": False},
                    "max_results": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAX_SEARCH_RESULTS,
                        "default": 50,
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            invoke=search_workspace,
        ),
        ToolSpec(
            name="workspace_write_file",
            description=f"Create or replace a UTF-8 text file inside the configured workspace root ({workspace.resolve()}). Paths are relative to that root; this tool cannot write outside that directory.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
            invoke=write_file,
        ),
    ]
