import os
import json
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


@dataclass(frozen=True)
class MCPServerSettings:
    name: str
    url: str
    description: str
    token: str | None


@dataclass(frozen=True)
class Settings:
    model: str
    api_key: str
    base_url: str
    workspace_root: Path
    history_db_path: Path
    mcp_servers: tuple[MCPServerSettings, ...]

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        api_key = os.getenv("AI_API_KEY")
        if not api_key:
            raise RuntimeError("AI_API_KEY is not set")
        default_workspace = Path(__file__).resolve().parents[3]
        workspace_root = Path(
            os.getenv("AI_HARNESS_WORKSPACE", str(default_workspace))
        ).expanduser().resolve()
        history_db_path = Path(
            os.getenv(
                "AI_HARNESS_DB_PATH",
                str(Path.home() / ".ai-harness" / "history.sqlite3"),
            )
        ).expanduser().resolve()
        server_configs = []

        github_token = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
        github_url = os.getenv(
            "GITHUB_MCP_URL",
            "https://api.githubcopilot.com/mcp/?toolsets=issues,repos",
        )
        if github_token and github_url:
            server_configs.append(
                MCPServerSettings(
                    name="github",
                    url=github_url,
                    description="GitHub repositories, source files, issues, and pull requests",
                    token=github_token,
                )
            )

        for name, variable, description in (
            (
                "filesystem",
                "FILESYSTEM_MCP_URL",
                "Filesystem and project file operations",
            ),
            ("terminal", "TERMINAL_MCP_URL", "Terminal and command execution"),
        ):
            url = os.getenv(variable)
            if url:
                server_configs.append(
                    MCPServerSettings(
                        name=name,
                        url=url,
                        description=description,
                        token=os.getenv(f"{name.upper()}_MCP_TOKEN"),
                    )
                )

        raw_servers = os.getenv("MCP_SERVERS_JSON", "[]")
        try:
            extra_servers = json.loads(raw_servers)
        except json.JSONDecodeError as exc:
            raise RuntimeError("MCP_SERVERS_JSON must contain a JSON array") from exc
        if not isinstance(extra_servers, list):
            raise RuntimeError("MCP_SERVERS_JSON must contain a JSON array")
        for item in extra_servers:
            if not isinstance(item, dict) or not item.get("name") or not item.get("url"):
                raise RuntimeError("Each MCP server needs name and url fields")
            token_env = item.get("token_env")
            server_configs.append(
                MCPServerSettings(
                    name=str(item["name"]),
                    url=str(item["url"]),
                    description=str(item.get("description", "")),
                    token=os.getenv(token_env) if token_env else None,
                )
            )
        names = [server.name for server in server_configs]
        if len(set(names)) != len(names):
            raise RuntimeError("MCP server names must be unique")

        return cls(
            model=os.getenv("LLM_MODEL", "deepseek/deepseek-v4.1-flash:free"),
            api_key=api_key,
            base_url=os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1"),
            workspace_root=workspace_root,
            history_db_path=history_db_path,
            mcp_servers=tuple(server_configs),
        )

    def create_chat_model(self) -> ChatOpenAI:
        return ChatOpenAI(
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
        )