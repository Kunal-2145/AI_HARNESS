from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from ai_harness_2.config.settings import MCPServerSettings
from ai_harness_2.tools.registry import ToolSpec

MAX_PLANNER_ATTEMPTS = 2


@dataclass(frozen=True)
class TaskPlan:
    kind: str
    steps: tuple[str, ...]
    server_names: tuple[str, ...]


def _parse_json(content: Any) -> dict[str, Any]:
    text = str(content).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("Planner response must be a JSON object")
    return parsed


async def plan_request(
    llm: ChatOpenAI,
    question: str,
    local_tools: list[ToolSpec],
    servers: tuple[MCPServerSettings, ...],
    history: list[dict[str, str]] | None = None,
) -> TaskPlan:
    capabilities = {
        "local_tools": [
            {"name": tool.name, "description": tool.description}
            for tool in local_tools
        ],
        "mcp_servers": [
            {"name": server.name, "description": server.description}
            for server in servers
        ],
    }
    messages = [
        SystemMessage(
            content=(
                "You are the task router and planner for a general AI harness, not "
                "a provider-specific assistant. Inspect the user's intent and the "
                "available capabilities. For greetings, general knowledge, or "
                "conversation, choose kind=chat and no servers. For actionable work "
                "covered by listed tools or servers, choose kind=task, list only the "
                "MCP servers that are actually configured and relevant, and give a "
                "concise ordered plan. Tool descriptions are authoritative about scope. "
                "Do not infer access to paths, systems, or operations no listed tool "
                "supports. If no listed capability can perform the requested action, "
                "choose kind=unsupported, set server_names to [], and put a concise "
                "explanation and required configuration in steps. Never invent a "
                "capability or server. Return ONLY JSON with keys kind (chat, task, "
                "or unsupported), "
                "steps (array of strings), and server_names (array of configured names)."
            )
        )
    ]
    for item in history or []:
        message_type = AIMessage if item["role"] == "assistant" else HumanMessage
        messages.append(message_type(content=item["content"]))
    messages.append(
        HumanMessage(
            content=(
                f"Available capabilities:\n{json.dumps(capabilities)}\n\n"
                f"User request:\n{question}"
            )
        )
    )
    last_error: Exception | None = None
    data: dict[str, Any] | None = None
    for attempt in range(MAX_PLANNER_ATTEMPTS):
        try:
            response = await llm.ainvoke(messages)
            data = _parse_json(response.content)
            break
        except Exception as exc:
            last_error = exc
            if attempt + 1 == MAX_PLANNER_ATTEMPTS:
                break
            messages.append(
                SystemMessage(
                    content=(
                        "The previous planning response was invalid or unavailable. "
                        "Retry once and return only the required JSON object."
                    )
                )
            )
    if data is None:
        raise RuntimeError("Planner failed after retrying") from last_error
    kind = data.get("kind")
    if kind not in {"chat", "task", "unsupported"}:
        raise ValueError(f"Planner returned invalid kind: {kind!r}")
    steps = data.get("steps", [])
    server_names = data.get("server_names", [])
    if not isinstance(steps, list) or not all(
        isinstance(step, str) for step in steps
    ):
        raise ValueError("Planner steps must be an array of strings")
    if not isinstance(server_names, list) or not all(
        isinstance(name, str) for name in server_names
    ):
        raise ValueError("Planner server_names must be an array of strings")
    configured_names = {server.name for server in servers}
    selected_names = tuple(
        dict.fromkeys(name for name in server_names if name in configured_names)
    )
    if kind in {"chat", "unsupported"}:
        selected_names = ()
    return TaskPlan(kind, tuple(steps), selected_names)
