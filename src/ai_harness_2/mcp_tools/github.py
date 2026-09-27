from __future__ import annotations

import base64
import json
from typing import Any, TYPE_CHECKING
from urllib.parse import quote

import httpx

if TYPE_CHECKING:
    from fastmcp import Client


def create_github_client(url: str, token: str) -> Client:
    from fastmcp import Client
    from fastmcp.client.auth import BearerAuth

    return Client(url, auth=BearerAuth(token))


async def call_github_tool(
    client: Any,
    name: str,
    arguments: dict[str, Any],
    token: str,
) -> Any:
    try:
        result = await client.call_tool(name, arguments)
        is_error = (
            result.get("is_error", False)
            if isinstance(result, dict)
            else getattr(result, "is_error", False)
        )
        if name != "get_file_contents" or not is_error:
            return result
        failure = "GitHub MCP returned an error for get_file_contents"
    except Exception as exc:
        if name != "get_file_contents":
            raise
        failure = str(exc)

    try:
        text = await get_file_contents_via_rest(arguments, token)
        return {"content": [{"type": "text", "text": text}]}
    except Exception as exc:
        message = f"MCP file read failed ({failure}); GitHub API fallback failed ({exc})"
        return {
            "content": [{"type": "text", "text": message}],
            "is_error": True,
        }


async def get_file_contents_via_rest(
    arguments: dict[str, Any], token: str
) -> str:
    owner = arguments["owner"]
    repo = arguments["repo"]
    path = str(arguments.get("path", "/")).strip("/")
    encoded_path = quote(path, safe="/")
    url = f"https://api.github.com/repos/{quote(owner)}/{quote(repo)}/contents/{encoded_path}"
    ref = arguments.get("ref") or arguments.get("sha")
    params = {"ref": ref} if ref else None
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    async with httpx.AsyncClient(timeout=30, headers=headers) as session:
        response = await session.get(url, params=params)
        response.raise_for_status()
        payload = response.json()

    if isinstance(payload, list):
        return "\n".join(
            f"{item.get('type', 'unknown')} {item.get('path', item.get('name', ''))}"
            for item in payload
        ) or "(empty directory)"

    if payload.get("type") == "file" and payload.get("content"):
        content = payload["content"]
        if payload.get("encoding") == "base64":
            return base64.b64decode(content).decode("utf-8", errors="replace")
        return str(content)

    return json.dumps(payload, ensure_ascii=True)
