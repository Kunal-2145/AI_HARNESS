import logging
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from ai_harness_2.tools.registry import result_to_text, to_openai_tools
from ai_harness_2.agents.tester import verify_python_tests

logger = logging.getLogger(__name__)


async def run_coding_agent(
    llm: ChatOpenAI,
    tools: list[Any],
    question: str,
    plan: tuple[str, ...],
    history: list[dict[str, str]] | None = None,
    workspace_root: Path | None = None,
) -> str:
    llm_with_tools = llm.bind_tools(to_openai_tools(tools))
    tools_by_name = {tool.name: tool for tool in tools}
    messages = [
        SystemMessage(
            content="You are the execution agent in a general AI harness. Follow the "
            "provided plan, choose only tools that are actually available, inspect "
            "results before proceeding, and do not claim an action succeeded unless "
            "the tool confirms it. Ask a focused question if a required detail is "
            "missing. Respect tool denials and safety errors."
        ),
        SystemMessage(content=f"Proposed task plan:\n{chr(10).join(f'- {step}' for step in plan)}"),
    ]
    for item in history or []:
        message_type = AIMessage if item["role"] == "assistant" else HumanMessage
        messages.append(message_type(content=item["content"]))
    messages.append(HumanMessage(content=question))
    workspace_changed = False

    while True:
        response = await llm_with_tools.ainvoke(messages)
        messages.append(response)
        if not response.tool_calls:
            answer = str(response.content)
            if workspace_changed:
                terminal_tool = tools_by_name.get("terminal_execute")
                if terminal_tool is None:
                    verification = "Automatic verification could not run: terminal tool unavailable."
                else:
                    try:
                        verification = await verify_python_tests(
                            workspace_root or Path.cwd(), terminal_tool
                        )
                    except Exception as exc:
                        verification = f"Automatic verification failed to start: {exc}"
                return f"{answer}\n\n{verification}"
            return answer

        for tool_call in response.tool_calls:
            logger.info(
                "Calling registered tool %s",
                tool_call["name"],
            )
            try:
                tool = tools_by_name.get(tool_call["name"])
                if tool is None:
                    raise ValueError("The model requested a tool that is not registered")
                result = await tool.invoke(tool_call["args"])
                content = result_to_text(result)
                if (
                    tool_call["name"] == "workspace_write_file"
                    and not content.startswith("TOOL ERROR:")
                ):
                    workspace_changed = True
            except Exception as exc:
                logger.warning("Tool %s failed: %s", tool_call["name"], exc)
                content = f"TOOL ERROR: {exc}"
            messages.append(
                ToolMessage(content=content, tool_call_id=tool_call["id"])
            )
