import asyncio
import sys
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

if __package__ in {None, ""}:
    package_dir = Path(__file__).resolve().parent
    source_dir = package_dir.parent
    sys.path[:] = [
        str(source_dir),
        *(
            entry
            for entry in sys.path
            if Path(entry or ".").resolve() != package_dir
        ),
    ]

from ai_harness_2.config.settings import Settings
from ai_harness_2.agents.coding_agent import run_coding_agent
from ai_harness_2.mcp_tools.filesystem import create_filesystem_tools
from ai_harness_2.mcp_tools.registry import discover_mcp_tools
from ai_harness_2.mcp_tools.terminal import create_terminal_tools
from ai_harness_2.orchestration.planner import plan_request
from ai_harness_2.orchestration.router import route_plan
from ai_harness_2.state.history import ConversationStore
from ai_harness_2.state.models import HarnessState


async def run(
    question: str,
    history: list[dict[str, str]] | None = None,
    state: HarnessState | None = None,
) -> str:
    settings = Settings.from_env()
    llm = settings.create_chat_model()
    local_tools = [
        *create_filesystem_tools(settings.workspace_root),
        *create_terminal_tools(settings.workspace_root),
    ]
    request_state = state if state is not None else {"question": question}
    plan = await plan_request(
        llm, question, local_tools, settings.mcp_servers, history=history
    )
    request_state.update(
        {
            "kind": plan.kind,
            "plan": list(plan.steps),
            "selected_servers": list(plan.server_names),
        }
    )
    print(f"[router] Decision: {route_plan(plan)}")
    if plan.server_names:
        print(f"[router] Selected MCP servers: {', '.join(plan.server_names)}")
    for index, step in enumerate(plan.steps, start=1):
        print(f"[plan {index}] {step}")

    if plan.kind == "chat":
        messages = []
        for item in history or []:
            message_type = AIMessage if item["role"] == "assistant" else HumanMessage
            messages.append(message_type(content=item["content"]))
        messages.append(HumanMessage(content=question))
        response = await llm.ainvoke(messages)
        request_state["answer"] = str(response.content)
        return request_state["answer"]
    if plan.kind == "unsupported":
        request_state["answer"] = "\n".join(plan.steps)
        return request_state["answer"]

    selected_servers = tuple(
        server for server in settings.mcp_servers if server.name in plan.server_names
    )
    try:
        async with discover_mcp_tools(selected_servers) as mcp_tools:
            available_tools = [*local_tools, *mcp_tools]
            request_state["answer"] = await run_coding_agent(
                llm=llm,
                tools=available_tools,
                question=question,
                plan=plan.steps,
                history=history,
                workspace_root=settings.workspace_root,
            )
    except Exception as exc:
        request_state["answer"] = (
            "Task could not start because the selected tool service failed: "
            f"{exc}. No task completion is being claimed."
        )
    return request_state["answer"]


def _execution_context(state: HarnessState) -> str:
    plan = state.get("plan", [])
    servers = state.get("selected_servers", [])
    steps = "\n".join(f"- {step}" for step in plan) or "(none)"
    selected = ", ".join(servers) or "none"
    return f"Execution plan:\n{steps}\nSelected MCP servers: {selected}"


async def chat() -> None:
    try:
        settings = Settings.from_env()
    except RuntimeError as exc:
        print(f"Configuration error: {exc}")
        print("Set AI_API_KEY in the environment before running make run.")
        return
    store = ConversationStore(settings.history_db_path)
    session_id, resumed = store.resume_or_create_session(settings.workspace_root)
    session_state = "Resumed" if resumed else "Started"
    print(
        f"AI harness ready. {session_state} session {session_id[:8]}. "
        "Type '/new' for a fresh session, or 'exit'/'quit' to finish."
    )
    while True:
        try:
            question = await asyncio.to_thread(input, "You> ")
        except EOFError:
            print()
            return
        question = question.strip()
        if question.lower() == "/new":
            session_id = store.create_session(settings.workspace_root)
            print(f"Started new session {session_id[:8]}.")
            continue
        if question.lower() in {"exit", "quit"}:
            return
        if not question:
            continue
        try:
            history = store.recent_messages(session_id)
            request_state: HarnessState = {"question": question}
            answer = await run(question, history=history, state=request_state)
            store.save_exchange(
                session_id,
                question,
                answer,
                execution_context=(
                    _execution_context(request_state)
                    if "plan" in request_state
                    else None
                ),
            )
            print(f"Assistant> {answer}")
        except Exception as exc:
            print(f"Request failed: {exc}")


def main() -> None:
    try:
        asyncio.run(chat())
    except KeyboardInterrupt:
        print("\nAI harness stopped.")


if __name__ == "__main__":
    main()