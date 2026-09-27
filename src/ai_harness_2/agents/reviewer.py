from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


async def review_changes(llm: ChatOpenAI, diff: str) -> str:
    response = await llm.ainvoke(
        [
            SystemMessage(
                content="Review the supplied code diff for correctness, regressions, "
                "security risks, and missing tests. Report actionable findings first."
            ),
            HumanMessage(content=diff),
        ]
    )
    return str(response.content)