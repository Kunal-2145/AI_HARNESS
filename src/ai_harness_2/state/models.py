from typing import Literal, TypedDict


class HarnessState(TypedDict, total=False):
    question: str
    kind: Literal["chat", "task", "unsupported"]
    plan: list[str]
    selected_servers: list[str]
    answer: str