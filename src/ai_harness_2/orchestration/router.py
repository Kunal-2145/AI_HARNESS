from typing import Literal

from ai_harness_2.orchestration.planner import TaskPlan


def route_plan(plan: TaskPlan) -> Literal["chat", "task", "unsupported"]:
    if plan.kind not in {"chat", "task", "unsupported"}:
        raise ValueError(f"Unsupported plan kind: {plan.kind!r}")
    return plan.kind