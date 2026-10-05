"""Week 9 — the routing graph from the content page, LLM-backed, with a persistent checkpointer.

classify (LLM, structured output) -> research_node | planner_node | both_node (deterministic) -> respond (LLM) -> END

Run with: python app.py
Type /state to print the currently persisted state for this thread.
"""

import os
import re
from typing import Literal, TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

COURSE_TOPICS = {
    6: "Agent fundamentals", 7: "Google ADK", 8: "Multi-agent systems",
    9: "LangGraph I", 10: "LangGraph II + advanced RAG",
}
STUDY_HOURS_TABLE = {"easy": 2, "medium": 4, "hard": 7}


class PlannerState(TypedDict):
    question: str
    route: str
    result: str


class RouteDecision(BaseModel):
    route: Literal["research_node", "planner_node", "both_node"]


def build_graph():
    llm = ChatGoogleGenerativeAI(model="gemini-flash-latest")
    router_llm = llm.with_structured_output(RouteDecision)

    def classify_node(state: PlannerState) -> dict:
        decision = router_llm.invoke(
            "Classify the user's question about a course. Return route='research_node' if it only "
            "asks what a given week covers, 'planner_node' if it only asks about study hours, or "
            "'both_node' if it asks about both.\n\nQuestion: " + state["question"]
        )
        return {"route": decision.route}

    def route_after_classify(state: PlannerState) -> str:
        return state["route"]

    def _extract_week(question: str) -> int | None:
        m = re.search(r"week\s*(\d+)", question, re.IGNORECASE)
        return int(m.group(1)) if m else None

    def _extract_difficulty(question: str) -> str:
        for level in STUDY_HOURS_TABLE:
            if level in question.lower():
                return level
        return "medium"  # default assumption, stated explicitly in the final answer

    def research_node(state: PlannerState) -> dict:
        week = _extract_week(state["question"])
        topic = COURSE_TOPICS.get(week, "unknown") if week else "unknown"
        return {"result": f"Week {week}: {topic}"}

    def planner_node(state: PlannerState) -> dict:
        difficulty = _extract_difficulty(state["question"])
        hours = STUDY_HOURS_TABLE[difficulty]
        return {"result": f"Estimated {hours}h (assumed difficulty: {difficulty})"}

    def both_node(state: PlannerState) -> dict:
        research = research_node(state)["result"]
        plan = planner_node(state)["result"]
        return {"result": f"{research}; {plan}"}

    def respond_node(state: PlannerState) -> dict:
        reply = llm.invoke(
            f"Phrase a short, friendly final answer to '{state['question']}' using this raw data: "
            f"{state['result']}"
        )
        return {"result": reply.content}

    builder = StateGraph(PlannerState)
    for name, fn in [
        ("classify", classify_node), ("research_node", research_node),
        ("planner_node", planner_node), ("both_node", both_node), ("respond", respond_node),
    ]:
        builder.add_node(name, fn)

    builder.add_edge(START, "classify")
    builder.add_conditional_edges("classify", route_after_classify)
    for worker in ["research_node", "planner_node", "both_node"]:
        builder.add_edge(worker, "respond")
    builder.add_edge("respond", END)

    return builder.compile(checkpointer=InMemorySaver())


def main():
    if not os.environ.get("GOOGLE_API_KEY"):
        raise SystemExit("Set GOOGLE_API_KEY before running (export GOOGLE_API_KEY=...)")

    graph = build_graph()
    config = {"configurable": {"thread_id": "student-session-1"}}
    print("Routing graph (LangGraph). Type a question, '/state' to inspect persisted state, or 'quit'.\n")
    while True:
        user_input = input("you> ").strip()
        if user_input.lower() in {"quit", "exit"}:
            break
        if user_input == "/state":
            print(graph.get_state(config).values, "\n")
            continue
        result = graph.invoke({"question": user_input}, config=config)
        print("agent>", result["result"], "\n")


if __name__ == "__main__":
    main()
