"""Week 7 — a simplified analog of the ADK study_planner_agent, in LangChain.

LangChain itself has no built-in scoped session/app/user state like ADK's ToolContext.state —
that concern formally belongs to LangGraph's checkpointer (Week 9). Here we fake the three scopes
with three plain Python dicts, so the *behavior* is comparable, but read this as a simplified
preview, not a real equivalent, of ../adk/study_planner_agent/agent.py.
Run with: python app.py
"""

import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

VALID_PACES = {"relaxed", "normal", "intense"}

# Stand-ins for ADK's three real state scopes -- reset "session_state" per run, keep the others.
user_state: dict = {}          # stands in for ADK's user: scope
app_state = {"total_sessions_logged_all_users": 0}  # stands in for ADK's app: scope
session_state = {"session_log": []}                 # stands in for ADK's session-only scope


def set_preferred_pace(pace: str) -> dict:
    """Remember the user's preferred study pace ('relaxed', 'normal', or 'intense') across sessions."""
    pace = pace.lower().strip()
    if pace not in VALID_PACES:
        return {"error": f"Unknown pace '{pace}'. Use relaxed, normal, or intense."}
    user_state["preferred_pace"] = pace
    return {"preferred_pace": pace}


def log_study_session(topic: str, hours: float) -> dict:
    """Record a completed study session for the current session only."""
    session_state["session_log"].append({"topic": topic, "hours": hours})
    app_state["total_sessions_logged_all_users"] += 1  # stands in for ADK's app: scope
    return {"logged": True, "sessions_this_conversation": len(session_state["session_log"])}


def get_study_summary() -> dict:
    """Summarize this session's logged study time, the saved pace preference, and the app-wide total."""
    log = session_state["session_log"]
    return {
        "preferred_pace": user_state.get("preferred_pace", "not set yet"),
        "sessions_this_conversation": len(log),
        "total_hours_this_conversation": sum(item["hours"] for item in log),
        "total_sessions_logged_all_users": app_state["total_sessions_logged_all_users"],
    }


def reset_session_log() -> dict:
    """Clear only session_state, leaving user_state (and app_state) untouched -- stands in for ADK's
    session-scoped tool_context.state['session_log'] reset."""
    session_state["session_log"] = []
    return {"session_log_cleared": True}


def build_agent():
    llm = ChatGoogleGenerativeAI(model="gemini-flash-latest")
    return create_react_agent(
        model=llm,
        tools=[set_preferred_pace, log_study_session, get_study_summary, reset_session_log],
        prompt=(
            "You help a student log and review their study sessions. Use set_preferred_pace when they state "
            "a pace preference. Use log_study_session whenever they mention finishing studying a topic, asking "
            "for both topic and hours if missing. Use get_study_summary when asked for a recap. Use "
            "reset_session_log only when they explicitly ask to clear or start over the session's logged study time."
        ),
    )


def main():
    if not os.environ.get("GOOGLE_API_KEY"):
        raise SystemExit("Set GOOGLE_API_KEY before running (export GOOGLE_API_KEY=...)")

    agent = build_agent()
    print("Study planner agent (LangChain/LangGraph). Type a question, or 'quit' to exit.\n")
    while True:
        user_input = input("you> ").strip()
        if user_input.lower() in {"quit", "exit"}:
            break
        result = agent.invoke({"messages": [{"role": "user", "content": user_input}]})
        print("agent>", result["messages"][-1].content, "\n")


if __name__ == "__main__":
    main()
