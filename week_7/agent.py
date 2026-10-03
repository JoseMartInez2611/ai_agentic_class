"""Week 7 — a stateful ADK agent exercising all three real state scopes.

- user:preferred_pace   -> survives across sessions for the same user_id
- session_log           -> reset every new session
- app:total_sessions_logged_all_users -> shared across every user of this app

Run with `adk web` from the `adk/` directory.
"""

from google.adk.agents import Agent
from google.adk.tools.tool_context import ToolContext

VALID_PACES = {"relaxed", "normal", "intense"}


def set_preferred_pace(pace: str, tool_context: ToolContext) -> dict:
    """Remember the user's preferred study pace ('relaxed', 'normal', or 'intense') across sessions."""
    pace = pace.lower().strip()
    if pace not in VALID_PACES:
        return {"error": f"Unknown pace '{pace}'. Use relaxed, normal, or intense."}
    tool_context.state["user:preferred_pace"] = pace
    return {"preferred_pace": pace}


def log_study_session(topic: str, hours: float, tool_context: ToolContext) -> dict:
    """Record a completed study session for the current session only."""
    log = tool_context.state.get("session_log", [])
    log.append({"topic": topic, "hours": hours})
    tool_context.state["session_log"] = log

    total_all_users = tool_context.state.get("app:total_sessions_logged_all_users", 0) + 1
    tool_context.state["app:total_sessions_logged_all_users"] = total_all_users

    return {"logged": True, "sessions_this_conversation": len(log)}


def get_study_summary(tool_context: ToolContext) -> dict:
    """Summarize this session's logged study time, the user's saved pace preference, and the app-wide total."""
    log = tool_context.state.get("session_log", [])
    return {
        "preferred_pace": tool_context.state.get("user:preferred_pace", "not set yet"),
        "sessions_this_conversation": len(log),
        "total_hours_this_conversation": sum(item["hours"] for item in log),
        "total_sessions_logged_all_users": tool_context.state.get("app:total_sessions_logged_all_users", 0),
    }


def reset_session_log(tool_context: ToolContext) -> dict:
    """Clear only this conversation's study log, leaving user:preferred_pace (and the app-wide total) untouched."""
    tool_context.state["session_log"] = []
    return {"session_log_cleared": True}


root_agent = Agent(
    model="gemini-flash-latest",
    name="study_planner_agent",
    description="Tracks a student's study sessions and pace preference across the AI Agentic Engineering course.",
    instruction=(
        "You help a student log and review their study sessions. Use set_preferred_pace when they state a "
        "pace preference (relaxed/normal/intense). Use log_study_session whenever they mention finishing "
        "studying a topic, asking for both the topic and hours if missing. Use get_study_summary when asked "
        "for a recap. Use reset_session_log only when they explicitly ask to clear or start over the "
        "session's logged study time. Always report numbers from the tool results, never invent them."
    ),
    tools=[set_preferred_pace, log_study_session, get_study_summary, reset_session_log],
)
