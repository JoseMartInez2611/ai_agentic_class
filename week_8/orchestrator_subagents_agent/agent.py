"""Week 8 — orchestrator + specialists using ADK's sub_agents (LLM-driven handoff).

Whichever specialist the coordinator routes to takes over the conversation and replies to the
user directly. Compare with ../orchestrator_agenttool_agent/agent.py, which keeps the parent in
control. Run with `adk web` from the `adk/` directory (both agents show up in the dropdown).
"""

import sympy
from google.adk.agents import Agent

COURSE_TOPICS = {
    6: "Agent fundamentals", 7: "Google ADK", 8: "Multi-agent systems",
    9: "LangGraph I", 10: "LangGraph II + advanced RAG",
}
STUDY_HOURS_TABLE = {"easy": 2, "medium": 4, "hard": 7}


def get_course_topic(week: int) -> dict:
    """Look up which topic is covered in a given week of the AI Agentic Engineering course."""
    return {"week": week, "topic": COURSE_TOPICS.get(week, "unknown")}


def estimate_study_hours(topic: str, difficulty: str) -> dict:
    """Estimate independent study hours for a topic given a difficulty: 'easy', 'medium', or 'hard'."""
    hours = STUDY_HOURS_TABLE.get(difficulty.lower().strip())
    if hours is None:
        return {"error": f"Unknown difficulty '{difficulty}'. Use easy, medium, or hard."}
    return {"topic": topic, "difficulty": difficulty, "hours": hours}


def differentiate_expression(expression: str, variable: str = "x") -> dict:
    """Compute the symbolic derivative of a single-variable calculus expression with respect to a variable."""
    try:
        var = sympy.symbols(variable)
        derivative = sympy.diff(sympy.sympify(expression), var)
        return {"expression": expression, "variable": variable, "derivative": str(derivative)}
    except (sympy.SympifyError, TypeError, ValueError) as exc:
        return {"error": f"Could not differentiate '{expression}': {exc}"}


research_agent = Agent(
    model="gemini-flash-latest",
    name="research_agent",
    description="Looks up which topic is covered in a given week of the AI Agentic Engineering course.",
    instruction="Answer schedule/topic questions using get_course_topic. Be brief.",
    tools=[get_course_topic],
)

planner_agent = Agent(
    model="gemini-flash-latest",
    name="planner_agent",
    description="Estimates how many hours a student should budget to study a topic at a given difficulty.",
    instruction="Answer study-time questions using estimate_study_hours. Be brief.",
    tools=[estimate_study_hours],
)

calculus_agent = Agent(
    model="gemini-flash-latest",
    name="calculus_agent",
    description=(
        "Computes the symbolic derivative of a single-variable calculus expression "
        "(e.g. 'x**3 + 2*x', 'sin(x)*x', 'exp(x)/x') with respect to a given variable. "
        "Only for differentiation / derivative requests, never for schedule or study-time questions."
    ),
    instruction="Answer derivative questions using differentiate_expression. Be brief.",
    tools=[differentiate_expression],
)

root_agent = Agent(
    model="gemini-flash-latest",
    name="orchestrator_subagents_agent",
    description="Coordinator that delegates schedule, study-time, and derivative questions to specialists.",
    instruction=(
        "You are a coordinator. Delegate questions about what a week covers to research_agent. "
        "Delegate questions about how long studying a topic will take to planner_agent. "
        "Delegate questions asking to differentiate/derive a calculus expression to calculus_agent. "
        "Once you delegate, let that specialist answer directly."
    ),
    sub_agents=[research_agent, planner_agent, calculus_agent],
)
