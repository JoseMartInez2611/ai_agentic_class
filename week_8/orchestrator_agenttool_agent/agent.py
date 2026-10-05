"""Week 8 — orchestrator + specialists using ADK's AgentTool (isolated call-and-return).

The coordinator stays in control: each specialist runs in isolation and returns a string, which
the coordinator synthesizes into its own reply. Compare with ../orchestrator_subagents_agent/agent.py,
where control transfers to the child. Run with `adk web` from the `adk/` directory.
"""

from google.adk.agents import Agent
from google.adk.tools.agent_tool import AgentTool

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

root_agent = Agent(
    model="gemini-flash-latest",
    name="orchestrator_agenttool_agent",
    description="Coordinator that calls specialists as tools and combines their answers itself.",
    instruction=(
        "Use research_agent (as a tool) for what-does-week-N-cover questions, and planner_agent "
        "(as a tool) for study-time estimates. When a question needs both, call both tools and "
        "combine the results yourself into one final answer."
    ),
    tools=[AgentTool(research_agent), AgentTool(planner_agent)],
)
