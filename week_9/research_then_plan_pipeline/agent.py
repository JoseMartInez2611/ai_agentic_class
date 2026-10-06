"""Week 9 — ADK's deterministic-pipeline counterpart to the LangGraph routing graph.

SequentialAgent always runs both specialists, in a fixed order — no conditional branching. This is
an honest limitation to see directly: ADK's workflow agents (SequentialAgent/ParallelAgent/LoopAgent)
give you deterministic *order*, but not the conditional routing a LangGraph graph gives you for free.
Run with `adk web` from the `adk/` directory.
"""

from google.adk.agents import Agent, SequentialAgent

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
    instruction="Look up the week's topic with get_course_topic and report it briefly.",
    tools=[get_course_topic],
    output_key="research_result",  # written to session.state["research_result"] for the next agent
)

planner_agent = Agent(
    model="gemini-flash-latest",
    name="planner_agent",
    instruction=(
        "The prior step's result is in state key 'research_result': {research_result}. "
        "Now estimate study hours for that topic (assume medium difficulty unless told otherwise) "
        "using estimate_study_hours, and give one combined final answer covering both the topic and the hours."
    ),
    tools=[estimate_study_hours],
)

root_agent = SequentialAgent(
    name="research_then_plan_pipeline",
    description="Always looks up the week's topic, then estimates study hours for it, in that fixed order.",
    sub_agents=[research_agent, planner_agent],
)
