"""Week 6 — a minimal tool-using ADK agent.

Two tools, no multi-agent structure yet (that starts Week 8). Run with `adk web` from the
`adk/` directory and watch the trace panel to see the Perception-Reasoning-Action loop happen
automatically for every user turn.
"""

from google.adk.agents import Agent

COURSE_TOPICS = {
    1: "LLM fundamentals",
    2: "Context engineering I",
    3: "Context engineering II",
    4: "Retrieval-Augmented Generation (RAG)",
    6: "Agent fundamentals",
    7: "Google ADK",
    8: "Multi-agent systems",
    9: "LangGraph I",
    10: "LangGraph II + advanced RAG",
    12: "Evaluation",
    13: "Observability",
    14: "Production deployment",
    15: "Security and ethics",
}

STUDY_HOURS_TABLE = {"easy": 2, "medium": 4, "hard": 7}


def get_course_topic(week: int) -> dict:
    """Look up which topic is covered in a given week of the AI Agentic Engineering course."""
    topic = COURSE_TOPICS.get(week)
    if topic is None:
        return {"week": week, "topic": None, "note": "No lecture that week (exam/project delivery)."}
    return {"week": week, "topic": topic}


def estimate_study_hours(topic: str, difficulty: str) -> dict:
    """Estimate independent study hours for a topic given a difficulty: 'easy', 'medium', or 'hard'."""
    difficulty = difficulty.lower().strip()
    hours = STUDY_HOURS_TABLE.get(difficulty)
    if hours is None:
        return {"error": f"Unknown difficulty '{difficulty}'. Use easy, medium, or hard."}
    return {"topic": topic, "difficulty": difficulty, "hours": hours}


def recommend_next_topics(completed_weeks: list[int]) -> dict:
    """Given the weeks a student has already completed, return the next uncompleted week
    (in order) and its topic, plus the full sorted list of remaining weeks."""
    remaining = sorted(w for w in COURSE_TOPICS if w not in completed_weeks)
    if not remaining:
        return {"remaining_weeks": [], "note": "All available weeks completed."}
    next_week = remaining[0]
    return {
        "remaining_weeks": remaining,
        "next_week": next_week,
        "next_topic": COURSE_TOPICS[next_week],
    }


root_agent = Agent(
    model="gemini-flash-latest",
    name="campus_helper_agent",
    description="Answers questions about the AI Agentic Engineering course schedule and study planning.",
    instruction=(
        "You help students of the AI Agentic Engineering course plan their study time. "
        "Use get_course_topic to find out what a given week covers. Use estimate_study_hours to "
        "estimate how long a topic takes to study, guessing a reasonable difficulty (easy/medium/hard) "
        "if the user does not specify one, and stating your assumption. Use recommend_next_topics when "
        "the student tells you which weeks they've already completed and asks what to study next. "
        "Always ground week-topic claims in the tool result, never guess them yourself."
    ),
    tools=[get_course_topic, estimate_study_hours, recommend_next_topics],
)
                    