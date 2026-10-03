"""Week 6 — the same campus-helper agent, built with LangGraph's prebuilt ReAct agent.

Same two tools as the ADK version in ../adk/campus_helper_agent/agent.py, so you can compare the
Perception-Reasoning-Action loop across frameworks. Run with: python app.py
"""

import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

load_dotenv()

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


def build_agent():
    llm = ChatGoogleGenerativeAI(model="gemini-flash-latest")
    return create_react_agent(
        model=llm,
        tools=[get_course_topic, estimate_study_hours, recommend_next_topics],
        prompt=(
            "You help students of the AI Agentic Engineering course plan their study time. "
            "Use get_course_topic to find out what a given week covers. Use estimate_study_hours to "
            "estimate how long a topic takes to study, guessing a reasonable difficulty "
            "(easy/medium/hard) if the user does not specify one, and stating your assumption. "
            "Use recommend_next_topics when the student tells you which weeks they've already "
            "completed and asks what to study next."
        ),
    )


def main():
    if not os.environ.get("GOOGLE_API_KEY"):
        raise SystemExit("Set GOOGLE_API_KEY before running (export GOOGLE_API_KEY=...)")

    agent = build_agent()
    print("Campus helper agent (LangGraph). Type a question, or 'quit' to exit.\n")
    while True:
        user_input = input("you> ").strip()
        if user_input.lower() in {"quit", "exit"}:
            break
        result = agent.invoke({"messages": [{"role": "user", "content": user_input}]})
        for msg in result["messages"]:
            if getattr(msg, "tool_calls", None):
                for call in msg.tool_calls:
                    print(f"  [Action] {call['name']}({call['args']})")
            if msg.__class__.__name__ == "ToolMessage":
                print(f"  [Observation] {msg.content}")
        print("agent>", result["messages"][-1].content, "\n")


if __name__ == "__main__":
    main()
