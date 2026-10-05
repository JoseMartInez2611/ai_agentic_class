"""Week 8 — orchestrator + specialists in LangChain, mirroring ADK's AgentTool pattern.

Each specialist is its own create_react_agent; we wrap each one as a plain callable tool for the
coordinator, so calling a specialist is isolated (it runs, returns a string, the coordinator stays
in control) — the LangChain analog of ADK's AgentTool. A true sub_agents-style handoff needs an
explicit graph with conditional routing, which is Week 9's job (LangGraph I).
Run with: python app.py
"""

import os

import sympy
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

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


def build_agent():
    llm = ChatGoogleGenerativeAI(model="gemini-flash-latest")

    research_agent = create_react_agent(
        model=llm, tools=[get_course_topic],
        prompt="Answer schedule/topic questions using get_course_topic. Be brief.",
    )
    planner_agent = create_react_agent(
        model=llm, tools=[estimate_study_hours],
        prompt="Answer study-time questions using estimate_study_hours. Be brief.",
    )
    calculus_agent = create_react_agent(
        model=llm, tools=[differentiate_expression],
        prompt="Answer derivative questions using differentiate_expression. Be brief.",
    )

    def ask_research_agent(question: str) -> str:
        """Ask the research specialist about the course schedule (e.g. what a given week covers)."""
        result = research_agent.invoke({"messages": [{"role": "user", "content": question}]})
        return result["messages"][-1].content

    def ask_planner_agent(question: str) -> str:
        """Ask the planner specialist for a study-time estimate for a topic and difficulty."""
        result = planner_agent.invoke({"messages": [{"role": "user", "content": question}]})
        return result["messages"][-1].content

    def ask_calculus_agent(question: str) -> str:
        """Ask the calculus specialist to differentiate a single-variable expression. Only for
        derivative requests, never for schedule or study-time questions."""
        result = calculus_agent.invoke({"messages": [{"role": "user", "content": question}]})
        return result["messages"][-1].content

    return create_react_agent(
        model=llm,
        tools=[ask_research_agent, ask_planner_agent, ask_calculus_agent],
        prompt=(
            "Use ask_research_agent for what-does-week-N-cover questions, ask_planner_agent for "
            "study-time estimates, and ask_calculus_agent for derivative/differentiation requests. "
            "When a question needs more than one, call each needed tool and combine the results "
            "yourself into one final answer."
        ),
    )


def main():
    if not os.environ.get("GOOGLE_API_KEY"):
        raise SystemExit("Set GOOGLE_API_KEY before running (export GOOGLE_API_KEY=...)")

    agent = build_agent()
    print("Orchestrator (LangChain, AgentTool-style). Type a question, or 'quit' to exit.\n")
    while True:
        user_input = input("you> ").strip()
        if user_input.lower() in {"quit", "exit"}:
            break
        result = agent.invoke({"messages": [{"role": "user", "content": user_input}]})
        print("agent>", result["messages"][-1].content, "\n")


if __name__ == "__main__":
    main()
