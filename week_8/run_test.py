"""Quick manual test for the ADK agents, without adk web.

Usage:
  python run_test.py subagents "your question"
  python run_test.py agenttool "your question"
"""
import asyncio
import sys

from dotenv import load_dotenv
from google.genai import types
from google.adk.runners import InMemoryRunner

load_dotenv()

from orchestrator_subagents_agent import agent as subagents_agent
from orchestrator_agenttool_agent import agent as agenttool_agent

VARIANTS = {
    "subagents": subagents_agent.root_agent,
    "agenttool": agenttool_agent.root_agent,
}


async def ask(agent, prompt, app_name="week8_test", user_id="student"):
    runner = InMemoryRunner(agent=agent, app_name=app_name)
    session = await runner.session_service.create_session(app_name=app_name, user_id=user_id)
    content = types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
    final_text = None
    final_author = None
    async for event in runner.run_async(user_id=user_id, session_id=session.id, new_message=content):
        for call in event.get_function_calls():
            print(f"  [{event.author}] Action: {call.name}({dict(call.args)})")
        for resp in event.get_function_responses():
            print(f"  [{event.author}] Observation: {resp.response}")
        if event.content and event.content.parts and event.content.parts[0].text:
            final_text = event.content.parts[0].text
            final_author = event.author
    return final_author, final_text


if __name__ == "__main__":
    variant = sys.argv[1] if len(sys.argv) > 1 else "subagents"
    question = sys.argv[2] if len(sys.argv) > 2 else "What does week 8 cover, and how many study hours should I budget if it's hard?"
    print(f"you> {question}\n")
    author, answer = asyncio.run(ask(VARIANTS[variant], question))
    print(f"\n[{author}] agent> {answer}")
