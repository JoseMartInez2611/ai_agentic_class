"""Quick manual test for agent.py, without adk web.

Usage: python run_test.py "your question here"
"""
import asyncio
import sys

from dotenv import load_dotenv
from google.genai import types
from google.adk.runners import InMemoryRunner

load_dotenv()

import agent as agent_module


async def ask(agent, prompt, app_name="week6_test", user_id="student"):
    runner = InMemoryRunner(agent=agent, app_name=app_name)
    session = await runner.session_service.create_session(app_name=app_name, user_id=user_id)
    content = types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
    final_text = None
    async for event in runner.run_async(user_id=user_id, session_id=session.id, new_message=content):
        for call in event.get_function_calls():
            print(f"  [Action] {call.name}({dict(call.args)})")
        for resp in event.get_function_responses():
            print(f"  [Observation] {resp.response}")
        if event.content and event.content.parts and event.content.parts[0].text:
            final_text = event.content.parts[0].text
    return final_text


if __name__ == "__main__":
    question = sys.argv[1] if len(sys.argv) > 1 else "What does week 6 cover, and how many hours at hard difficulty?"
    print(f"you> {question}\n")
    answer = asyncio.run(ask(agent_module.root_agent, question))
    print("\nagent>", answer)
