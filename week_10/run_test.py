"""Exercises re-ranking visibility and all three interrupt() resume paths, without the
interactive CLI.

Usage: python run_test.py
"""
from dotenv import load_dotenv

load_dotenv()

import app
from langgraph.types import Command

graph = app.build_graph()


def run_once(thread_id, question):
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke({"question": question}, config=config)
    return config, result


print("=== Good question (should answer confidently, no interrupt) ===")
config, result = run_once("good-q", "What does Corte 2 cover and how much is it worth?")
print("interrupted:", "__interrupt__" in result)
print("final_answer:", result["final_answer"])
state = graph.get_state(config).values
print("chunks_before_rerank:", state["chunks_before_rerank"])
print("chunks (after rerank):", state["chunks"])
print("re-ranking changed order:", state["chunks_before_rerank"] != state["chunks"])

LOW_CONF_Q = "What is my professor's personal email address?"

print("\n=== Low-confidence question, APPROVE path ===")
config, result = run_once("low-conf-approve", LOW_CONF_Q)
print("interrupted:", "__interrupt__" in result)
if "__interrupt__" in result:
    payload = result["__interrupt__"][0].value
    print("review payload:", payload)
    result = graph.invoke(Command(resume={"action": "approve"}), config=config)
print("final_answer:", result["final_answer"])

print("\n=== Low-confidence question, EDIT path ===")
config, result = run_once("low-conf-edit", LOW_CONF_Q)
print("interrupted:", "__interrupt__" in result)
if "__interrupt__" in result:
    result = graph.invoke(
        Command(resume={"action": "edit", "text": "Please email the course coordinator instead of the professor directly."}),
        config=config,
    )
print("final_answer:", result["final_answer"])

print("\n=== Low-confidence question, REJECT path ===")
config, result = run_once("low-conf-reject", LOW_CONF_Q)
print("interrupted:", "__interrupt__" in result)
if "__interrupt__" in result:
    result = graph.invoke(Command(resume={"action": "reject"}), config=config)
print("final_answer:", result["final_answer"])
