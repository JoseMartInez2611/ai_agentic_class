"""Confirms checkpointer state persistence across invoke() calls, without the interactive CLI.

Usage: python run_test.py
"""
from dotenv import load_dotenv

load_dotenv()

import app

graph = app.build_graph()

config_a = {"configurable": {"thread_id": "test-thread-A"}}

print("--- invoke 1 (thread A): schedule question ---")
r1 = graph.invoke({"question": "What does week 9 cover?"}, config=config_a)
print("result:", r1["result"])
print("get_state(thread A).values:", graph.get_state(config_a).values)

print("\n--- invoke 2 (thread A, SAME thread_id): study-time question ---")
r2 = graph.invoke({"question": "How many hours should I budget for a hard topic?"}, config=config_a)
print("result:", r2["result"])
print("get_state(thread A).values:", graph.get_state(config_a).values)

history_a = list(graph.get_state_history(config_a))
print(f"\nthread A has {len(history_a)} checkpoints persisted across the 2 invoke() calls")

config_b = {"configurable": {"thread_id": "test-thread-B"}}
print("\n--- thread B (NEW thread_id), before any invoke ---")
print("get_state(thread B).values:", graph.get_state(config_b).values)

print("\n--- invoke 3 (thread B): deliberately vague question, to exercise clarify_node ---")
r3 = graph.invoke({"question": "Tell me something about the course."}, config=config_b)
print("result:", r3["result"])
print("get_state(thread B).values:", graph.get_state(config_b).values)

print("\n--- thread A, re-checked after all of the above: still holds its own last state ---")
print("get_state(thread A).values:", graph.get_state(config_a).values)
