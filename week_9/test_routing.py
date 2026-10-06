"""Plain-assert unit test for route_after_classify -- no LLM call, no graph build needed.

Usage: python test_routing.py
"""
from app import route_after_classify

ALL_BRANCHES = ["research_node", "planner_node", "both_node", "clarify_node"]


def test_route_after_classify_covers_every_branch():
    for route in ALL_BRANCHES:
        state = {"question": "irrelevant for this test", "route": route, "result": ""}
        assert route_after_classify(state) == route, f"expected {route!r}, got a different route"


if __name__ == "__main__":
    test_route_after_classify_covers_every_branch()
    print(f"OK -- route_after_classify correctly forwards all {len(ALL_BRANCHES)} branches: {ALL_BRANCHES}")
