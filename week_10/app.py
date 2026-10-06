"""Week 10 — retrieve -> generate -> review graph: hybrid search, LLM re-ranking, and interrupt()-based
human review of low-confidence answers. Same sample course corpus as the Week 4 RAG notebook.

Run with: python app.py
"""

import os
import re
from typing import TypedDict

import chromadb
from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

COURSE_DOCS = """
AI Agentic Engineering is a 16-week elective course for Systems Engineering students at Universidad de
Santander. It is organized into three graded cuts called cortes. Corte 1 (weeks 1-5) covers LLM fundamentals,
context engineering, and RAG. Corte 2 (weeks 6-11) covers agents, multi-agent systems, Google ADK, and
LangGraph. Corte 3 (weeks 12-16) covers evaluation, observability, and deployment to production.

Corte 2 is worth 30% of the final grade: 25% for a multi-agent system project and 5% for in-class activities.
Students must build a multi-agent system that solves a real problem using Google ADK or LangGraph, integrating
RAG capabilities, delivered with a system diagram and a 15-minute live demonstration.

Week 10 covers LangGraph II and advanced RAG: human-in-the-loop with interrupt(), hybrid search combining
vector and keyword retrieval, re-ranking retrieved chunks, and integrating RAG into an agentic flow as a
graph node rather than a standalone script.

All labs in this course use free-tier tools: the Gemini API through Google AI Studio, ChromaDB and FAISS for
vector storage, and Langfuse's free tier for observability starting in Corte 3.
"""


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    text = " ".join(text.split())
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks


CHUNKS = chunk_text(COURSE_DOCS)


def keyword_search(query: str, chunks: list[str], top_k: int = 8) -> list[str]:
    """Naive keyword overlap search, standing in for BM25 — no extra dependency required."""
    q_words = set(re.findall(r"\w+", query.lower()))
    scored = []
    for c in chunks:
        c_words = set(re.findall(r"\w+", c.lower()))
        scored.append((len(q_words & c_words), c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for score, c in scored[:top_k] if score > 0]


def reciprocal_rank_fusion(list_a: list[str], list_b: list[str], k: int = 60) -> list[str]:
    scores: dict[str, float] = {}
    for rank, item in enumerate(list_a):
        scores[item] = scores.get(item, 0) + 1 / (k + rank)
    for rank, item in enumerate(list_b):
        scores[item] = scores.get(item, 0) + 1 / (k + rank)
    return [item for item, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


def _as_text(content) -> str:
    """ChatGoogleGenerativeAI's .content is sometimes a plain str and sometimes a block list
    like [{'type': 'text', 'text': '...', 'extras': {...}}] -- normalize to a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(block.get("text", "") for block in content if isinstance(block, dict))
    return str(content)


class RAGState(TypedDict):
    question: str
    chunks: list[str]
    chunks_before_rerank: list[str]
    draft_answer: str
    confidence: float
    final_answer: str


def build_graph():
    genai_client = genai.Client()  # reads GOOGLE_API_KEY from the environment
    llm = ChatGoogleGenerativeAI(model="gemini-flash-latest")

    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection("week10_course_docs")
    if collection.count() == 0:
        embeddings = [
            genai_client.models.embed_content(model="gemini-embedding-001", contents=c).embeddings[0].values
            for c in CHUNKS
        ]
        collection.add(documents=CHUNKS, embeddings=embeddings, ids=[f"c{i}" for i in range(len(CHUNKS))])

    def retrieve_node(state: RAGState) -> dict:
        q_embedding = genai_client.models.embed_content(
            model="gemini-embedding-001", contents=state["question"]
        ).embeddings[0].values
        vector_hits = collection.query(query_embeddings=[q_embedding], n_results=6)["documents"][0]
        kw_hits = keyword_search(state["question"], CHUNKS, top_k=6)
        merged = reciprocal_rank_fusion(vector_hits, kw_hits)[:6]

        rerank_prompt = (
            f"Question: {state['question']}\n\nRank these passages from most to least relevant. "
            "Reply with only the passage numbers, most relevant first, comma-separated.\n\n"
            + "\n".join(f"{i}: {c}" for i, c in enumerate(merged))
        )
        order = _as_text(llm.invoke(rerank_prompt).content)
        indices = [int(n) for n in re.findall(r"\d+", order) if int(n) < len(merged)]
        ranked = [merged[i] for i in indices] or merged
        # Kept alongside the re-ranked result so callers can verify re-ranking actually
        # changed the order, not just silently passed the fused list through unchanged.
        return {"chunks": ranked[:3], "chunks_before_rerank": merged[:3]}

    def generate_node(state: RAGState) -> dict:
        context = "\n\n---\n\n".join(state["chunks"])
        prompt = (
            f"Answer using ONLY this context. If unsure, say 'I don't have enough information'.\n\n"
            f"Context:\n{context}\n\nQuestion: {state['question']}"
        )
        answer = _as_text(llm.invoke(prompt).content)
        confidence = 0.3 if "don't have enough information" in answer.lower() else 0.9
        return {"draft_answer": answer, "confidence": confidence}

    def review_node(state: RAGState) -> dict:
        if state["confidence"] < 0.6:
            decision = interrupt({
                "question": state["question"],
                "draft_answer": state["draft_answer"],
                "ask": "approve, edit, or reject this low-confidence answer",
            })
            if decision["action"] == "edit":
                return {"final_answer": decision["text"]}
            if decision["action"] == "reject":
                return {"final_answer": "(rejected by reviewer — no answer given)"}
            return {"final_answer": state["draft_answer"]}
        return {"final_answer": state["draft_answer"]}

    builder = StateGraph(RAGState)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate", generate_node)
    builder.add_node("review", review_node)
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", "review")
    builder.add_edge("review", END)

    return builder.compile(checkpointer=InMemorySaver())


def main():
    if not os.environ.get("GOOGLE_API_KEY"):
        raise SystemExit("Set GOOGLE_API_KEY before running (export GOOGLE_API_KEY=...)")

    graph = build_graph()
    config = {"configurable": {"thread_id": "student-session-1"}}
    print("Hybrid-search + re-rank + human-review RAG graph. Type a question, or 'quit'.\n")
    while True:
        user_input = input("you> ").strip()
        if user_input.lower() in {"quit", "exit"}:
            break
        result = graph.invoke({"question": user_input}, config=config)
        if "__interrupt__" in result:
            payload = result["__interrupt__"][0].value
            print(f"[REVIEW NEEDED] draft: {payload['draft_answer']}")
            action = input("  approve / edit / reject > ").strip().lower()
            decision = {"action": action}
            if action == "edit":
                decision["text"] = input("  edited answer> ")
            result = graph.invoke(Command(resume=decision), config=config)
        print("agent>", result["final_answer"], "\n")


if __name__ == "__main__":
    main()
