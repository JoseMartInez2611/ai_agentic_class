"""Week 10 — an ADK agent whose one tool does hybrid search + LLM re-ranking before returning chunks.

LangGraph's interrupt()-based human review (content page, Section 1) is this week's dedicated
LangGraph-only topic per the syllabus — this ADK app focuses on the retrieval-quality half:
hybrid search and re-ranking, portable to either framework. Run with `adk web` from the `adk/` dir.
"""

import re

import chromadb
from google import genai
from google.adk.agents import Agent

COURSE_DOCS = """
AI Agentic Engineering is a 16-week elective course for Systems Engineering students at Universidad de
Santander. It is organized into three graded cuts called cortes. Corte 1 (weeks 1-5) covers LLM fundamentals,
context engineering, and RAG. Corte 2 (weeks 6-11) covers agents, multi-agent systems, Google ADK, and
LangGraph. Corte 3 (weeks 12-16) covers evaluation, observability, and deployment to production.

Corte 2 is worth 30% of the final grade: 25% for a multi-agent system project and 5% for in-class activities.
Students must build a multi-agent system that solves a real problem using Google ADK or LangGraph, integrating
RAG capabilities, delivered with a system diagram and a 15-minute live demonstration.

Week 10 covers LangGraph II and advanced RAG: human-in-the-loop with interrupt(), hybrid search combining
vector and keyword retrieval, re-ranking retrieved chunks, and integrating RAG into an agentic flow.
"""


def _chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    text = " ".join(text.split())
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks


_CHUNKS = _chunk_text(COURSE_DOCS)
_genai_client = genai.Client()
_collection = chromadb.Client().get_or_create_collection("week10_adk_course_docs")
if _collection.count() == 0:
    _embeddings = [
        _genai_client.models.embed_content(model="gemini-embedding-001", contents=c).embeddings[0].values
        for c in _CHUNKS
    ]
    _collection.add(documents=_CHUNKS, embeddings=_embeddings, ids=[f"c{i}" for i in range(len(_CHUNKS))])


def _keyword_search(query: str, top_k: int = 6) -> list[str]:
    q_words = set(re.findall(r"\w+", query.lower()))
    scored = [(len(q_words & set(re.findall(r"\w+", c.lower()))), c) for c in _CHUNKS]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for score, c in scored[:top_k] if score > 0]


def _reciprocal_rank_fusion(list_a: list[str], list_b: list[str], k: int = 60) -> list[str]:
    scores: dict[str, float] = {}
    for rank, item in enumerate(list_a):
        scores[item] = scores.get(item, 0) + 1 / (k + rank)
    for rank, item in enumerate(list_b):
        scores[item] = scores.get(item, 0) + 1 / (k + rank)
    return [item for item, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


def search_course_docs(query: str) -> dict:
    """Search the course documents with hybrid (vector + keyword) search and LLM re-ranking. Returns top chunks."""
    q_embedding = _genai_client.models.embed_content(model="gemini-embedding-001", contents=query).embeddings[0].values
    vector_hits = _collection.query(query_embeddings=[q_embedding], n_results=6)["documents"][0]
    kw_hits = _keyword_search(query)
    merged = _reciprocal_rank_fusion(vector_hits, kw_hits)[:6]

    rerank_prompt = (
        f"Question: {query}\n\nRank these passages from most to least relevant. "
        "Reply with only the passage numbers, most relevant first, comma-separated.\n\n"
        + "\n".join(f"{i}: {c}" for i, c in enumerate(merged))
    )
    order = _genai_client.models.generate_content(model="gemini-flash-latest", contents=rerank_prompt).text
    indices = [int(n) for n in re.findall(r"\d+", order) if int(n) < len(merged)]
    ranked = [merged[i] for i in indices] or merged
    return {"chunks": ranked[:3]}


root_agent = Agent(
    model="gemini-flash-latest",
    name="hybrid_rag_agent",
    description="Answers questions about the course using hybrid search and re-ranked retrieval.",
    instruction=(
        "Use search_course_docs to find relevant passages before answering. Answer ONLY using the "
        "returned chunks. If they don't contain the answer, say so explicitly instead of guessing."
    ),
    tools=[search_course_docs],
)
