"""RAG pipeline as a LangGraph: START -> retrieve -> generate -> END."""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from openai import OpenAI

from app.config import GROQ_API_KEY, GROQ_BASE_URL, GROQ_MODEL
from app.embeddings import embed_query
from app.vector_store import search_chunks

llm = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)
CHAT_MODEL = GROQ_MODEL

TOP_K = 4
SIMILARITY_THRESHOLD = 0.40

NOT_FOUND_ANSWER = "I couldn't find this information in the provided ebook."

SYSTEM_PROMPT = """You are a question-answering assistant for the Agentic AI ebook
(Konverge AI).

Answer the user's question ONLY using the provided context from the ebook.

If the answer cannot be found in the provided context, say:
"{not_found}"

Do not use outside knowledge.
Do not guess.
Do not make up information.

Context from the ebook:
{context}
"""


class RagState(TypedDict):
    question: str
    context: list[dict]
    answer: str
    confidence: float


def retrieve(state: RagState) -> dict:
    chunks = search_chunks(embed_query(state["question"]), top_k=TOP_K)
    return {"context": chunks}


def generate(state: RagState) -> dict:
    chunks = state["context"]

    if not chunks or chunks[0]["score"] < SIMILARITY_THRESHOLD:
        return {
            "answer": NOT_FOUND_ANSWER,
            "confidence": chunks[0]["score"] if chunks else 0.0,
        }

    context_text = "\n\n".join(
        f"[Page {chunk['page']}]\n{chunk['text']}" for chunk in chunks
    )

    response = llm.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT.format(
                    not_found=NOT_FOUND_ANSWER, context=context_text
                ),
            },
            {"role": "user", "content": state["question"]},
        ],
    )
    return {
        "answer": response.choices[0].message.content.strip(),
        "confidence": chunks[0]["score"],
    }


def build_graph():
    graph = StateGraph(RagState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


rag_graph = build_graph()


def answer_question(question: str) -> dict:
    final_state: RagState = rag_graph.invoke(
        {"question": question, "context": [], "answer": "", "confidence": 0.0}
    )
    return {
        "answer": final_state["answer"],
        "confidence": final_state["confidence"],
        "sources": final_state["context"],
    }
