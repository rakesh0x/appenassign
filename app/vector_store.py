"""Pinecone index setup and search."""

from pinecone import Pinecone, ServerlessSpec

from app.config import EMBEDDING_DIMENSIONS, PINECONE_API_KEY, PINECONE_INDEX_NAME

_client = Pinecone(api_key=PINECONE_API_KEY)


def get_index():
    existing = [i["name"] for i in _client.list_indexes()]
    if PINECONE_INDEX_NAME not in existing:
        _client.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSIONS,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not _client.describe_index(PINECONE_INDEX_NAME).status["ready"]:
            pass
    return _client.Index(PINECONE_INDEX_NAME)


def upsert_chunks(chunks: list[dict], vectors: list[list[float]]) -> int:
    index = get_index()
    records = [
        {
            "id": chunk["id"],
            "values": vector,
            "metadata": {
                "text": chunk["text"],
                "page": chunk["page"],
                "chunk_id": chunk["id"],
            },
        }
        for chunk, vector in zip(chunks, vectors)
    ]
    index.upsert(vectors=records)
    return len(records)


def delete_index() -> None:
    if PINECONE_INDEX_NAME in [i["name"] for i in _client.list_indexes()]:
        _client.delete_index(PINECONE_INDEX_NAME)


def search_chunks(question_vector: list[float], top_k: int = 4) -> list[dict]:
    result = get_index().query(
        vector=question_vector,
        top_k=top_k,
        include_metadata=True,
    )
    return [
        {
            "id": match["id"],
            "text": match["metadata"]["text"],
            "page": match["metadata"]["page"],
            "score": round(float(match["score"]), 4),
        }
        for match in result["matches"]
    ]
