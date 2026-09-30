"""Turns text into vectors with a local sentence-transformers model."""

from fastembed import TextEmbedding

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_model = None


def _get_model() -> TextEmbedding:
    global _model
    if _model is None:
        _model = TextEmbedding(EMBEDDING_MODEL)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    return [vector.tolist() for vector in _get_model().embed(texts)]


def embed_query(question: str) -> list[float]:
    return embed_texts([question])[0]