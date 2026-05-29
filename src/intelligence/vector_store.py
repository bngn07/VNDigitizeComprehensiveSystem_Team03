from __future__ import annotations
from typing import Any
import faiss
import numpy as np


class VectorStore:
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.texts: list[str] = []
        self.metadata: list[dict[str, Any]] = []

    def add(self, embeddings: np.ndarray, texts: list[str], metadata: list[dict[str, Any]] | None = None) -> None:
        embeddings = np.array(embeddings, dtype=np.float32)
        self.index.add(embeddings)
        self.texts.extend(texts)

        if metadata is None:
            metadata = [{} for _ in texts]

        self.metadata.extend(metadata)

    def search(self, query_embedding: np.ndarray, top_k: int = 3,) -> list[dict]:
        query_embedding = np.array([query_embedding], dtype=np.float32,)
        scores, indices = self.index.search(query_embedding, top_k)
        results: list[dict] = []

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue

            results.append(
                {
                    "text": self.texts[idx],
                    "score": float(score),
                    "metadata": self.metadata[idx],
                }
            )

        return results

    def total_documents(self) -> int:
        return len(self.texts)