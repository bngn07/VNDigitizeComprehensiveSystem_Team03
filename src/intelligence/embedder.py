from __future__ import annotations
import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL_NAME = ("paraphrase-multilingual-MiniLM-L12-v2")

class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: str | list[str], normalize_embeddings: bool = True) -> np.ndarray:
        """
        Chuyển đổi văn bản thành các embedding.
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        return embeddings

    def embedding_dimension(self) -> int:
        return self.model.get_embedding_dimension()