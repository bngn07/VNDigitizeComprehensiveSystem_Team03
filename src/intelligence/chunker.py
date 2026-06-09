from __future__ import annotations

class Chunker:

    def __init__(self, chunk_size=1000):
        self.chunk_size = chunk_size

    def chunk(self, text: str):

        chunks = []

        for i in range(0, len(text), self.chunk_size):
            chunks.append(text[i:i + self.chunk_size])

        return chunks
