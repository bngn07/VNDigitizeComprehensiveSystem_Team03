from __future__ import annotations

class Chunker:

    def __init__(self, chunk_size=1000):
        self.chunk_size = chunk_size

    def chunk(self, text: str):

        chunks = []

        for i in range(0, len(text), self.chunk_size):
            chunks.append(
                text[i:i + self.chunk_size]
            )

        return chunks


# def normalize_text(text: str) -> str:
#     """
#     Normalize OCR text trước khi chunking
#     """

#     if not text:
#         return ""

#     text = re.sub(r"\s+", " ", text)
#     text = text.replace("\x0c", " ")

#     return text.strip()


# def merge_blocks(blocks: Iterable[TextBlock]) -> str:
#     """
#     Merge OCR TextBlocks thành 1 text hoàn chỉnh
#     """

#     texts: list[str] = []

#     for block in blocks:
#         if not block.text:
#             continue

#         cleaned = normalize_text(block.text)

#         if cleaned:
#             texts.append(cleaned)

#     return "\n".join(texts)


# def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100,) -> list[str]:
#     """
#     Chia văn bản thành các đoạn chunk
#     """

#     text = normalize_text(text)
#     if not text:
#         return []

#     chunks: list[str] = []

#     start = 0
#     text_length = len(text)

#     while start < text_length:
#         end = start + chunk_size
#         chunk = text[start:end].strip()

#         if chunk:
#             chunks.append(chunk)
#         start += chunk_size - overlap

#     return chunks


# def build_chunks_from_ocr(ocr_result: OCRResult, chunk_size: int = 500, overlap: int = 100,) -> list[str]:
#     full_text = merge_blocks(ocr_result.texts)

#     return chunk_text(
#         full_text,
#         chunk_size=chunk_size,
#         overlap=overlap,
#     )